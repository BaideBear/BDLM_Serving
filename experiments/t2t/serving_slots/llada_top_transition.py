"""Experimental LLaDA2.1 B32/B64 executor driven by external, per-forward offers.

Use the same raw eval model and rectangular-mask adapter as DualBlockKernel.
Create one request, then call step(state, 32 or 64), or run(state, lookup).
Pure prompt prefill is outside the offer clock. Commit and reencode each cost
one offer and one real forward. Only mandatory refresh can exceed Q32;
Q64 is always a physical ceiling.
"""

from dataclasses import dataclass

import torch

from bdlm_serving.methods.t2t.llada_dual import BLOCK, PAIR, DualBlockKernel, DualRequestState
from bdlm_serving.methods.t2t.llada_sparse import _SelectedCache


@dataclass
class TopRequestState(DualRequestState):
    offer_tick: int = 0
    previous_offer: int | None = None


@dataclass(frozen=True)
class OfferTick:
    tick: int
    block: int
    phase: str
    committed_end: int
    tail_resident: bool


class TopTransitionKernel(DualBlockKernel):
    """One rolling canvas with a parked tail and optional fused tail growth.

    Q32 projects only dirty tail rows and never samples them. Other tail KV
    and the tail's sampling history stay resident unchanged. All writes,
    including edits back to MASK, must be projected on the next forward.
    At promotion the old prefix is genuinely different: keep tokens, histories
    and clocks, but invalidate resident KV and restart post-step verification.

    When a new tail opens at Q64, fuse its bootstrap with the head's denoise.
    ``fuse_growth=False`` is the split-forward control on the same offer clock;
    it bootstraps the new tail without sampling the resident head, but still
    refreshes any pending head writes in the same forward.
    """

    def __init__(self, model, *, fuse_growth=True):
        super().__init__(model, q_budget=PAIR, mask_slots=56, policy="hard_frontier")
        self.fuse_growth = fuse_growth

    def create(self, input_ids, config, gen_length, seed, mask_id, eos_ids, cached=True):
        if gen_length <= 0:
            raise ValueError("Expected a positive generation length")
        state = super().create(input_ids, config, gen_length, seed, mask_id, eos_ids, cached)
        return TopRequestState(**vars(state))

    @staticmethod
    def _resident_end(state):
        return state.resident[0][0].shape[-2] if state.resident is not None else 0

    def _tail_resident(self, state):
        return self._resident_end(state) == (state.block + 2) * BLOCK

    def _prefill(self, state):
        if state.phase == "prefill":
            super().step(state)
            state.phase = "denoise"

    def _selected_forward(self, state, picked, visible_end, *, phase="denoise",
                          stage="selected", store=False):
        positions = torch.tensor(picked, dtype=torch.long, device=state.x.device)
        # Scatter into the existing storage, preserving all unprojected rows.
        resident = state.resident
        if visible_end != self._resident_end(state):
            resident = tuple((k[..., :visible_end, :], v[..., :visible_end, :])
                             for k, v in resident)
        cache = _SelectedCache(resident, positions, state.committed_end, visible_end)
        return self._execute(state, positions, visible_end, cache, phase, stage, store)

    def _select(self, state, active, visible_end, tail_refresh=()):
        picked, reserved, width, saturated, full_two_block = self._dual_pick(state, active)
        self._prepare_hard_metadata(state, active)
        # The parked tail's positions are after all sampled head positions.
        logits = self._selected_forward(state, picked + list(tail_refresh), visible_end)
        rows = [[i for i, p in enumerate(picked) if p // BLOCK == b] for b in active]
        local = [[picked[i] % BLOCK for i in block_rows] for block_rows in rows]
        self._apply_pass(state, logits[:, :len(picked)], active, local, rows, reserved, width,
                         saturated, full_two_block)
        state.events[-1]["refresh_only_positions"] = list(tail_refresh)

    def _split_growth(self, state, head_end, head_pending):
        if not head_pending:
            self._bootstrap_right(state)
            return
        # A new tail needs all B32 slots. Every new slot is filled by the
        # scatter before attention at each layer; untouched head rows survive.
        state.resident = tuple(tuple(torch.nn.functional.pad(t, (0, 0, 0, BLOCK))
                                     for t in pair) for pair in state.resident)
        tail = state.block + 1
        end = head_end + BLOCK
        self._prepare_hard_metadata(state, [tail])
        logits = self._selected_forward(state, head_pending + list(range(head_end, end)),
                                        end, stage="bootstrap")
        self._apply_pass(state, logits[:, -BLOCK:], [tail], active_width=BLOCK,
                         saturated=True, full_two_block=False)
        state.events[-1]["refresh_only_positions"] = head_pending

    def _denoise(self, state, q, pending):
        head = state.block
        start, head_end = head * BLOCK, (head + 1) * BLOCK
        wants_tail = q == PAIR and self._has_right(state)
        has_tail = self._tail_resident(state)
        if state.resident is None:
            self._execute_full(state, start, head_end + (BLOCK if wants_tail else 0),
                               "bootstrap")
            return "pair_bootstrap" if wants_tail else "head_bootstrap"
        if wants_tail and not has_tail:
            if self.fuse_growth:
                self._execute_full(state, start, head_end + BLOCK, "bootstrap")
                return "fused_growth"
            self._split_growth(state, head_end, sorted(p for p in pending if p < head_end))
            return "split_growth"
        if wants_tail and not state.right_done and not self._limit_reached(state, head + 1):
            self._select(state, [head, head + 1], head_end + BLOCK)
            return "pair_denoise"
        tail_refresh = sorted(p for p in pending if p >= head_end)
        if tail_refresh and not has_tail:
            raise RuntimeError("Pending tail refresh requires resident tail KV")
        self._select(state, [head], head_end + (BLOCK if tail_refresh else 0), tail_refresh)
        return "head_denoise"

    def _commit_with_tail_refresh(self, state, tail_refresh):
        head = state.block
        start, end = head * BLOCK, (head + 1) * BLOCK
        if bool((state.x[:, start:end] == state.mask_id).any()):
            raise RuntimeError("Refusing to commit a block containing MASK")
        self._prepare_hard_metadata(state, [head])
        self._selected_forward(state, list(range(start, end)) + tail_refresh, end + BLOCK,
                               phase="commit", stage="commit", store=True)
        # The head cannot attend to the tail. Append only the finalized head
        # KV; the refreshed tail is speculative even on a terminal commit.
        for layer, (k, v) in enumerate(state.resident):
            state.cache.update(k[..., start:end, :], v[..., start:end, :], layer)
        if state.cache.get_seq_length() != end:
            raise RuntimeError("Commit did not store the final block KV")
        state.committed_end = end
        counts = dict(accepted=0, edited=0, masked_before=0, masked_after=0,
                      selected_masked_before=0)
        state.events[-1].update(
            **counts, block_counts={str(head): counts}, step_in_block=state.step_in_block,
            post_steps=max(state.left_post_steps, state.right_post_steps), blocks=[head],
            pair=False, active_width=BLOCK, effective_q=BLOCK, saturated=True,
            full_two_block=False, refresh_only_positions=tail_refresh,
        )

    def _commit_head(self, state, pending):
        head = state.block
        resident_tail = self._tail_resident(state)
        tail_refresh = sorted(p for p in pending if p >= (head + 1) * BLOCK)
        if tail_refresh:
            if not resident_tail:
                raise RuntimeError("Pending tail refresh requires resident tail KV")
            self._commit_with_tail_refresh(state, tail_refresh)
        else:
            self._commit_block(state, head)
        state.context_version += 1
        if self._eos_or_end(state, head):
            state.phase = "done"
        elif resident_tail:
            self._advance_one(state)
            state.left_post_steps = 0
            state.left_done = False
            state.phase = "reencode"
        else:
            self._advance_one(state)
            state.phase = "denoise"
        state.resident = None
        state.history = None  # transient combined view, not either block's history
        state.step_in_block = 0

    @torch.inference_mode()
    def step(self, state, q):
        """Consume one integer offer; return its audited event (None at EOS/end)."""
        if type(q) is not int or q not in (BLOCK, PAIR):
            raise ValueError("LLaDA top transition expects an integer offer of 32 or 64")
        if not isinstance(state, TopRequestState) or state.model_kind != self.kind:
            raise ValueError("Request/model mismatch")
        if state.phase == "done":
            return None
        self._prefill(state)
        self.last_logits = None
        start = state.block * BLOCK
        # Includes edits back to MASK, which are not in History.fresh.
        pending = {start + i for i, (w, r) in enumerate(zip(state.last_write, state.last_query))
                   if w >= 0 and w >= r}
        pending_fresh = 0
        for block, history in ((state.block, state.left_history),
                               (state.block + 1, state.right_history)):
            if history is not None:
                pending_fresh += sum(block * BLOCK + i in pending for i in history.fresh)
        if state.committed_end != start or state.cache.get_seq_length() != start:
            raise RuntimeError("Committed prefix does not end at head start")
        if state.phase == "reencode":
            self._execute_full(state, start, start + BLOCK, "reencode")
            state.phase = "commit" if state.left_done else "denoise"
            action = "promote_verify"
        elif state.phase == "commit" or self._limit_reached(state, state.block):
            self._commit_head(state, pending)
            action = "commit_head"
        elif state.phase == "denoise":
            action = self._denoise(state, q, pending)
            if state.left_done:
                state.phase = "commit"
        else:
            raise RuntimeError(f"Unexpected top transition phase: {state.phase}")
        event = state.events[-1]
        physical_q = event["physical_q"]
        actual = set(event["absolute_positions"])
        refresh_only = event.setdefault("refresh_only_positions", [])
        if not pending <= actual or not set(refresh_only) <= pending:
            raise RuntimeError("Generation did not project exactly the required refresh-only rows")
        overflow = max(0, physical_q - q)
        if not 0 < physical_q <= PAIR or overflow > len(refresh_only):
            raise RuntimeError("Generation exceeded its offer except for mandatory refresh")
        event.update(q_budget=q, nominal_q=q, overflow=overflow, offer_tick=state.offer_tick,
                     previous_offer=state.previous_offer, top_action=action,
                     mandatory_fresh=pending_fresh, refreshed_pending=sorted(pending),
                     sampled_q=0 if event["phase"] == "commit" else physical_q - len(refresh_only))
        state.offer_tick += 1
        state.previous_offer = q
        return event

    def run(self, state, next_budget, max_actions=8192):
        """Query the scheduler exactly once per generation forward, until done."""
        for _ in range(max_actions):
            if state.phase == "done":
                return state
            q = next_budget(OfferTick(state.offer_tick, state.block, state.phase,
                                      state.committed_end, self._tail_resident(state)))
            self.step(state, q)
        if state.phase == "done":
            return state
        raise RuntimeError("Top transition action limit exceeded")

    @staticmethod
    def output(state):
        if state.phase != "done":
            raise RuntimeError("Output is not final; a resident tail is still speculative")
        return DualBlockKernel.output(state)

"""Five external physical-Q offers on one rolling LLaDA TopTransition request.

An offer pays for one generation forward (including bootstrap, verification and
commit). Prompt-only prefill is separate. Offers 32/64 use the original routes;
4/8/16 can underfill and use mandatory writes before discretionary rows. The
native rectangular-mask adapter and audited QKV path are required, just as for
TopTransitionKernel. Do not share the raw model with concurrent forwards.
"""

from collections.abc import Mapping
from random import Random

import torch

from experiments.t2t.serving_slots.llada_top_transition import (
    BLOCK,
    PAIR,
    TopRequestState,
    TopTransitionKernel,
)


LOW_OFFERS = (4, 8, 16)
OFFERS = (*LOW_OFFERS, BLOCK, PAIR)
# Absolute row ranges are derived from the pre-forward head start, not from
# measured Q. Split growth densely initializes only the new tail.
_DENSE_ACTIONS = {
    "head_bootstrap": ("bootstrap", 0, BLOCK),
    "pair_bootstrap": ("bootstrap", 0, PAIR),
    "fused_growth": ("bootstrap", 0, PAIR),
    "split_growth": ("bootstrap", BLOCK, PAIR),
    "head_verify": ("verify", 0, BLOCK),
    "promote_verify": ("promotion_verify", 0, BLOCK),
    "commit_head": ("commit", 0, BLOCK),
}


class AllSlotKernel(TopTransitionKernel):
    """Optional prefix_slots maps low offers to nonnegative row quotas.

    A prefix starts at the physical B32 boundary and ends at the first MASK.
    Prompt and pending rows cannot be optional. Sparse selection and optional
    refresh are approximate; neither claims equivalence to a dense forward.
    """

    def __init__(self, model, *, prefix_slots=None, prefix_mode="zero", fuse_growth=True):
        if prefix_mode not in ("zero", "right", "random"):
            raise ValueError("Expected prefix_mode zero, right or random")
        if prefix_slots is not None and not isinstance(prefix_slots, Mapping):
            raise ValueError("prefix_slots must map offers to row quotas")
        quotas = {}
        for offer, quota in (prefix_slots or {}).items():
            if isinstance(offer, str) and offer in map(str, OFFERS):
                offer = int(offer)
            if (type(offer) is not int or offer not in OFFERS or offer in quotas
                    or type(quota) is not int or not 0 <= quota <= offer
                    or (quota and (offer not in LOW_OFFERS or prefix_mode == "zero"))):
                raise ValueError("Nonzero prefix quotas require a low offer and right/random mode")
            quotas[offer] = quota
        super().__init__(model, fuse_growth=fuse_growth)
        self.prefix_mode = prefix_mode
        self.prefix_slots = quotas

    def _low_denoise(self, state, q, pending):
        head = state.block
        start, end = head * BLOCK, (head + 1) * BLOCK
        head_pending = sorted(p for p in pending if p < end)
        tail_pending = sorted(p for p in pending if p >= end)
        if tail_pending and not self._tail_resident(state):
            raise RuntimeError("Pending tail refresh requires resident tail KV")
        visible_end = end + (BLOCK if tail_pending else 0)
        editable = [start + i for i in self._editable(state, head)]
        tokens = state.x[0, start:end].tolist()
        masks = [p for p in editable if tokens[p - start] == state.mask_id]

        if not masks:
            # A no-MASK convergence decision requires full-head verification.
            # Scatter in place to retain a parked tail, even with no tail queries.
            positions = list(range(start, end))
            self._prepare_hard_metadata(state, [head])
            logits = self._selected_forward(state, positions + tail_pending, visible_end,
                                            stage="verify")
            self._apply_pass(state, logits[:, :BLOCK], [head], active_width=BLOCK,
                             saturated=True, full_two_block=False)
            state.events[-1]["refresh_only_positions"] = tail_pending
            return "head_verify"

        prefix_end = masks[0]
        room = max(0, q - len(pending))
        mask_rows = [p for p in masks if p not in pending][:room]
        room -= len(mask_rows)
        backfill = [p for p in editable if p >= prefix_end and p not in pending
                    and tokens[p - start] != state.mask_id][:room]
        zero = sorted(head_pending + mask_rows + backfill)
        capacity = max(0, q - len(zero) - len(tail_pending))
        pool = [p for p in editable if p < prefix_end and p not in pending]
        quota = self.prefix_slots.get(q, 0)
        count = min(quota, capacity, len(pool)) if zero else 0
        optional = []
        if count and self.prefix_mode == "right":
            optional = pool[-count:]
        elif count and self.prefix_mode == "random":
            rank = pool.copy()
            Random(f"physical-prefix:{state.selection_seed}:{start}:{state.offer_tick}").shuffle(rank)
            optional = sorted(rank[:count])

        picked = sorted(zero + optional)
        self._prepare_hard_metadata(state, [head] if picked else [])
        if not (picked or tail_pending):
            raise RuntimeError("A masked head must have a query or mandatory tail work")
        logits = self._selected_forward(state, picked + tail_pending, visible_end)
        if picked:
            self._apply_pass(state, logits[:, :len(picked)], [head],
                             [[p - start for p in picked]], [list(range(len(picked)))],
                             optional, len(editable), len(picked) == len(editable), False)
        else:
            # Tail debt consumed the offer: scatter only, no head pass or RNG draw.
            state.events[-1].update(
                accepted=0, edited=0, masked_before=0, masked_after=0,
                selected_masked_before=0, block_counts={}, blocks=[], pair=False,
                active_width=0, effective_q=0, saturated=False, full_two_block=False,
                step_in_block=state.step_in_block,
                post_steps=max(state.left_post_steps, state.right_post_steps),
            )
        state.events[-1].update(
            refresh_only_positions=tail_pending,
            prefix_plan=dict(
                mode=self.prefix_mode, quota=quota, prefix_end=prefix_end,
                eligible_prefix_rows=pool, zero_rows=zero, zero_mask_rows=mask_rows,
                zero_backfill_rows=backfill, slack=capacity, added_rows=optional,
                evicted_rows=[], selected_rows=picked,
            ),
        )
        return "head_denoise" if picked else "tail_dirty_drain"

    @torch.inference_mode()
    def step(self, state, q):
        """Consume exactly one offered generation forward, or return None at end."""
        if type(q) is not int or q not in OFFERS:
            raise ValueError(f"LLaDA all slots expects an integer offer in {OFFERS}")
        if not isinstance(state, TopRequestState) or state.model_kind != self.kind:
            raise ValueError("Request/model mismatch")
        if state.phase == "done":
            return None
        self._prefill(state)
        self.last_logits = None
        start = state.block * BLOCK
        head_end = start + BLOCK
        if state.committed_end != start or state.cache.get_seq_length() != start:
            raise RuntimeError("Committed prefix does not end at head start")
        # Equal write/query clocks mean the write happened *after* that query.
        # History.fresh omits revocations to MASK; the clocks include them.
        pending = {start + i for i, (w, r) in enumerate(zip(state.last_write, state.last_query))
                   if w >= 0 and w >= r}
        pending_fresh = set()
        for offset, history in ((0, state.left_history), (BLOCK, state.right_history)):
            if history is not None:
                pending_fresh.update(start + offset + i for i in history.fresh)
        pending_fresh &= pending

        events_before = len(state.events)
        if state.phase == "reencode":
            self._execute_full(state, start, head_end, "reencode")
            state.phase = "commit" if state.left_done else "denoise"
            action = "promote_verify"
        elif state.phase == "commit" or self._limit_reached(state, state.block):
            self._commit_head(state, pending)
            action = "commit_head"
        elif state.phase == "denoise":
            if q in LOW_OFFERS and state.resident is not None:
                action = self._low_denoise(state, q, pending)
            else:
                action = self._denoise(state, q, pending)
            if state.left_done:
                state.phase = "commit"
        else:
            raise RuntimeError(f"Unexpected all-slot phase: {state.phase}")

        if len(state.events) != events_before + 1:
            raise RuntimeError("An offer must consume exactly one generation forward")
        event = state.events[-1]
        actual = event["absolute_positions"]
        actual_set = set(actual)
        scatter_only = event.setdefault("refresh_only_positions", [])
        scatter_set = set(scatter_only)
        sampled = [] if event["phase"] == "commit" else sorted(actual_set - scatter_set)
        dense_reason, lo, hi = _DENSE_ACTIONS.get(action, (None, 0, 0))
        dense_rows = set(range(start + lo, start + hi))
        physical_q = event["physical_q"]
        overflow = max(0, physical_q - q)
        if (actual != sorted(actual_set) or len(actual) != physical_q
                or event["F"] != 1 or not 0 < physical_q <= PAIR
                or not pending <= actual_set or not scatter_set <= pending
                or (dense_reason and actual_set != dense_rows | scatter_set)
                or not dense_rows.isdisjoint(scatter_set)):
            raise RuntimeError("Invalid audited generation positions or missing required refresh")
        if q in LOW_OFFERS:
            if physical_q > max(q, len(dense_rows | pending)):
                raise RuntimeError("Low offer exceeded without mandatory physical work")
        elif overflow > len(scatter_only):
            raise RuntimeError("Top offer exceeded except for mandatory scatter")

        # Split overflow into non-overlapping causes; rows covered by a dense
        # protocol operation are charged there, even if also pending.
        dense_overflow = max(0, len(dense_rows) - q)
        reasons = {}
        if dense_overflow:
            reasons[dense_reason] = dense_overflow
        if overflow > dense_overflow:
            reasons["mandatory_refresh"] = overflow - dense_overflow
        event.update(
            q_budget=q, nominal_q=q, overflow=overflow, overflow_reasons=reasons,
            underfill=max(0, q - physical_q), offer_tick=state.offer_tick,
            previous_offer=state.previous_offer, top_action=action,
            mandatory_fresh=len(pending_fresh), mandatory_fresh_positions=sorted(pending_fresh),
            refreshed_pending=sorted(pending), mandatory_q=len(pending),
            mandatory_head_positions=sorted(p for p in pending if p < head_end),
            mandatory_tail_positions=sorted(p for p in pending if p >= head_end),
            dense_reason=dense_reason, dense_required_q=len(dense_rows),
            dense_required_positions=sorted(dense_rows),
            sampled_q=len(sampled), sampled_positions=sampled,
        )
        state.offer_tick += 1
        state.previous_offer = q
        return event

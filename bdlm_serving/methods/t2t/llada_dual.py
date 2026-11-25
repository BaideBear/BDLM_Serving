"""Two-B32 rolling LLaDA2.1 hard-refresh runner.

This module is deliberately separate from ``llada_sparse``: the single-block
engine and its policies stay untouched.  The dual runner reuses the native
``_execute``/``_sample`` path, the rectangular block-causal mask, the audited
cache wrappers, ``History`` and ``frontier.select_hard``.

A pair is two consecutive B32 blocks.  The left block may see only earlier
blocks and itself; the right block may also see the left block.  The left
block is never allowed to see the right block.  Q is the sum of the two block
selections, so the maximum physical Q is 64.  At nominal Q64 the active width
is therefore a saturated selection (``effective_q == active_width``), not a
sparse regime.  Prompts that end inside the left B32 block reduce the active
width below 64; the runner records that cap and never fabricates 64 rows.

Commit schedule:

1. pair bootstrap: full forward over both blocks, resident KV retained;
2. selected/verify passes over the pair;
3. once the left block has no MASK and no edit was applied, commit it with a
   store=True forward;
4. always refresh the right block with the new committed prefix (reencode,
   store=False, native accept/edit rules).  The committed left changes the
   right block's prefix, so its old no-MASK pass count and done flag are reset:
   the reencode is the first verification pass in the new context and the
   max_post budget restarts;
5. if the refreshed right block is done, commit it and advance by two;
   otherwise it becomes the next left block and a new right block is
   bootstrapped with visible left KV.

The left block is never committed while it still has a MASK or while the last
pass edited it.  The right block is always refreshed after a left commit, so
it cannot be committed against stale left KV.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from transformers.cache_utils import DynamicCache

from bdlm_serving.methods.t2t.frontier import select_hard
from bdlm_serving.methods.t2t.llada_sparse import (
    LLaDASparseKernel, _AuditedDenseCache, _SelectedCache)
from bdlm_serving.methods.t2t.planner import History
from bdlm_serving.runtime.batch import ModelFacade
from bdlm_serving.runtime.serial import RequestState, SerialKernel


BLOCK = 32
PAIR = 2 * BLOCK
HARD = ("hard_frontier", "hard_random", "hard_noedit")


@dataclass
class DualRequestState(RequestState):
    history: object = None          # combined 0..63 fresh proxy for _execute
    left_history: object = None
    right_history: object = None
    resident: object = None         # per-layer (k, v) through the pair end
    last_write: object = None       # combined pair-local 0..63
    last_query: object = None       # combined pair-local 0..63
    selection_seed: int = 0
    left_post_steps: int = 0
    right_post_steps: int = 0
    left_done: bool = False
    right_done: bool = False


class _AppendCache(DynamicCache):
    """Full pass over a new right block, appending to resident prefix KV."""

    def __init__(self, resident, start, end):
        super().__init__()
        self.resident = list(resident)
        self.start = start
        self.visible_end = end
        self.writes = []
        for key, value in self.resident:
            if key.shape[-2] != start + BLOCK or value.shape[-2] != start + BLOCK:
                raise RuntimeError("Resident prefix is not at the expected start")
            if end - start != 2 * BLOCK:
                raise RuntimeError("A right bootstrap must cover left plus one new B32")

    def get_seq_length(self, layer_idx=0):
        return self.start + BLOCK

    def update(self, key, value, layer_idx, cache_kwargs=None):
        if (key.shape[-2] != value.shape[-2]
                or key.shape[-2] != BLOCK
                or layer_idx != len(self.writes)):
            raise RuntimeError("Unexpected append KV rows/order at layer %d" % layer_idx)
        old_key, old_value = self.resident[layer_idx]
        key = torch.cat([old_key, key], dim=-2)
        value = torch.cat([old_value, value], dim=-2)
        if key.shape[-2] != self.visible_end or value.shape[-2] != self.visible_end:
            raise RuntimeError("Appended resident length disagrees with visible end")
        self.resident[layer_idx] = (key, value)
        self.writes.append((layer_idx, BLOCK, BLOCK))
        return key, value

    def resident_tuple(self):
        return tuple(self.resident)


class DualBlockKernel(LLaDASparseKernel):
    """Rolling two-block hard-refresh runner with Q64 saturation accounting."""

    def __init__(self, model, q_budget=64, fresh_slots=0, mask_slots=0,
                 policy="hard_noedit", signal="uncertain", selection_seed=None):
        if isinstance(model, ModelFacade) or model.training:
            raise ValueError("Pass the raw eval LLaDA2.1 model, not a facade")
        if q_budget != PAIR:
            raise ValueError("This dual runner is the exact Q64 two-B32 runner")
        if not (type(q_budget) is int and type(fresh_slots) is int and type(mask_slots) is int):
            raise ValueError("Dual quotas must be integers")
        if min(fresh_slots, mask_slots) < 0 or fresh_slots + mask_slots > q_budget:
            raise ValueError("Invalid dual quota split")
        if fresh_slots != 0:
            raise ValueError("Hard dual policies require fresh_slots=0")
        if policy not in HARD:
            raise ValueError("Dual runner supports hard policies only")
        edit = q_budget - fresh_slots - mask_slots
        if policy == "hard_noedit" and edit != 0:
            raise ValueError("hard_noedit requires zero reserved edit seats")
        if policy != "hard_noedit" and edit == 0:
            raise ValueError("hard_frontier/hard_random require at least one edit seat")
        SerialKernel.__init__(self, model, "llada21")
        self.projections = tuple(layer.attention.query_key_value for layer in model.model.layers)
        if not self.projections:
            raise ValueError("No native LLaDA2.1 QKV projections")
        self.q_budget, self.fresh_slots, self.mask_slots = q_budget, fresh_slots, mask_slots
        self.policy, self.signal, self.selection_seed = policy, signal, selection_seed
        self.pair_width = PAIR

    def create(self, input_ids, config, gen_length, seed, mask_id, eos_ids, cached=True):
        if not cached or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Dual runner requires cached batch-one input")
        if bool((input_ids == mask_id).any()):
            raise ValueError("Prompt cannot contain the generation mask ID")
        base = SerialKernel.create(self, input_ids, config, gen_length, seed,
                                   mask_id, eos_ids, cached=True)
        return DualRequestState(
            **vars(base), selection_seed=seed if self.selection_seed is None else self.selection_seed,
            last_write=[-1] * PAIR, last_query=[-1] * PAIR)

    def _has_left(self, state):
        return state.block * BLOCK < state.x.shape[1]

    def _has_right(self, state):
        return (state.block + 1) * BLOCK < state.x.shape[1]

    def _history(self, state, block):
        return state.left_history if block == state.block else state.right_history

    def _set_history(self, state, block, history):
        if block == state.block:
            state.left_history = history
        else:
            state.right_history = history

    def _prepare_hard_metadata(self, state, blocks):
        fresh = set()
        for block in blocks:
            history = self._history(state, block)
            if history is not None:
                offset = 0 if block == state.block else BLOCK
                fresh.update(offset + i for i in history.fresh)
        state.history = SimpleNamespace(fresh=fresh)

    def _editable(self, state, block):
        start = block * BLOCK
        return list(range(max(0, state.prompt_length - start), BLOCK))

    def _post_steps(self, state, block):
        return state.left_post_steps if block == state.block else state.right_post_steps

    def _set_post_steps(self, state, block, value):
        if block == state.block:
            state.left_post_steps = value
        else:
            state.right_post_steps = value

    def _set_done(self, state, block, value):
        if block == state.block:
            state.left_done = value
        else:
            state.right_done = value

    def _apply_block(self, state, block, local, rows, proposed, confidence, event):
        start = block * BLOCK
        cur = state.x[:, start:start + BLOCK]
        before = cur[0].tolist()
        resolved = [token != state.mask_id for token in before]
        history = self._history(state, block)
        if history is None:
            history = History(resolved, [None] * BLOCK, [0] * BLOCK, set(), set())

        local_index = torch.tensor(local, dtype=torch.long, device=state.x.device)
        row_index = torch.tensor(rows, dtype=torch.long, device=state.x.device)
        old = cur.index_select(1, local_index)
        prop = proposed.index_select(1, row_index)
        conf = confidence.index_select(1, row_index)
        masked = old == state.mask_id
        threshold = state.config.get("confidence_threshold", state.config.get("threshold"))
        accept = masked & (conf > threshold)
        if bool(masked.any()) and not bool(accept.any()):
            order = conf.masked_fill(~masked, -torch.inf)
            accept[0, torch.topk(order[0], k=1).indices] = True
        absolute = torch.tensor([start + i for i in local], dtype=torch.long,
                                device=state.x.device)
        edit = (~masked) & (absolute[None] >= state.prompt_length) \
            & (prop != old) & (conf > state.config["editing_threshold"])
        cur.index_copy_(1, local_index, torch.where(accept | edit, prop, old))

        after = cur[0].tolist()
        changed = {i for i in local if before[i] != after[i]}
        now_resolved = [token != state.mask_id for token in after]
        fresh = {i for i in changed if now_resolved[i]}
        strict = {i for i in fresh if resolved[i]}
        clock = event["hard_clock"]
        offset = 0 if block == state.block else BLOCK
        for i in changed:
            state.last_write[offset + i] = clock

        masked_before = resolved.count(False)
        masked_after = now_resolved.count(False)
        edited = int(edit.sum())
        post_steps = self._post_steps(state, block)
        if masked_before == 0:
            post_steps += 1
        # Native semantics: a pass that starts with MASK is not a post-step and
        # does not reset the counter. New blocks start at zero; a committed
        # prefix change explicitly resets the right block before reencode.
        self._set_post_steps(state, block, post_steps)
        max_post = state.config.get("max_post_steps", 16)
        if masked_before > 0 or masked_after > 0:
            # A MASK-filling pass still needs a following no-MASK pass.  A
            # pass that writes a MASK is also not final.
            done = False
        else:
            # No MASK at the start and none after: this is a verification
            # pass.  Commit on no edits, or on the native post-step cap.
            done = (edited == 0) or (post_steps >= max_post)
        self._set_done(state, block, done)

        streak = [history.streak[i] + 1 if now_resolved[i] and i not in fresh else 0
                  for i in range(BLOCK)]
        history = History(now_resolved, history.certainty, streak, changed, fresh)
        self._set_history(state, block, history)
        return dict(
            strict=[start + i for i in sorted(strict)],
            masked_before=masked_before,
            masked_after=masked_after,
            selected_masked_before=int(masked.sum()),
            accepted=int(accept.sum()),
            edited=edited,
        )

    def _apply_pass(self, state, logits, blocks, picked_by_block=None,
                    rows_by_block=None, reserved=None, active_width=None,
                    saturated=None, full_two_block=None):
        proposed, confidence = self._sample(state, logits)
        if not bool(torch.isfinite(confidence).all()):
            raise RuntimeError("Nonfinite confidence in dual pass")
        event = state.events[-1]
        totals = dict(masked_before=0, masked_after=0, selected_masked_before=0,
                      accepted=0, edited=0)
        per_block = {}
        strict = []
        for index, block in enumerate(blocks):
            if picked_by_block is None:
                local = list(range(BLOCK))
                rows = list(range(index * BLOCK, (index + 1) * BLOCK))
            else:
                local = picked_by_block[index]
                rows = rows_by_block[index]
            result = self._apply_block(state, block, local, rows, proposed, confidence, event)
            strict.extend(result["strict"])
            per_block[str(block)] = dict(
                accepted=result["accepted"],
                edited=result["edited"],
                masked_before=result["masked_before"],
                masked_after=result["masked_after"],
                selected_masked_before=result["selected_masked_before"],
            )
            for key in totals:
                totals[key] += result[key]
        event.update(
            hard_strict_edits=len(strict),
            reserved_extra_edit_positions=sorted(reserved or []),
            reserved_strict_edit_positions=[i for i in (reserved or []) if i in set(strict)],
            masked_before=totals["masked_before"],
            masked_after=totals["masked_after"],
            selected_masked_before=totals["selected_masked_before"],
            accepted=totals["accepted"],
            edited=totals["edited"],
            block_counts=per_block,
            step_in_block=state.step_in_block,
            post_steps=max(state.left_post_steps, state.right_post_steps),
            blocks=list(blocks),
            pair=len(blocks) == PAIR // BLOCK,
            active_width=active_width,
            effective_q=int(logits.shape[1]),
            saturated=bool(saturated),
            full_two_block=bool(full_two_block),
            nominal_q=self.q_budget,
            overflow=max(0, int(logits.shape[1]) - self.q_budget),
        )
        state.step_in_block += 1
        return event

    def _execute_full(self, state, start, end, stage):
        q = end - start
        positions = torch.arange(start, end, device=state.x.device)
        temp = DynamicCache.from_legacy_cache(state.cache.to_legacy_cache())
        audit = _AuditedDenseCache(temp, q, end)
        blocks = list(range(start // BLOCK, end // BLOCK))
        self._prepare_hard_metadata(state, blocks)
        logits = self._execute(state, positions, end, audit, "denoise", stage, False)
        state.resident = temp.to_legacy_cache()
        self._apply_pass(state, logits, blocks, active_width=q, saturated=True,
                         full_two_block=(q == PAIR))
        return blocks

    def _bootstrap_pair(self, state):
        start = state.block * BLOCK
        end = min((state.block + PAIR // BLOCK) * BLOCK, state.x.shape[1])
        self._execute_full(state, start, end, "bootstrap")

    def _selected_pass(self, state, picked, reserved, active_width, saturated, full_two_block):
        positions = torch.tensor(picked, dtype=torch.long, device=state.x.device)
        end = (state.block + PAIR // BLOCK) * BLOCK if self._has_right(state) \
            else (state.block + 1) * BLOCK
        if state.resident[0][0].shape[-2] != end:
            raise RuntimeError("Resident KV length disagrees with pair visible end")
        row_of = {position: index for index, position in enumerate(picked)}
        blocks, picked_by_block, rows_by_block = [], [], []
        for block in (state.block, state.block + 1):
            if block > state.block + 1:
                break
            start = block * BLOCK
            local = [position - start for position in picked
                     if start <= position < start + BLOCK]
            if not local:
                continue
            blocks.append(block)
            picked_by_block.append(local)
            rows_by_block.append([row_of[start + i] for i in local])
        cache = _SelectedCache(state.resident, positions, state.committed_end, end)
        self._prepare_hard_metadata(state, blocks)
        logits = self._execute(state, positions, end, cache, "denoise", "selected", False)
        self._apply_pass(state, logits, blocks, picked_by_block, rows_by_block,
                         reserved=reserved, active_width=active_width,
                         saturated=saturated, full_two_block=full_two_block)

    def _dual_pick(self, state, active):
        clock = len(state.events) + 1
        edit_total = self.q_budget - self.fresh_slots - self.mask_slots
        if self.policy == "hard_noedit":
            edit_total = 0
        if len(active) == 2:
            edits = [edit_total // 2, edit_total - edit_total // 2]
        else:
            edits = [edit_total]
        if self.policy != "hard_noedit" and min(edits) < 1:
            raise ValueError("frontier needs at least one edit seat per active block")
        picked, reserved, active_width = [], [], 0
        for index, block in enumerate(active):
            start = block * BLOCK
            editable = self._editable(state, block)
            resolved = [token != state.mask_id
                        for token in state.x[0, start:start + BLOCK].tolist()]
            history = self._history(state, block)
            fresh = set(history.fresh) if history is not None else set()
            fresh &= set(editable)
            offset = 0 if block == state.block else BLOCK
            last_write = state.last_write[offset:offset + BLOCK]
            last_query = state.last_query[offset:offset + BLOCK]
            block_picked, block_reserved = select_hard(
                editable, resolved, fresh, last_write, last_query, clock,
                BLOCK, min(edits[index], BLOCK), self.policy,
                state.selection_seed + start)
            block_picked, block_reserved = list(block_picked), list(block_reserved)
            if (block_picked != sorted(set(block_picked))
                    or not set(block_picked).issubset(editable)
                    or not fresh.issubset(block_picked)
                    or not set(block_reserved).issubset(set(block_picked) - fresh)):
                raise RuntimeError("select_hard returned invalid dual positions")
            picked.extend(start + i for i in block_picked)
            reserved.extend(start + i for i in block_reserved)
            active_width += len(editable)
        picked, reserved = sorted(picked), sorted(reserved)
        effective_q = len(picked)
        saturated = effective_q == active_width
        full_two_block = active_width == PAIR
        return picked, reserved, active_width, saturated, full_two_block

    def _commit_block(self, state, block):
        start = block * BLOCK
        end = start + BLOCK
        if bool((state.x[:, start:end] == state.mask_id).any()):
            raise RuntimeError("Refusing to commit a block containing MASK")
        positions = torch.arange(start, end, device=state.x.device)
        audit = _AuditedDenseCache(state.cache, BLOCK, end)
        self._prepare_hard_metadata(state, [block])
        self._execute(state, positions, end, audit, "commit", "commit", True)
        if state.cache.get_seq_length() != end:
            raise RuntimeError("Commit did not store the final block KV")
        state.committed_end = end
        event = state.events[-1]
        event.update(
            edited=0,
            accepted=0,
            masked_before=0,
            masked_after=0,
            selected_masked_before=0,
            hard_strict_edits=0,
            reserved_extra_edit_positions=[],
            reserved_strict_edit_positions=[],
            block_counts={str(block): dict(accepted=0, edited=0, masked_before=0,
                                           masked_after=0, selected_masked_before=0)},
            step_in_block=state.step_in_block,
            post_steps=max(state.left_post_steps, state.right_post_steps),
            blocks=[block],
            pair=False,
            active_width=BLOCK,
            effective_q=BLOCK,
            saturated=True,
            full_two_block=False,
            nominal_q=self.q_budget,
            overflow=0,
        )

    def _reencode_right(self, state):
        # The committed left changes the right block's prefix.  Its old
        # no-MASK pass count and done flag belong to the old prefix; reset both
        # so the reencode is the first verification pass in the new context
        # and max_post restarts.
        state.right_post_steps = 0
        state.right_done = False
        block = state.block + 1
        start = block * BLOCK
        end = start + BLOCK
        self._execute_full(state, start, end, "reencode")

    def _bootstrap_right(self, state):
        block = state.block + 1
        state.right_post_steps = 0
        state.right_done = False
        start = block * BLOCK
        end = start + BLOCK
        cache = _AppendCache(state.resident, state.committed_end, end)
        positions = torch.arange(start, end, device=state.x.device)
        self._prepare_hard_metadata(state, [block])
        logits = self._execute(state, positions, end, cache, "denoise", "bootstrap", False)
        state.resident = cache.resident_tuple()
        self._apply_pass(state, logits, [block], active_width=BLOCK,
                         saturated=True, full_two_block=False)

    def _advance_one(self, state):
        old_right = state.block + 1
        if state.committed_end != old_right * BLOCK:
            raise RuntimeError("Left commit / right shift boundary mismatch")
        state.block = old_right
        state.left_history = state.right_history
        state.right_history = None
        state.left_post_steps = state.right_post_steps
        state.right_post_steps = 0
        state.left_done = state.right_done
        state.right_done = False
        state.last_write = state.last_write[BLOCK:PAIR] + [-1] * BLOCK
        state.last_query = state.last_query[BLOCK:PAIR] + [-1] * BLOCK

    def _limit_reached(self, state, block):
        start = block * BLOCK
        if bool((state.x[:, start:start + BLOCK] == state.mask_id).any()):
            return False
        return self._post_steps(state, block) >= state.config.get("max_post_steps", 16)

    def _advance_two(self, state):
        state.block += 2
        state.committed_end = state.block * BLOCK
        state.left_history = state.right_history = None
        state.left_post_steps = state.right_post_steps = 0
        state.left_done = state.right_done = False
        state.last_write = [-1] * PAIR
        state.last_query = [-1] * PAIR
        state.resident = None

    def _eos_or_end(self, state, block):
        end = (block + 1) * BLOCK
        generated = state.x[0, state.prompt_length:min(end, state.x.shape[1])].tolist()
        eos = any(token in state.eos_ids for token in generated)
        return eos or end >= state.x.shape[1]

    @torch.inference_mode()
    def step(self, state):
        if not isinstance(state, DualRequestState) or state.model_kind != self.kind:
            raise ValueError("Request/model mismatch")
        self.last_logits = None
        while True:
            if state.phase == "done":
                return "done"
            if state.phase == "prefill":
                state.phase = "bootstrap"
                state.committed_end = state.block * BLOCK
                if state.committed_end:
                    positions = torch.arange(0, state.committed_end, device=state.x.device)
                    audit = _AuditedDenseCache(state.cache, state.committed_end,
                                               state.committed_end)
                    old_query = state.last_query
                    state.last_query = None
                    self._execute(state, positions, state.committed_end, audit,
                                  "prefill", "prefill", True)
                    state.last_query = old_query
                    event = state.events[-1]
                    event.update(
                        hard_clock=len(state.events),
                        mandatory_fresh=0,
                        reserved_extra_edit_positions=[],
                        reserved_strict_edit_positions=[],
                        overflow=0,
                        hard_strict_edits=0,
                        blocks=[state.block],
                        pair=False,
                        active_width=state.committed_end,
                        effective_q=state.committed_end,
                        saturated=True,
                        full_two_block=False,
                        nominal_q=self.q_budget,
                    )
                return "prefill"
            if state.phase == "bootstrap":
                self._bootstrap_pair(state)
                state.phase = "denoise"
                return "denoise"
            if state.phase == "denoise":
                if self._limit_reached(state, state.block):
                    state.left_done = True
                if self._has_right(state) and self._limit_reached(state, state.block + 1):
                    state.right_done = True
                if state.left_done:
                    state.phase = "commit_left"
                    continue
                active = [state.block]
                if self._has_right(state) and not state.right_done:
                    active.append(state.block + 1)
                picked, reserved, active_width, saturated, full_two_block = \
                    self._dual_pick(state, active)
                if not picked:
                    state.phase = "commit_left"
                    continue
                self._selected_pass(state, picked, reserved, active_width,
                                    saturated, full_two_block)
                if state.left_done:
                    state.phase = "commit_left"
                return "denoise"
            if state.phase == "commit_left":
                self._commit_block(state, state.block)
                if self._eos_or_end(state, state.block) or not self._has_right(state):
                    state.phase = "done"
                else:
                    state.phase = "reencode_right"
                return "commit"
            if state.phase == "reencode_right":
                self._reencode_right(state)
                if state.right_done:
                    state.phase = "commit_right"
                else:
                    self._advance_one(state)
                    state.phase = "bootstrap_right" if self._has_right(state) else "denoise"
                return "reencode"
            if state.phase == "bootstrap_right":
                self._bootstrap_right(state)
                state.phase = "denoise"
                return "denoise"
            if state.phase == "commit_right":
                self._commit_block(state, state.block + 1)
                if self._eos_or_end(state, state.block + 1):
                    state.phase = "done"
                else:
                    self._advance_two(state)
                    state.phase = "bootstrap" if self._has_left(state) else "done"
                return "commit"
            raise RuntimeError("Unknown dual phase: %s" % state.phase)

    def run(self, state, max_actions=8192):
        for _ in range(max_actions):
            if state.phase == "done":
                return state
            self.step(state)
        raise RuntimeError("Dual action limit exceeded")

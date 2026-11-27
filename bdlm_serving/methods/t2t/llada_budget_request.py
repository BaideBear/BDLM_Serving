"""Batch-one B32 LLaDA2.1, one shared choose() and one measured forward per offer.

Install install_llada_rectangular_mask(model, allow_selected_cache=True) once
on the raw eval model. Construction pays one separately logged pure-prompt
prefill. Generation never rounds the output cap up or calls a dense verifier,
commit, promotion, or uncharged bootstrap. Not native dense KV/convergence.
"""
from dataclasses import replace
from types import SimpleNamespace

import torch

from bdlm_serving.methods.optimus.upstream import SelectedResidentCache
from bdlm_serving.methods.t2t.budget_family import Law, Plan, choose
from bdlm_serving.methods.t2t.llada_sparse import LLaDASparseKernel

B = 32


def _interleave(left, right, first):
    sides = (left, right) if not first else (right, left)
    return [side[i] for i in range(max(len(left), len(right)))
            for side in sides if i < len(side)]


def _joint_plan(q, split, dirty, masks, prefix, cold, law, first):
    """Use choose's seat law with scheduler ranks, never absolute-id priority."""
    candidates = [sorted((p for p in masks-dirty if (p >= split) == side),
                         key=lambda p: (p not in cold, p)) for side in (False, True)]
    ranked = _interleave(*candidates, first)
    fronts = _interleave([p for p in prefix if p < split], [p for p in prefix if p >= split], first)
    position = sorted(dirty) + ranked + fronts
    rank = {p:i for i,p in enumerate(position)}
    picked = choose(q, B, (rank[p] for p in dirty), (rank[p] for p in masks),
                    (rank[p] for p in fronts), law=law)
    def absolute(indices):
        return tuple(sorted(position[i] for i in indices))
    selected = absolute(picked.mask)
    following = int(position[picked.mask[-1]] < split) if picked.mask else first
    return Plan(absolute(picked.rows), absolute(picked.dirty), tuple(p for p in selected if p in cold),
                tuple(p for p in selected if p not in cold), absolute(picked.prefix),
                picked.drain_only), following


class _Bank(SelectedResidentCache):
    """Use upstream absolute K/V scatter, but keep all initialized keys visible.

    New positions must extend a contiguous high-water mark. Don't publish it
    until the framework's per-layer projection AND cache-write audit succeeds.
    """
    def begin(self, positions, rows):
        self.writes = []
        self.selected = positions
        end = max(self.last_end, rows[-1]+1)
        if end > self.capacity or tuple(r for r in rows if r >= self.last_end) != tuple(range(self.last_end, end)):
            raise RuntimeError("Projection would expose an uninitialized KV hole")
        self.visible_end = end

    def update(self, key, value, layer_idx, cache_kwargs=None):
        w = self.selected.numel()
        if (layer_idx != len(self.writes) or key.shape[-2] != w or value.shape[-2] != w):
            raise RuntimeError("Missing, widened or out-of-order layer write")
        self.writes.append((layer_idx, w, w))
        return super().update(key, value, layer_idx, cache_kwargs)


class LLaDABudgetRequest:
    """Absolute positions throughout. records excludes prefill; tokens is the retired suffix.

    In the legacy family q<=B offers one block and q>B offers up to two.
    In the registered ablation 2B always selects the full live window;
    below 2B the arm controls early opening. Debt from
    parked blocks is global. Cold MASKs are sampled immediately; on mixed
    cold calls other rows only refresh. Warm calls sample the selected masks,
    editable prefix and active dirty rows in blocks still containing MASKs.
    No sample on a pure drain. Right edge is ranked within each block, head
    first. Retirement needs complete physical residency and zero token debt;
    stale peers/deeper-layer KV are not claimed to be dense-equivalent.

    ``cross`` is a second, independent switch used by the C64/C16/C8 race.
    At ``q == 2B`` it registers when the next physical block may enter the
    eligible window: ``off`` never, ``eager`` as soon as the offer exists,
    ``half``/``quarter`` only after the head block is resident and its
    remaining MASK share is <= 1/2 or <= 1/4. At ``q < 2B`` the only legal
    mode is ``head``: the head plan is byte-identical to the legacy family,
    while ``borrow`` spends residual seats on a read-only contiguous tail
    build. Borrowed rows are projected and cached but never sampled, so the
    tail cannot advance the answer before it becomes the head block.

    ``coverage`` is a second independent axis on the fixed C64 race, evaluated
    strictly against ``sparse`` (the original measured family). ``sparse`` keeps
    the original ``choose`` plan: roughly ``1/8`` of leftover seats to clean
    block-internal prefix, and the head block is not guaranteed to be fully
    projected. ``head`` forces every valid row of the current head block into
    the physical Q set at ``q == 2B`` (the FlowBlock endpoint), then lets the
    same seat law choose positions in the opened second block from the residual
    budget. It is only legal for the four fixed two-block gates; ``off`` therefore
    projects the full head block alone. Below ``2B`` the arm stays sparse and
    byte-identical to the legacy family. ``switchable`` may select the same
    ``coverage="head"`` registration with ``top_gate`` standing in for ``cross``;
    at ``q == 2B`` it is plan-equivalent to the fixed arm, while lower offers
    stay on the sparse head plan.

    ``switchable`` is a third, independent registration for one request whose
    offer changes between calls. ``top_gate`` is used only at ``q == 2B`` and
    ``lower_borrow`` only below it; both defaults are inert (``quarter`` /
    ``False``). A downshift never closes or drops KV: the tail is parked in the
    bank, dirty debt stays global, and an upshift reuses every resident row.
    Pure drains refresh debt only and never advance ``sample_clock``.

    ``warm_head_sample`` is an opt-in single-factor arm for ``coverage="head"``.
    In a full-head forward whose cold rows are all in the next block, with warm
    unresolved head MASKs and global debt payable in this call, it adds those
    already-projected head MASKs to ``sampled`` alongside the tail cold. It
    selects no extra Q, no clean prefix and no dirty row; the default is False
    and the old control keeps its exact plan and record schema.
    """
    @torch.inference_mode()
    def __init__(self, model, input_ids, *, mask_id, eos_ids=(), seed=0,
                 max_new_tokens=2048, law=Law(), confidence_threshold=.9, editing_threshold=.5,
                 tail_rule="eager", ablation=None, cross=None, borrow=False,
                 switchable=False, top_gate="quarter", lower_borrow=False,
                 coverage="sparse", warm_head_sample=False, close_policy="unchanged",
                 audit_admission=False, admission_scope="global", borrow_at_half=False):
        if type(audit_admission) is not bool:
            raise ValueError("Expected a boolean sample receipt option")
        self.audit_admission = audit_admission
        if admission_scope not in ("global","warm_head") or (admission_scope=="warm_head" and
                (confidence_threshold!=.9 or editing_threshold!=.5)):
            raise ValueError("Warm-head admission requires the registered 0.9/0.5 control")
        self.admission_scope = admission_scope
        if type(borrow_at_half) is not bool or (borrow_at_half and (not switchable or lower_borrow)):
            raise ValueError("Half-block borrowing is a distinct switchable policy")
        self.borrow_at_half = borrow_at_half
        if close_policy not in ("unchanged","full_write","edit_once") or (close_policy!="unchanged" and not (
                (cross=="head" and not borrow and not switchable and ablation is None)
                or (switchable and ablation is None))):
            raise ValueError("Unsupported terminal policy for this request")
        self.close_policy, self.close_attempted = close_policy, None
        if (model.training or input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1
                or input_ids.dtype != torch.long or input_ids.device != model.get_input_embeddings().weight.device
                or type(max_new_tokens) is not int or max_new_tokens < 1 or type(seed) is not int
                or type(mask_id) is not int or not isinstance(law, Law)
                or tail_rule not in ("eager","defer_tail","fill_head")
                or ablation not in (None, "00", "10", "11")
                or (ablation is not None and tail_rule != "eager")
                or cross not in (None, "off", "eager", "half", "quarter", "head")
                or type(borrow) is not bool
                or (cross is not None and (ablation is not None or tail_rule != "eager"))
                or type(switchable) is not bool
                or top_gate not in ("off", "eager", "half", "quarter")
                or type(lower_borrow) is not bool
                or (switchable and (cross is not None or ablation is not None or tail_rule != "eager"))
                or coverage not in ("sparse", "head")
                or (coverage == "head" and (ablation is not None or
                                            (not switchable and cross not in
                                             ("off", "eager", "half", "quarter"))))
                or type(warm_head_sample) is not bool
                or (warm_head_sample and coverage != "head")
                or not (0 <= confidence_threshold <= 1 and 0 <= editing_threshold <= 1)):
            raise ValueError("Expected a raw eval model, unpadded batch-one IDs, and valid protocol")
        self.model, self.law, self.seed = model, law, seed  # Native temperature zero uses no RNG.
        self.tail_rule = tail_rule
        self.ablation = ablation  # None preserves the original registered family/pilots.
        self.cross, self.borrow = cross, borrow
        self.coverage = coverage
        self.switchable, self.top_gate, self.lower_borrow = switchable, top_gate, lower_borrow
        self.warm_head_sample = warm_head_sample
        self._cross_evidence = None
        self._switch_evidence = None
        self.pending_build = None
        self.next_side = 0
        vocab = model.get_input_embeddings().weight.shape[0]
        self.mask_id, self.eos_ids = mask_id, frozenset(eos_ids)
        if (not 0 <= mask_id < vocab or mask_id in self.eos_ids
                or any(type(t) is not int or not 0 <= t < vocab for t in self.eos_ids)):
            raise ValueError("Invalid MASK/EOS vocabulary IDs")
        self.threshold, self.edit_threshold = confidence_threshold, editing_threshold
        self.origin = input_ids.shape[1]
        self.end = self.origin + max_new_tokens
        self.x = torch.cat((input_ids, input_ids.new_full((1,max_new_tokens), mask_id)), dim=1)
        self.head = self.origin // B * B
        self.bank = _Bank(self.end)
        # _execute only: gather -> embedding -> block-causal forward -> per-layer proof.
        # Never call the old scheduler's create/step, bootstrap or commit paths.
        self.engine = LLaDASparseKernel(model, q_budget=B)
        self.records, self.tokens, self.dirty = [], [], set()
        self.status, self.stopping, self.sample_clock = 'ready', False, 0
        self._stall = 0
        self.prefill_record, _ = self._project(tuple(range(self.origin)), self.origin, 'prefill')

    def _project(self, rows, cap, phase):
        if not 0 < len(rows) <= cap or rows != tuple(sorted(set(rows))) or rows[0] < 0:
            raise RuntimeError("Invalid physical Q or positions")
        pos = self.x.new_tensor(rows)
        self.bank.begin(pos, rows)
        state = SimpleNamespace(x=self.x, events=[], block=self.head//B,
                                committed_end=self.head, context_version=self.head//B)
        logits = self.engine._execute(state, pos, self.bank.visible_end, self.bank, phase,
                                      'prefill' if phase == 'prefill' else 'selected', True, budget=cap)
        event, = state.events
        self.bank.initialized.update(rows)
        return event, logits

    def _inputs(self, q):
        masks, prefix, eligible = set(), [], set()
        if not self.stopping:
            for block in range(self.head, min(self.end, self.head+((q+B-1)//B)*B), B):
                lo, hi = max(block,self.origin), min(block+B,self.end)
                block_masks = {lo+i for i,t in enumerate(self.x[0,lo:hi].tolist()) if t == self.mask_id}
                if block_masks:
                    masks.update(block_masks)
                    eligible.update(range(lo,hi))
                    prefix.extend(r for r in range(min(block_masks)-1, lo-1, -1)
                                  if r not in self.dirty and r in self.bank.initialized)
        return masks, prefix, eligible

    def _ablation_plan(self, q):
        """Return a single physical call; only build_sample may exceed the offer."""
        split = self.head+B
        lo = max(self.head, self.origin)
        dirty = self.dirty
        def drain():
            plan = choose(q, B, dirty, (), (), law=self.law)
            return plan, (), "drain", q

        if self.stopping:
            return drain()
        head_ids = self.x[0,lo:min(split,self.end)].tolist()
        has_next = split < self.end and self.eos_ids.isdisjoint(head_ids)
        if q == 2*B:
            # Check the current offer before either arm bit, even after a down/upshift.
            self.pending_build = None
            end = min(self.end, split+B if has_next else split)
            masks, prefix, eligible = self._inputs(2*B if has_next else B)
            rows = tuple(sorted(set(range(lo, end)) | dirty))
            cold = masks-self.bank.initialized
            plan = Plan(rows, tuple(sorted(dirty)), tuple(sorted(cold)),
                        tuple(sorted(masks-cold-dirty)), tuple(sorted(prefix)), not eligible)
            return plan, tuple(p for p in rows if p in eligible), "dense_joint", q

        gate = (has_next and self.bank.last_end >= split
                and 4*head_ids.count(self.mask_id) <= len(head_ids))
        # An opening survives revocations, but not an EOS in its predecessor.
        joint = self.ablation != "00" and has_next and (gate or self.bank.last_end > split)
        if self.ablation == "11":
            if self.pending_build == split and not has_next:
                self.pending_build = None
            if joint and self.bank.last_end == split:
                self.pending_build = split
            if self.pending_build is not None:
                if dirty:
                    return drain()
                start = self.pending_build
                if self.bank.last_end != start or start not in (self.head, split):
                    raise RuntimeError("Invalid pending whole-block opening")
                rows = tuple(range(start, min(start+B, self.end)))
                self.pending_build = None
                return Plan(rows, (), rows, (), (), False), rows, "build_sample", len(rows)

        masks, prefix, eligible = self._inputs(2*B if joint else B)
        cold = masks-self.bank.initialized
        if joint:
            plan, self.next_side = _joint_plan(q, split, dirty, masks, prefix, cold, self.law,
                                               self.next_side)
            sampled = tuple(p for p in plan.rows if p in eligible)
        else:
            plan = choose(q, B, dirty, masks, prefix, cold=cold, law=self.law)
            sampled = plan.cold or tuple(p for p in plan.rows if p in eligible)
        sampled = () if plan.drain_only else sampled
        return plan, sampled, "sample" if sampled else "drain", q

    def _tail_open(self, split, head_ids, head_masks, gate=None):
        """Whether the next physical block may enter this offer's eligible window."""
        gate = self.cross if gate is None else gate
        if gate in (None, "off", "head"):
            return False
        if self.bank.last_end > split:
            return True  # An opening survives revocations, never an EOS in its predecessor.
        if gate == "eager":
            return True
        if self.bank.last_end < split:
            return False  # Need a resident, hole-free head block before a gated opening.
        remaining, width = len(head_masks), len(head_ids)
        if gate == "half":
            return 2 * remaining <= width
        if gate == "quarter":
            return 4 * remaining <= width
        raise RuntimeError("Unknown registered cross mode")

    def _terminal_head_plan(self, q, head_rows, head_masks):
        if (q != B or getattr(self, "close_policy", "unchanged") == "unchanged" or head_masks
                or not self.dirty or self.close_attempted == self.head):
            return None
        if not (self.dirty <= set(head_rows) <= self.bank.initialized):
            return None
        self.close_attempted = self.head
        edit = self.close_policy == "edit_once"
        plan = Plan(head_rows, tuple(sorted(self.dirty)), (), (), (), not edit)
        return plan, head_rows if edit else (), "close_edit_once" if edit else "close_full_write", q

    def _cross_plan(self, q):
        """One physical call for the registered cross-block race; never over-offer."""
        split = self.head + B
        lo = max(self.head, self.origin)
        dirty = self.dirty
        self._cross_evidence = None
        if self.stopping:
            plan = choose(q, B, dirty, (), (), law=self.law)
            return plan, (), "drain", q
        head_ids = self.x[0, lo:min(split, self.end)].tolist()
        has_next = split < self.end and self.eos_ids.isdisjoint(head_ids)
        head_masks = {lo + i for i, t in enumerate(head_ids) if t == self.mask_id}

        if getattr(self,"close_policy","unchanged")!="unchanged":
            if q!=B:
                raise ValueError("Terminal row experiment is registered only for fixed B")
            close = self._terminal_head_plan(q, tuple(range(lo, lo+len(head_ids))), head_masks)
            if close is not None:
                self._cross_evidence = dict(head_masks=0,opened=False,borrowed=0)
                return close

        if q == 2 * B:
            opened = has_next and self._tail_open(split, head_ids, head_masks)
            if self.coverage == "head":
                return self._head_full_plan(lo, split, q, opened, head_masks)
            masks, prefix, eligible = self._inputs(2 * B if opened else B)
            cold = masks - self.bank.initialized
            plan = choose(q, B, dirty, masks, prefix, cold=cold, law=self.law)
            sampled = () if plan.drain_only else (plan.cold or tuple(r for r in plan.rows if r in eligible))
            if len(plan.rows) > q:
                raise RuntimeError("Cross plan exceeds the physical offer")
            self._cross_evidence = dict(head_masks=len(head_masks), opened=opened, borrowed=0)
            return plan, sampled, ("sample" if sampled else "drain"), q

        if self.coverage == "head":
            raise RuntimeError("Full head coverage is only registered at the two-block offer")
        masks, prefix, eligible = self._inputs(B)  # Same head plan in both borrow arms.
        cold = masks - self.bank.initialized
        plan = choose(q, B, dirty, masks, prefix, cold=cold, law=self.law)
        sampled = () if plan.drain_only else (plan.cold or tuple(r for r in plan.rows if r in eligible))
        rows, borrowed = plan.rows, ()
        if self.borrow and has_next and len(rows) < q and not any(r >= split for r in dirty):
            start, limit = self.bank.last_end, min(split + B, self.end)
            if start >= split and start < limit:
                borrowed = tuple(range(start, min(start + (q - len(rows)), limit)))
                rows = tuple(sorted(set(rows) | set(borrowed)))
        self._cross_evidence = dict(head_masks=len(head_masks), opened=False, borrowed=len(borrowed))
        return replace(plan, rows=rows), sampled, ("sample" if sampled else "drain"), q

    def _head_full_plan(self, lo, split, q, opened, head_masks):
        """FlowBlock endpoint: project every valid head row, then select the tail.

        The head block is fixed and fully resident in this call. Residual seats
        (``q`` minus the head width) go through the same ``choose`` seat law over
        the opened second block. Mandatory dirty rows in an already resident tail
        are always refreshed, even when the tail is no longer eligible to sample
        (for example an EOS in the head); they stay read-only there.
        """
        dirty = self.dirty
        head_rows = tuple(range(lo, min(split, self.end)))
        rem = q - len(head_rows)
        if rem < 0:
            raise RuntimeError("Head block alone exceeds the physical offer")
        masks, prefix, eligible = self._inputs(2 * B if opened else B)
        cold = masks - self.bank.initialized
        tlo, thi = split, min(split + B, self.end)
        tail_dirty = tuple(sorted(p for p in dirty if tlo <= p < thi))
        tail_rows, tail_cold = (), ()
        if tlo < thi and rem:
            tail_set = set(range(tlo, thi))
            if opened:
                tail_plan = choose(min(rem, thi - tlo), thi - tlo, tail_dirty,
                                   tuple(p for p in masks if p in tail_set),
                                   tuple(p for p in prefix if p in tail_set),
                                   cold=tuple(p for p in cold if p in tail_set), law=self.law)
                tail_rows = tail_plan.rows
                tail_cold = tuple(p for p in tail_rows if p in cold)
            else:
                tail_rows = tail_dirty[:rem]
        rows = tuple(sorted(set(head_rows) | set(tail_rows)))
        if len(rows) > q:
            raise RuntimeError("Head-full plan exceeds the physical offer")
        row_set = set(rows)
        cold_sel = tuple(sorted(p for p in rows if p in cold))
        dirty_sel = tuple(sorted(p for p in rows if p in dirty))
        mask_sel = tuple(sorted(p for p in rows if p in masks and p not in cold and p not in dirty))
        prefix_sel = tuple(sorted(p for p in rows if p in set(prefix) and p not in dirty))
        drain = len(dirty) >= q
        sampled = () if drain else (cold_sel or tuple(p for p in rows if p in eligible))
        if not sampled:
            drain = True
        warm_added = ()
        if getattr(self, "warm_head_sample", False) and not drain:
            # Only already-selected warm head MASKs; no extra Q, prefix or dirty.
            head_warm = tuple(p for p in mask_sel if p < split)
            debt_payable = set(dirty) <= set(rows)
            if cold_sel and all(p >= split for p in cold_sel) and head_warm and debt_payable:
                warm_added = head_warm
                sampled = tuple(sorted(set(sampled) | set(warm_added)))
        plan = Plan(rows, dirty_sel, cold_sel, mask_sel, prefix_sel, drain)
        self._cross_evidence = dict(head_masks=len(head_masks), opened=opened, borrowed=0,
                                    head_full=set(head_rows) <= row_set,
                                    head_rows=len(head_rows), tail_rows=len(tail_rows),
                                    tail_cold=len(tail_cold))
        if getattr(self, "warm_head_sample", False):
            self._cross_evidence.update(warm_head_sample=len(warm_added),
                                        warm_head_rows=list(warm_added),
                                        warm_head_debt_payable=set(dirty) <= set(rows))
        return plan, sampled, ("sample" if sampled else "drain"), q

    def _switch_plan(self, q):
        """One physical call for a request whose offer changes; KV is never dropped."""
        split = self.head + B
        lo = max(self.head, self.origin)
        dirty = self.dirty
        self._switch_evidence = None
        self._cross_evidence = None
        before_end = self.bank.last_end
        if self.stopping:
            plan = choose(q, B, dirty, (), (), law=self.law)
            return plan, (), "drain", q
        head_ids = self.x[0, lo:min(split, self.end)].tolist()
        has_next = split < self.end and self.eos_ids.isdisjoint(head_ids)
        head_masks = {lo + i for i, t in enumerate(head_ids) if t == self.mask_id}
        evidence = dict(head_masks=len(head_masks), tail_resident=max(0, before_end - split))

        if q == B and getattr(self, "close_policy", "unchanged") != "unchanged":
            close = self._terminal_head_plan(q, tuple(range(lo, lo+len(head_ids))), head_masks)
            if close is not None:
                self._switch_evidence = dict(evidence, opened=False, borrowed=0, parked=0, head_full=True)
                return close

        if q == 2 * B:
            opened = has_next and self._tail_open(split, head_ids, head_masks, gate=self.top_gate)
            if getattr(self, "coverage", "sparse") == "head":
                # Same FlowBlock endpoint as the fixed arm; only the evidence
                # carrier differs, so a constant-C64 switch request is plan-equivalent.
                plan, sampled, phase, cap = self._head_full_plan(lo, split, q, opened, head_masks)
                evidence.update(self._cross_evidence or {})
                self._cross_evidence = None
                evidence.update(parked=sum(1 for r in plan.rows if split <= r < before_end))
                self._switch_evidence = evidence
                return plan, sampled, phase, cap
            masks, prefix, eligible = self._inputs(2 * B if opened else B)
            cold = masks - self.bank.initialized
            plan = choose(q, B, dirty, masks, prefix, cold=cold, law=self.law)
            sampled = () if plan.drain_only else (plan.cold or tuple(r for r in plan.rows if r in eligible))
            evidence.update(opened=opened, borrowed=0,
                            parked=sum(1 for r in plan.rows if split <= r < before_end))
            self._switch_evidence = evidence
            return plan, sampled, ("sample" if sampled else "drain"), q

        masks, prefix, eligible = self._inputs(B)
        cold = masks - self.bank.initialized
        plan = choose(q, B, dirty, masks, prefix, cold=cold, law=self.law)
        sampled = () if plan.drain_only else (plan.cold or tuple(r for r in plan.rows if r in eligible))
        rows, borrowed = plan.rows, ()
        borrow = self.lower_borrow or (getattr(self, "borrow_at_half", False) and q == B//2)
        if borrow and has_next and len(rows) < q and not any(r >= split for r in dirty):
            start, limit = self.bank.last_end, min(split + B, self.end)
            if start >= split and start < limit:
                borrowed = tuple(range(start, min(start + (q - len(rows)), limit)))
                rows = tuple(sorted(set(rows) | set(borrowed)))
        evidence.update(opened=False, borrowed=len(borrowed),
                        parked=sum(1 for r in rows if split <= r < before_end))
        self._switch_evidence = evidence
        return replace(plan, rows=rows), sampled, ("sample" if sampled else "drain"), q

    def _retire(self):
        retired = []
        while not self.stopping and self.head < self.end:
            lo, end = max(self.origin,self.head), min(self.end,self.head+B)
            if end > self.bank.last_end or any(lo <= r < end for r in self.dirty):
                break
            block = self.x[0,lo:end].tolist()
            if self.mask_id in block:
                break
            eos = next((j+1 for j,t in enumerate(block) if t in self.eos_ids), None)
            self.tokens.extend(block[:eos])
            retired.append(self.head//B)
            self.head = end
            self.stopping = eos is not None or end == self.end
        if self.stopping and not self.dirty:
            self.status = 'done'
        return retired

    @torch.inference_mode()
    def step(self, q):
        if self.status != 'ready' or type(q) is not int or not 4 <= q <= 2*B:
            raise ValueError("Expected a ready request and integer Q in [4,64]")
        if getattr(self, "switchable", False):
            pass  # Any offer in [4, 2B] is the current tier of this request.
        elif getattr(self, "cross", None) is not None:
            if (q == 2*B) != (self.cross in ("off", "eager", "half", "quarter")):
                raise ValueError("Cross mode is not defined at this offer")
            if getattr(self, "borrow", False) and q == 2*B:
                raise ValueError("Borrow is a residual-seat option below one block")
        try:
            return self._step(q)
        except BaseException:
            self.status = 'failed'  # A failed layer may already have scattered KV. Never replay.
            raise

    def _step(self, q):
        if self._stall >= 8*B+2:
            raise RuntimeError("No head retirement within 258 paid calls; not a valid completion")
        before = self.dirty.copy()
        if getattr(self, "switchable", False):
            plan, sampled, phase, cap = self._switch_plan(q)
        elif getattr(self, "cross", None) is not None:
            plan, sampled, phase, cap = self._cross_plan(q)
        elif self.ablation is None:
            masks, prefix, eligible = self._inputs(q)
            cold = masks-self.bank.initialized
            plan = choose(q,B,before,masks,prefix,cold=cold,law=self.law)
            sampled = () if plan.drain_only else plan.cold or tuple(r for r in plan.rows if r in eligible)
            phase, cap = None, q
        else:
            plan, sampled, phase, cap = self._ablation_plan(q)
        rows = plan.rows
        full_close_edit = (phase=="close_edit_once" and q==B and len(before)==q
                           and rows==sampled==tuple(sorted(before)) and len(rows)==q)
        if (not rows or not set(plan.dirty) == before.intersection(rows)
                or (len(before) >= q and (len(plan.dirty) != q or (sampled and phase != "dense_joint"
                                                                   and not full_close_edit)))):
            raise RuntimeError("Empty plan, skipped required refresh or sampled through a full debt offer")
        split=self.head+B
        intervention=None
        if self.tail_rule!="eager" and q>B and any(p>=split for p in plan.cold):
            head_masks={p for p in masks if p<split}
            if head_masks and head_masks<=self.bank.initialized:
                if self.tail_rule=="defer_tail":
                    sampled=tuple(p for p in sampled if p<split)
                    intervention=dict(rule=self.tail_rule,withheld=list(p for p in plan.cold if p>=split))
                else:
                    extra=tuple(sorted(head_masks-set(rows)))[:q-len(rows)]
                    if extra:
                        rows=tuple(sorted((*rows,*extra)))
                        sampled=tuple(sorted((*sampled,*extra)))
                        plan=replace(plan,rows=rows,mask=tuple(sorted((*plan.mask,*extra))))
                        intervention=dict(rule=self.tail_rule,added=list(extra))
        scoped = getattr(self, "admission_scope", "global") == "warm_head"
        warm_head = scoped and all(p in self.bank.initialized for p in
                                   range(max(self.origin,self.head), min(self.end,split)))
        e, logits = self._project(rows, cap, phase or ('sample' if sampled else 'drain'))
        if scoped:
            e["admission_head_warm_before"] = warm_head
        if getattr(self, "switchable", False):
            evidence = self._switch_evidence or {}
            self._switch_evidence = None
            e.update(switch=True, top_gate=self.top_gate, lower_borrow=self.lower_borrow,
                     head_q=sum(1 for r in rows if r < split),
                     tail_q=sum(1 for r in rows if r >= split),
                     tail_read_only=[r for r in rows if r >= split and r not in sampled],
                     head_mask_at_plan=evidence.get("head_masks"),
                     tail_opened=bool(evidence.get("opened")),
                     borrowed_rows=evidence.get("borrowed", 0),
                     parked_tail=evidence.get("parked", 0),
                     tail_resident=evidence.get("tail_resident", 0))
            if getattr(self, "coverage", "sparse") == "head":
                # Keep the legacy sparse switch record schema untouched.
                e.update(coverage="head", head_full=bool(evidence.get("head_full", False)))
            if getattr(self, "warm_head_sample", False):
                e.update(warm_head_sample=evidence.get("warm_head_sample", 0),
                         warm_head_rows=evidence.get("warm_head_rows", []))
        elif getattr(self, "cross", None) is not None:
            evidence = self._cross_evidence or {}
            self._cross_evidence = None
            e.update(cross_open=self.cross, borrow=self.borrow, coverage=self.coverage,
                     head_q=sum(1 for r in rows if r < split),
                     tail_q=sum(1 for r in rows if r >= split),
                     tail_read_only=[r for r in rows if r >= split and r not in sampled],
                     head_mask_at_plan=evidence.get("head_masks"),
                     tail_opened=bool(evidence.get("opened")),
                     borrowed_rows=evidence.get("borrowed", 0),
                     head_full=bool(evidence.get("head_full", False)))
            if getattr(self, "warm_head_sample", False):
                e.update(warm_head_sample=evidence.get("warm_head_sample", 0),
                         warm_head_rows=evidence.get("warm_head_rows", []))
        admitted, edited, revoked = [], [], []
        if sampled:
            idx = self.x.new_tensor(sampled)
            selected = logits if sampled == rows else logits[:,[rows.index(r) for r in sampled]]
            draft, conf = self.model._sample_with_temperature_topk_topp(
                selected, temperature=0., top_k=None, top_p=None)
            if draft.shape != (1,len(sampled)) or conf.shape != draft.shape or not bool(conf.isfinite().all()):
                raise RuntimeError("Invalid measured sampler output")
            old = self.x.index_select(1,idx)
            unresolved = old == self.mask_id
            if self.audit_admission:
                # Only positions really sent to the sampler, never unsampled logits.
                e["proposals"] = list(zip(old[0].tolist(), draft[0].tolist(), conf[0].tolist(),
                                          (conf[0] > .5).tolist(), (conf[0] > .9).tolist()))
            accept = unresolved & (conf > self.threshold)
            if warm_head:
                # Only the already-established current block can take the lower cutoff.
                accept |= unresolved & (idx//B == self.head//B) & (conf > .5)
            # Preserve the per-B32 fallback, but only over measured MASK proposals.
            for block in sorted({p//B for p in sampled}):
                group = unresolved & (idx//B == block)
                if group.any() and not (accept & group).any():
                    accept[0,conf.masked_fill(~group, -torch.inf)[0].argmax()] = True
            change = ~unresolved & (conf > self.edit_threshold) & (draft != old)
            write = (change | accept) & (draft != old)
            self.x[0,idx[write[0]]] = draft[write]
            admitted = idx[(unresolved & write)[0]].tolist()
            edited = idx[(change & (draft != self.mask_id))[0]].tolist()
            revoked = idx[(change & (draft == self.mask_id))[0]].tolist()
            self.sample_clock += 1
        written = set(admitted+edited+revoked)
        self.dirty = (before-set(rows)) | written
        was_stopping = self.stopping
        if self.ablation is not None:
            # Revisions are immediately visible at absolute positions, not append-only output.
            e["position_updates"] = [[p,t] for p,t in zip(idx[write[0]].tolist(), draft[write].tolist())] if written else []
        retired = self._retire()
        if (self.ablation is not None and not was_stopping and self.stopping and self.tokens
                and self.tokens[-1] in self.eos_ids):
            # Withdraw provisional positioned output at/after this exclusive output end.
            e["invalidate_from"] = self.origin+len(self.tokens)
        # Nonconvergent MASK proposals / revocation cycles must not emit incomplete text.
        self._stall = 0 if retired else self._stall+1
        e.update(tick=len(self.records), offer=q, q=len(rows), rows=list(rows), dirty_before=sorted(before),
                 dirty=list(plan.dirty), cold=list(plan.cold), mask=list(plan.mask), prefix=list(plan.prefix),
                 sampled=list(sampled), drain_only=not sampled, admitted=admitted, edited=edited,
                 revoked=revoked, written_absolute=sorted(written), dirty_after=sorted(self.dirty),
                 sample_clock=self.sample_clock, retired_blocks=retired)
        if intervention is not None:
            e["tail_intervention"]=intervention
        self.records.append(e)
        return e

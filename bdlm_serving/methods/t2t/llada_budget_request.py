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
    """
    @torch.inference_mode()
    def __init__(self, model, input_ids, *, mask_id, eos_ids=(), seed=0,
                 max_new_tokens=2048, law=Law(), confidence_threshold=.9, editing_threshold=.5,
                 tail_rule="eager", ablation=None):
        if (model.training or input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1
                or input_ids.dtype != torch.long or input_ids.device != model.get_input_embeddings().weight.device
                or type(max_new_tokens) is not int or max_new_tokens < 1 or type(seed) is not int
                or type(mask_id) is not int or not isinstance(law, Law)
                or tail_rule not in ("eager","defer_tail","fill_head")
                or ablation not in (None, "00", "10", "11")
                or (ablation is not None and tail_rule != "eager")
                or not (0 <= confidence_threshold <= 1 and 0 <= editing_threshold <= 1)):
            raise ValueError("Expected a raw eval model, unpadded batch-one IDs, and valid protocol")
        self.model, self.law, self.seed = model, law, seed  # Native temperature zero uses no RNG.
        self.tail_rule = tail_rule
        self.ablation = ablation  # None preserves the original registered family/pilots.
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
        try:
            return self._step(q)
        except BaseException:
            self.status = 'failed'  # A failed layer may already have scattered KV. Never replay.
            raise

    def _step(self, q):
        if self._stall >= 8*B+2:
            raise RuntimeError("No head retirement within 258 paid calls; not a valid completion")
        before = self.dirty.copy()
        if self.ablation is None:
            masks, prefix, eligible = self._inputs(q)
            cold = masks-self.bank.initialized
            plan = choose(q,B,before,masks,prefix,cold=cold,law=self.law)
            sampled = () if plan.drain_only else plan.cold or tuple(r for r in plan.rows if r in eligible)
            phase, cap = None, q
        else:
            plan, sampled, phase, cap = self._ablation_plan(q)
        rows = plan.rows
        if (not rows or not set(plan.dirty) == before.intersection(rows)
                or (len(before) >= q and (len(plan.dirty) != q or (sampled and phase != "dense_joint")))):
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
        e, logits = self._project(rows, cap, phase or ('sample' if sampled else 'drain'))
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
            accept = unresolved & (conf > self.threshold)
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

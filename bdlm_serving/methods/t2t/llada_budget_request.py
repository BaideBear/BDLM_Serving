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
from bdlm_serving.methods.t2t.budget_family import Law, choose
from bdlm_serving.methods.t2t.llada_sparse import LLaDASparseKernel

B = 32


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

    q<=B offers one active physical block, q>B offers up to two. Debt from
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
                 tail_rule="eager"):
        if (model.training or input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1
                or input_ids.dtype != torch.long or input_ids.device != model.get_input_embeddings().weight.device
                or type(max_new_tokens) is not int or max_new_tokens < 1 or type(seed) is not int
                or type(mask_id) is not int or not isinstance(law, Law)
                or tail_rule not in ("eager","defer_tail","fill_head")
                or not (0 <= confidence_threshold <= 1 and 0 <= editing_threshold <= 1)):
            raise ValueError("Expected a raw eval model, unpadded batch-one IDs, and valid protocol")
        self.model, self.law, self.seed = model, law, seed  # Native temperature zero uses no RNG.
        self.tail_rule = tail_rule
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
        masks, prefix, eligible = self._inputs(q)
        cold = masks-self.bank.initialized
        plan = choose(q,B,before,masks,prefix,cold=cold,law=self.law)
        rows = plan.rows
        if (not rows or not set(plan.dirty) == before.intersection(rows)
                or (len(before) >= q and (len(plan.dirty) != q or not plan.drain_only))):
            raise RuntimeError("Empty plan, skipped required refresh or sampled through a full debt offer")
        sampled = () if plan.drain_only else plan.cold or tuple(r for r in rows if r in eligible)
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
        e, logits = self._project(rows, q, 'sample' if sampled else 'drain')
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
        retired = self._retire()
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

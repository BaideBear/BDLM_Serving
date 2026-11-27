"""Single-variable low-Q probes layered over the pinned LLaDA request."""
import torch

from bdlm_serving.methods.t2t.budget_family import Law, choose
from bdlm_serving.methods.t2t.llada_budget_request import B, LLaDABudgetRequest


class LowStartRequest(LLaDABudgetRequest):
    def __init__(self, *args, arm, law=Law(0, 8), **kwargs):
        if (arm not in ("eager", "resident", "credit", "confident")
                or law not in (Law(0, 8), Law(1, 8))
                or (arm in ("resident","credit") and law != Law(0,8))):
            raise ValueError(arm)
        super().__init__(*args, law=law, **kwargs)
        self.arm, self.building, self.credit, self.credit_head = arm, False, 0, None

    def _step(self, q):
        if q not in (4, 8, 32) or (q == 32 and self.arm == "credit"):
            raise ValueError("Fixed single-block diagnostic only")
        end = min(self.head+B, self.end)
        missing = end - self.bank.last_end
        if self.arm == "confident" and missing and not self.stopping:
            masks, prefix, _ = self._inputs(q)
            plan = choose(q,B,self.dirty,masks,prefix,cold=masks-self.bank.initialized,law=self.law)
            if plan.cold and max(plan.rows)+1 < end:
                return self._partial_cold(q,plan)
        if self.arm == "resident" and (self.building or missing > q):
            if self.dirty or self.stopping or missing <= 0:
                raise RuntimeError("Build must precede the first write of this block")
            rows = tuple(range(self.bank.last_end, min(end, self.bank.last_end+q)))
            e, _ = self._project(rows, q, "build")
            self.building = self.bank.last_end < end
            self._stall += 1
            e.update(tick=len(self.records), offer=q, q=len(rows), rows=list(rows), dirty_before=[],
                     dirty=[], cold=list(rows), mask=[], prefix=[], sampled=[], drain_only=True,
                     admitted=[], edited=[], revoked=[], written_absolute=[], dirty_after=[],
                     sample_clock=self.sample_clock, retired_blocks=[])
            self.records.append(e)
            return e

        credit_in = self.credit
        room = max(0, q-len(self.dirty))
        if self.arm == "credit":
            if self.credit_head != self.head:
                self.credit_head, self.credit = self.head, 0
            credit_in = self.credit
            masks, prefix, _ = self._inputs(q)
            eligible = bool(masks and not masks-self.bank.initialized and prefix and room >= 2)
            if eligible:
                self.credit += min(room, B)
            # At Q4/Q8 the 1/8 token bucket earns at most one row per warm call.
            self.law = Law(1, room) if eligible and self.credit >= 8 else Law(0, 8)
        e = super()._step(q)
        if self.arm == "credit":
            self.credit -= 8*len(e["prefix"])
            if not 0 <= self.credit < 8:
                raise RuntimeError("Bad one-seat credit ledger")
            e.update(credit_in=credit_in, credit_out=self.credit)
        if self.arm == "resident" and e["sampled"] and e["visible_kv"] < end:
            raise RuntimeError("Sampler read a logit without all physical block keys")
        return e

    def _partial_cold(self,q,plan):
        """Only this case differs: no forced admission in a partially resident block."""
        if self._stall >= 8*B+2:
            raise RuntimeError("No head retirement within 258 paid calls; not a valid completion")
        before=self.dirty.copy()
        rows, sampled=plan.rows,plan.cold
        e,logits=self._project(rows,q,"sample")
        sel=logits if rows==sampled else logits[:,[rows.index(r) for r in sampled]]
        draft,conf=self.model._sample_with_temperature_topk_topp(sel,temperature=0.,top_k=None,top_p=None)
        if draft.shape!=(1,len(sampled)) or draft.shape!=conf.shape or not bool(conf.isfinite().all()):
            raise RuntimeError("Invalid measured cold sampler")
        idx=self.x.new_tensor(sampled)
        if not bool((self.x[0,idx]==self.mask_id).all()):
            raise RuntimeError("Unexpected non-MASK cold proposal")
        high=conf[0]>self.threshold
        take=high & (draft[0]!=self.mask_id)
        written=idx[take].tolist()
        blocked=[]
        if not bool(high.any()):
            best=int(conf[0].argmax())
            if int(draft[0,best])!=self.mask_id:
                blocked=[sampled[best]]
        self.x[0,idx[take]]=draft[0,take]
        self.dirty=(before-set(rows)) | set(written)
        self._stall+=1
        self.sample_clock+=1
        e.update(tick=len(self.records),offer=q,q=len(rows),rows=list(rows),dirty_before=sorted(before),
                 dirty=list(plan.dirty),cold=list(sampled),mask=[],prefix=[],sampled=list(sampled),
                 drain_only=False,admitted=written,edited=[],revoked=[],written_absolute=written,
                 dirty_after=sorted(self.dirty),sample_clock=self.sample_clock,retired_blocks=[],
                 deferred_fallback=blocked,cold_max_conf=float(conf.max()))
        self.records.append(e)
        return e

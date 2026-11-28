"""Opt-in request: reuse the measured fixed small-offer window across switches."""
from bdlm_serving.methods.t2t.llada_budget_request import B, LLaDABudgetRequest
from .llada_window_geometry import window


class SwitchableWindowRequest(LLaDABudgetRequest):
    def _switch_plan(self, q):
        plan, sampled, phase, cap = super()._switch_plan(q)
        self._window_geometry = None
        if q not in (4, 8, 16) or not sampled:
            return plan, sampled, phase, cap
        masks, _, eligible = self._inputs(B)
        plan, sampled, self._window_geometry = window(
            plan, sampled, masks, self.bank.initialized, min(self.head+B, self.end))
        # Fixed head-only runs never park a dirty successor. On a downshift
        # it is still a mandatory projection, but its logits must not edit.
        sampled = tuple(row for row in sampled if row in eligible)
        return plan, sampled, "sample" if sampled else "drain", cap

    def _step(self, q):
        event = super()._step(q)
        if self._window_geometry is not None:
            event["window_geometry"] = self._window_geometry
        return event


class TopAgreementWindowRequest(SwitchableWindowRequest):
    """Experimental: a second paid Q64 vote; never reuse a smaller offer's vote."""
    def _step(self,q):
        self._top_warm = q==2*B and self.bank.last_end>=min(self.head+B,self.end)
        if not self._top_warm:
            self._agree_stamp = self._agree_memo = None
        return super()._step(q)

    def _additional_admissions(self,idx,old,draft,conf,accepted):
        import torch
        from .llada_agreement import confirmation
        if not self._top_warm:
            return torch.zeros_like(accepted)
        tick=len(self.records)
        prev=(self._agree_memo if getattr(self,"_agree_stamp",None)==(self.head,tick-1)
              else draft.new_full((B,),-1))
        extra,self._agree_memo=confirmation(self.head,idx,old,draft,conf,accepted,prev,self.mask_id)
        self._agree_stamp=(self.head,tick)
        return extra

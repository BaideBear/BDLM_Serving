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

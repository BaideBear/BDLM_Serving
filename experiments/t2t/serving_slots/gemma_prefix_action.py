"""Split physical prefix refresh from sampler participation, without changing the cap.

Extends the pinned strict single-canvas control. K0 does not use the new planner.
The optional prefix has already been sampled and accepted; KV-only recomputes
its decoder state using its existing token and self-conditioning, not fresh logits
from unmeasured rows. This does not imply native dense-KV equivalence.
"""
from dataclasses import replace

from .gemma_strict_request import StrictRequest
from .gemma_strict_rows import StrictRows
from .gemma_strict_sampler import SelectedRowSampler


class RefreshOnlyRows(StrictRows):
    """Keep projection positions and their order; exclude extra rows from sampling."""

    def plan(self, offer, *, prefix_quota=0, prefix_mode="right"):
        plan = super().plan(offer, prefix_quota=prefix_quota, prefix_mode=prefix_mode)
        if plan is None or not plan.prefix:
            return plan
        pref = frozenset(plan.prefix)
        if (not pref <= self.initialized & self.accepted or pref & self.dirty
                or not pref <= set(plan.sampled) or plan.cold):
            raise RuntimeError("KV-only action requires an initialized clean prefix")
        sampled = tuple(r for r in plan.sampled if r not in pref)
        if not sampled:
            raise RuntimeError("An optional prefix must accompany an ordinary sample")
        self._pending = replace(plan, sampled=sampled)
        return self._pending


class PrefixActionRequest(StrictRequest):
    """`editable` is the registered control; `kv_only` locks only the extra prefix."""

    def __init__(self, model, data, *, prefix_action, **kwargs):
        if prefix_action not in ("editable", "kv_only"):
            raise ValueError("Expected prefix_action editable or kv_only")
        self.prefix_action = prefix_action
        super().__init__(model, data, **kwargs)

    def _start_block(self):
        super()._start_block()
        if self.prefix_action == "kv_only" and self.quota:
            self.rows = RefreshOnlyRows(SelectedRowSampler.steps, seed=self.seed, ordinal=self.ordinal)

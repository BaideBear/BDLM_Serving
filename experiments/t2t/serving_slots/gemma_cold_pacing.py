"""Limit first visits, before projection. No sampler, clock or KV implementation here."""
from importlib.util import module_from_spec, spec_from_file_location

from bdlm_serving.methods.t2t.budget_family import Law
from .gemma_fixed_dual import AblationRequest, B


def first_batch(pos, sample, cold, debt, q):
    first = sorted(cold)
    if (set(first) != sample or len(first) != len(sample) or sample & debt
            or set(pos) != debt | sample or len(pos) > q):
        raise ValueError("Mixed cold selection or missing mandatory positions")
    keep = tuple(first[:q//2])
    return tuple(sorted(debt.union(keep))), set(keep), keep


class ColdPacingRequest(AblationRequest):
    def __init__(self, model, data, *, q, seed, max_new_tokens, law, ablation, pace):
        if (type(pace) is not bool or q not in (32, 64, 128) or ablation != "00"
                or law != Law(1, 4 if q == 128 else 8)):
            raise ValueError("Expected the low-tier fixed90_fused controls")
        self.pace = pace
        super().__init__(model, data, q=q, seed=seed, max_new_tokens=max_new_tokens,
                         law=law, ablation=ablation, top_gate="half", fuse_first_verify=True)

    def step(self, q=None):
        if q is not None and q != self.fixed_q:
            raise ValueError("Fixed offers only")
        return super().step(q)

    def _apply(self, states, pos, sample, prefix, cold, offer, route):
        if self.pace and cold:
            if len(states) != 1 or route != "sparse" or prefix:
                raise ValueError("Not the head-only first-visit branch")
            base = self.origin + B*self.ordinal
            debt = {base+i for i in self.rows.dirty}
            pos, sample, cold = first_batch(pos, sample, cold, debt, offer)
        return super()._apply(states, pos, sample, prefix, cold, offer, route)


def make_auditor():
    """Private binding to the pinned auditor; never replace a shared module global."""
    from . import gemma_fixed_dual_eval as original
    spec = spec_from_file_location(__package__ + "._cold_pacing_audit", original.__file__)
    private = module_from_spec(spec)
    spec.loader.exec_module(private)
    private.AblationRequest = ColdPacingRequest
    return private.audited

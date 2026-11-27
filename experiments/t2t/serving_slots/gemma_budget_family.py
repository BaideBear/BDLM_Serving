"""Shared allocation law on the real strict single-canvas Gemma request.

Q>B on a single canvas is not a true two-canvas execution. Refuse that tier
until the multi-canvas adapter implements the same transaction and sampler.
"""
from bdlm_serving.methods.t2t.budget_family import Law, choose
from .gemma_strict_request import StrictRequest
from .gemma_strict_rows import B, OFFERS, RowPlan, StrictRows


class FamilyRows(StrictRows):
    def __init__(self, *args, law, **kwargs):
        super().__init__(*args, **kwargs)
        self.law = law

    def plan(self, offer, *, prefix_quota=0, prefix_mode="right"):
        if type(offer) is not int or offer not in OFFERS or prefix_quota or prefix_mode != "right":
            raise ValueError("Use one shared prefix law and the right-edge rank")
        if self._pending is not None:
            raise RuntimeError("Finish outstanding plan first")
        if self.status != "running":
            return None
        full = set(range(B))
        stop = self.age == self.max_steps or self._native_stop or self.accepted == full
        unresolved = set() if stop else full - self.accepted
        cold = set() if stop else full - self.initialized
        frontier = min(unresolved, default=0)
        prefix = sorted((set(range(frontier)) & self.initialized) - self.dirty, reverse=True)
        a = choose(offer, B, self.dirty, unresolved, prefix, cold=cold, law=self.law)
        if not a.rows:
            raise RuntimeError("Unfinished request with no eligible row")
        sampled = () if a.drain_only else a.cold or a.rows
        self._pending = RowPlan(offer, a.rows, sampled, a.cold, a.prefix)
        return self._pending


class FamilyRequest(StrictRequest):
    def __init__(self, model, data, *, seed, max_new_tokens=2048, law=Law()):
        self.law = law
        super().__init__(model, data, seed=seed, max_new_tokens=max_new_tokens)

    def _start_block(self):
        super()._start_block()
        self.rows = FamilyRows(self.rows.max_steps, seed=self.seed, ordinal=self.ordinal, law=self.law)

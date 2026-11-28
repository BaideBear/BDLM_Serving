"""One paid sampler tick on the resolved rows of a warm all-debt drain.

The control drain already projects every outstanding write.  This factor keeps
that exact physical call and only lets the native sampler consume the measured
resolved dirty rows (rows with a token, not unresolved MASKs).  It never adds a
row, a cold seat or a second tick, and it fires at most once per physical head.
"""
from .gemma_fixed_dual import B, FULL
from .gemma_cold_first_retry import RecordingVerifiedEOSRequest

REQUESTS = []


def warm_debt_sample(states, pos, sample, route, *, tail, ordinal, origin, retried):
    """Return the resolved dirty rows to sample, or None when not eligible."""
    if route != "drain" or sample or tail is not None or ordinal in retried:
        return None
    head = states[0]
    debt = {origin + B * c.ordinal + i for c in states for i in c.rows.dirty}
    if debt != set(pos) or len(head.rows.initialized) < B:
        return None
    if not head.can_sample or head.sampler.phase != "sample":
        return None
    unresolved = {origin + B * head.ordinal + i for i in FULL - head.rows.accepted}
    extra = set(pos) - unresolved
    return extra or None


class RecordingControlRequest(RecordingVerifiedEOSRequest):
    def __init__(self, model, data, *, seed, q, early):
        super().__init__(model, data, seed=seed, q=q, early=early)
        REQUESTS.append(self)


class WarmDebtResampleRequest(RecordingControlRequest):
    """Verified-EOS control plus the one-shot warm resolved-debt retry."""

    def __init__(self, model, data, *, seed, q, early):
        super().__init__(model, data, seed=seed, q=q, early=early)
        self._warm_debt_resampled = set()

    def _apply(self, states, pos, sample, prefix, cold, offer, route):
        extra = warm_debt_sample(
            states, pos, sample, route, tail=self.tail, ordinal=self.ordinal,
            origin=self.origin, retried=self._warm_debt_resampled)
        if extra is not None:
            self._warm_debt_resampled.add(self.ordinal)
            return super()._apply(states, pos, extra, prefix, cold, offer, "warm_debt_resample")
        return super()._apply(states, pos, sample, prefix, cold, offer, route)

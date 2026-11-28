"""One paid sampler tick on the first pure all-debt drain of a cold head.

The control drain already projects every outstanding write.  This factor keeps
that exact physical call and only lets the native sampler consume the measured
unresolved dirty rows.  It never adds a row, a cold seat or a second tick, and
it fires at most once per physical head.
"""
from .gemma_fixed_dual import B, FULL
from .gemma_verified_eos import VerifiedEOSRequest

REQUESTS = []


def retry_sample(states, pos, sample, route, *, tail, ordinal, origin, retried):
    """Return the unresolved dirty rows to sample, or None when not eligible."""
    if route != "drain" or sample or tail is not None or ordinal in retried:
        return None
    head = states[0]
    debt = {origin + B * c.ordinal + i for c in states for i in c.rows.dirty}
    if debt != set(pos) or len(head.rows.initialized) >= B:
        return None
    if not head.can_sample or head.sampler.phase != "sample":
        return None
    unresolved = {origin + B * head.ordinal + i for i in FULL - head.rows.accepted}
    extra = set(pos) & unresolved
    return extra or None


def _pre_state(request):
    return dict(
        ordinal=request.ordinal,
        initialized=set(request.rows.initialized),
        accepted=set(request.rows.accepted),
        dirty=set(request.rows.dirty),
        phase=request.sampler.phase,
        age=request.sampler.age,
        tail=request.tail is not None,
    )


class RecordingVerifiedEOSRequest(VerifiedEOSRequest):
    """Control request that records events and pre-states without changing them."""

    def __init__(self, model, data, *, seed, q, early):
        super().__init__(model, data, seed=seed, q=q, early=early)
        self.events = []
        self.pre_states = []
        REQUESTS.append(self)

    def step(self, q=None):
        self.pre_states.append(_pre_state(self))
        event = super().step(q)
        self.events.append(event)
        return event


class ColdFirstRetryRequest(RecordingVerifiedEOSRequest):
    """Verified-EOS control plus the one-shot cold-drain retry."""

    def __init__(self, model, data, *, seed, q, early):
        super().__init__(model, data, seed=seed, q=q, early=early)
        self._cold_first_retried = set()

    def _apply(self, states, pos, sample, prefix, cold, offer, route):
        extra = retry_sample(
            states, pos, sample, route, tail=self.tail, ordinal=self.ordinal,
            origin=self.origin, retried=self._cold_first_retried)
        if extra is not None:
            self._cold_first_retried.add(self.ordinal)
            return super()._apply(states, pos, extra, prefix, cold, offer, "cold_first_retry")
        return super()._apply(states, pos, sample, prefix, cold, offer, route)

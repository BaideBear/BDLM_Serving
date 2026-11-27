"""Experimental B256 close: refresh terminal dirty rows in the sole argmax sweep.

Sampler IDs and self-conditioning stay frozen; resident layer KV does not.
Earlier chunks can see stale peers. This is not a replay of drain-then-verify.
"""
from copy import copy

from .gemma_strict_request import StrictRequest, _valid_offer
from .gemma_strict_rows import B


class ClosingRequest(StrictRequest):
    """StrictRequest API and events; verify.dirty_before is outstanding KV debt."""

    def _step(self, q):
        rows, sampler = self.rows, self.sampler
        if sampler.phase == "sample":
            # Ask the existing stop predicate without hiding debt from the live planner.
            clean = copy(rows)
            clean.dirty = frozenset()
            reason = clean.status
            if reason == "running":
                return super()._step(q)
            if reason == "incomplete" or len(rows.initialized) != B:
                raise RuntimeError("Cannot verify an incompletely initialized canvas")
            sampler.begin_close()  # Also requires full SC coverage.
            self._reason = reason
            self._order = sorted(rows.dirty) + [i for i in range(B) if i not in rows.dirty]

        # Dirty wins admission. One decoder call still takes sorted physical positions.
        take = tuple(sorted(self._order[self.cursor:self.cursor+q]))
        record = dict(tick=self.tick, ordinal=self.ordinal, offer=q, q=len(take), rows=list(take),
                      phase="verify", stop_reason=self._reason, dirty_before=sorted(rows.dirty),
                      prefix=[], sampled=[], admitted=[], edited=[], revoked=[], encoded=False)
        with self.lane as lane:
            sampler.close_chunk(take, lambda r, ids, sc: lane.project(
                sampler.ids, r, q, conditioning=sc))
        # Only retire debt after every layer wrote KV and the fresh argmax was sealed.
        rows.dirty = rows.dirty.difference(take)
        self.cursor += len(take)
        if self.cursor == B:
            record["encoded"] = self._commit()
        self.tick += 1
        return record


def generate(model, data, *, seed, offers, prefix_quota=0, prefix_mode="right", max_new_tokens=2048):
    """Convenience wrapper with the strict_request.generate signature."""
    offers = tuple(offers)
    if not offers or not all(map(_valid_offer, offers)):
        raise ValueError("Unsupported single-canvas physical budget")
    request = ClosingRequest(model, data, seed=seed, max_new_tokens=max_new_tokens,
                             prefix_quota=prefix_quota, prefix_mode=prefix_mode)
    trace = []
    while request.status == "ready":
        trace.append(request.step(offers[request.tick % len(offers)]))
    return request.tokens, trace

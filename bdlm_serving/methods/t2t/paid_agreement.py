"""A two-observation admission experiment; one ledger per request."""
from math import isfinite
from operator import index


class EvidenceLedger:
    """Only currently measured unresolved rows may receive an extra admission."""

    def __init__(self, floor=0.5):
        if not 0 <= floor < 1:
            raise ValueError("invalid confidence floor")
        self.floor = floor
        self._block, self._tick, self._votes = None, -1, {}

    def observe(self, *, block, tick, positions, proposals, probabilities, native, excluded_ids):
        """Record the base sampler result, not a second draw or stale logits.

        `tick` is the request-global physical decoder count, not the number of
        sampler invocations. Positions are absolute. A skipped call expires the
        last observation. The adapter owns acceptance and mandatory writeback.
        """
        block, tick = index(block), index(tick)
        if tick < 0 or tick <= self._tick:
            raise ValueError("tick must strictly increase within request")
        adjacent = block == self._block and tick == self._tick+1
        votes, seen, extra = {}, set(), []
        for pos, token, prob, accepted in zip(positions, proposals, probabilities, native, strict=True):
            pos, token, prob = index(pos), index(token), float(prob)
            if pos < 0 or pos in seen or token < 0 or not isfinite(prob) or not 0 <= prob <= 1:
                raise ValueError("invalid measured unresolved row")
            if accepted not in (False, True):
                raise ValueError("invalid native acceptance bit")
            seen.add(pos)
            if accepted or token in excluded_ids or prob <= self.floor:
                continue
            if adjacent and self._votes.get(pos) == token:
                extra.append(pos)
            votes[pos] = token
        self._block, self._tick, self._votes = block, tick, votes
        return tuple(extra)

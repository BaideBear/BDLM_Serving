"""Filter optional admissions at the first measured unresolved hole."""
from operator import index

from .paid_agreement import EvidenceLedger


def before_measured_hole(positions, native, repeated):
    """Inputs are the validated, measured MASK rows of one block decision."""
    optional = set(repeated)
    stop = min((p for p, take in zip(positions, native, strict=True)
                if not take and p not in optional), default=None)
    if stop is None:
        return repeated
    return tuple(p for p in repeated if p < stop)


class FrontierEvidenceLedger(EvidenceLedger):
    def observe(self, *, block, tick, positions, proposals, probabilities, native, excluded_ids):
        positions, native = tuple(map(index, positions)), tuple(native)
        extra = super().observe(block=block, tick=tick, positions=positions, proposals=proposals,
                                probabilities=probabilities, native=native, excluded_ids=excluded_ids)
        return before_measured_hole(positions, native, extra)

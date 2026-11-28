"""Bound the aggregate uncertainty of optional paid-agreement admissions."""
from math import isfinite
from operator import index

from .paid_agreement import EvidenceLedger


class CappedEvidenceLedger(EvidenceLedger):
    """The allowance applies to one sampling decision, not the physical Q bill."""

    def __init__(self, floor=0.5, allowance=0.5):
        super().__init__(floor=floor)
        self.allowance = float(allowance)
        if not isfinite(self.allowance) or self.allowance < 0:
            raise ValueError("allowance must be finite and nonnegative")
        self._probabilities = {}

    def observe(self, *, block, tick, positions, proposals, probabilities, native, excluded_ids):
        positions = tuple(map(index, positions))
        probabilities = tuple(map(float, probabilities))
        current = dict(zip(positions, probabilities, strict=True))
        repeated = super().observe(block=block, tick=tick, positions=positions, proposals=proposals,
                                   probabilities=probabilities, native=native, excluded_ids=excluded_ids)
        # The parent already excludes skipped, changed, special and native-accepted rows.
        ranked = sorted((1 - min(self._probabilities[pos], current[pos]), pos) for pos in repeated)
        self._probabilities = current
        remaining, chosen = self.allowance, set()
        for cost, pos in ranked:
            if cost > remaining:
                break
            remaining -= cost
            chosen.add(pos)
        return tuple(pos for pos in repeated if pos in chosen)

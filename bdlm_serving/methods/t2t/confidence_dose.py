"""Confidence-only placement with a dose supplied by measured repeated proposals."""
from .paid_agreement import EvidenceLedger


class ConfidenceDose:
    def __init__(self, floor=.5):
        self.floor = floor
        self.repeat = EvidenceLedger(floor=floor)

    def observe(self, **observed):
        k = len(self.repeat.observe(**observed))
        eligible = [(p, pos) for p, pos, token, take in zip(
            observed["probabilities"], observed["positions"], observed["proposals"],
            observed["native"], strict=True)
            if not take and token not in observed["excluded_ids"] and p > self.floor]
        picked = sorted(eligible, key=lambda pair: (-pair[0], pair[1]))[:k]
        if len(picked) != k:
            raise ValueError("repeat dose exceeds the current eligible row count")
        return tuple(sorted(pos for _, pos in picked))

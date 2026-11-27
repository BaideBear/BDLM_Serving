"""Registered LLaDA offer traces. A study is selected before worker startup."""
from dataclasses import dataclass


LOW = (4, 8, 16, 32)
REPLAY = (64, 64, 32, 64, 64, 16, 64, 64, 8, 64, 64, 4)


@dataclass(frozen=True)
class MaskStudy:
    trace: tuple
    targets: tuple
    folder: str


def study(name="replay"):
    if name == "replay":
        return MaskStudy(REPLAY, LOW, "llada21-offer-masks-gpu90-v1")
    for q in LOW:
        if name == f"fixed{q}":
            return MaskStudy((q,), (q,), f"llada21-fixed-q{q}-mask-gpu90-v1")
    raise ValueError(f"Unregistered LLaDA mask study: {name!r}")

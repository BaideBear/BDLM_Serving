"""A terminal decision for one active physical block with a *frozen* prefix."""
from dataclasses import dataclass
from typing import Literal, Optional, Tuple


@dataclass(frozen=True)
class StopDecision:
    action: Literal["continue", "pay", "finish"]
    reason: str
    rows: Tuple[int, ...] = ()


def frozen_stop(*, start, width, ready, generated_from, generated_until,
                eos_position: Optional[int], debt, offer) -> StopDecision:
    """All row coordinates are absolute; ready is the length of a paid immutable prefix.

    eos_position is the first known EOS within this block's generation interval,
    never one from the prompt. Call after native admissions/cache accounting.
    A pay decision requires a real projection without sampling and a fresh
    decision with the updated ledger. It does not cancel any debt by itself.
    """
    if not (start >= 0 and width > 0 and 0 <= ready <= width and
            0 <= generated_from < generated_until and offer > 0):
        raise ValueError("Invalid block, generation interval, frontier, or offer")
    end, frontier = start + width, start + ready
    rows = tuple(sorted(debt))
    if len(set(rows)) != len(rows) or any(not frontier <= p < end for p in rows):
        raise ValueError("Duplicate, out-of-block or frozen-prefix debt")
    if eos_position is not None and not (
            max(start, generated_from) <= eos_position < min(end, generated_until)):
        raise ValueError("EOS is not in this block's generated range")
    if eos_position is None or eos_position >= frontier:
        return StopDecision("continue", "no_frozen_eos")
    if ready == width:
        return StopDecision("finish", "ordinary_block_end")
    if len(rows) > offer:
        return StopDecision("continue", "debt_exceeds_offer")
    if rows:
        return StopDecision("pay", "suffix_debt", rows)
    return StopDecision("finish", "frozen_eos")

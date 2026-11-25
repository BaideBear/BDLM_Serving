"""Rank resolved positions using only the previous block-local decoder state."""
from dataclasses import dataclass


@dataclass
class History:
    # `None` certainty means no measured logits; never replace it with fresh dense information.
    resolved: list
    certainty: list
    streak: list
    changed: set
    fresh: set

    def rank(self, eligible, signal):
        change = sorted(self.changed)
        def key(i):
            uncertain = float("inf") if self.certainty[i] is None else self.certainty[i]
            if signal == "uncertain":
                return (uncertain, i)
            if signal == "young":
                return (self.streak[i], i)
            if signal == "neighbor":
                return (min((abs(i-j) for j in change if j != i), default=len(self.streak)), i)
            if signal == "recent_then_uncertain":
                return (self.streak[i], uncertain, i)
            raise ValueError(signal)
        return sorted(eligible, key=key)


def allocate(history, editable, budget, fresh_slots, mask_slots, signal="uncertain"):
    """Disjoint useful Q positions, not a model call. Unused quotas backfill to unresolved then edits.

    Fresh positions changed *to* a resolved ID last step (including a T2T); their new ID
    was not the one previously encoded. Callers charge any separate mandatory commit.
    """
    if min(budget, fresh_slots, mask_slots) < 0 or fresh_slots + mask_slots > budget:
        raise ValueError("invalid budget split")
    editable = sorted(set(editable))
    if not all(0 <= i < len(history.resolved) for i in editable):
        raise ValueError("position outside block")
    masks = [i for i in editable if not history.resolved[i]]
    fresh = history.rank([i for i in editable if i in history.fresh and history.resolved[i]], signal)
    edits = history.rank([i for i in editable if i not in history.fresh and history.resolved[i]], signal)
    eq = budget - mask_slots - fresh_slots
    chosen = fresh[:fresh_slots] + masks[:mask_slots] + edits[:eq]
    remaining = budget - len(chosen)
    return sorted(chosen + (masks[mask_slots:] + fresh[fresh_slots:] + edits[eq:])[:remaining])

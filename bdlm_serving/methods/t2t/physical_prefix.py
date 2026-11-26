"""Select accepted rows in the live physical block, by exchange or unused seats."""

from dataclasses import dataclass
from random import Random


MATCHED_MODES = ("hard_prefix_zero", "hard_prefix_random", "hard_prefix_frontier_right")
SLACK_MODES = ("hard_prefix_slack_right", "hard_prefix_slack_random")
PREFIX_MODES = MATCHED_MODES + SLACK_MODES


def prefix_quota(width, budget, fresh_slots, mask_slots, policy):
    """Use Q:0:Q-K with mandatory fresh and at least one nominal MASK seat."""
    if (policy not in PREFIX_MODES or not all(type(x) is int for x in
            (width, budget, fresh_slots, mask_slots)) or not 1 <= budget <= width
            or fresh_slots != 0 or not 1 <= mask_slots <= budget):
        raise ValueError("Physical prefix policies require Q:0:Q-K with Q <= block and 0 <= K < Q")
    quota = budget - mask_slots
    if (policy == "hard_prefix_zero") != (quota == 0):
        raise ValueError("hard_prefix_zero requires K=0; prefix placement requires K>0")
    return quota


@dataclass(frozen=True)
class PrefixPlan:
    budget: int
    quota: int
    block_start: int
    clock: int
    prefix_end: int
    fresh_rows: tuple[int, ...]
    eligible_prefix_rows: tuple[int, ...]
    first_mask: int | None
    zero_rows: tuple[int, ...]
    zero_mask_rows: tuple[int, ...]
    zero_backfill_rows: tuple[int, ...]
    added_rows: tuple[int, ...]
    evicted_rows: tuple[int, ...]
    positions: tuple[int, ...]

    def audit(self):
        """All rows are physical block-local; add block_start for absolute IDs."""
        return dict(block_start=self.block_start, clock=self.clock,
                    prefix_end=self.prefix_end, budget=self.budget, quota=self.quota,
                    fresh_rows=list(self.fresh_rows),
                    eligible_prefix_rows=list(self.eligible_prefix_rows), first_mask=self.first_mask,
                    zero_rows=list(self.zero_rows), zero_mask_rows=list(self.zero_mask_rows),
                    zero_backfill_rows=list(self.zero_backfill_rows),
                    added_rows=list(self.added_rows), evicted_rows=list(self.evicted_rows),
                    selected_rows=list(self.positions), zero_width=len(self.zero_rows),
                    actual_width=len(self.positions), underfill=max(0, self.budget - len(self.positions)),
                    overflow=max(0, len(self.positions) - self.budget))


def plan_prefix(accepted, editable, fresh, budget, quota, policy, *, seed, block_start, clock):
    """Build zero first, then exchange unprotected rows or use genuine slack.

    ``accepted`` covers the full *current* physical block, including any
    non-editable prompt rows. Scan to the first unaccepted row before removing
    fresh and non-editable positions from the prefix pool. Never backfill from
    that pool. Fresh can overflow Q; protect the first MASK whenever it fits.
    """
    if type(quota) is not int:
        raise ValueError("Prefix quota must be an integer")
    prefix_quota(len(accepted), budget, 0, budget - quota, policy)
    rows = sorted(set(editable))
    fresh = set(fresh)
    if any(i < 0 or i >= len(accepted) for i in rows) or not fresh.issubset(rows):
        raise ValueError("Rows must belong to the current block and fresh rows must be editable")
    if any(not accepted[i] for i in fresh):
        raise ValueError("Every fresh row must be accepted")

    prefix_end = next((i for i, admitted in enumerate(accepted) if not admitted), len(accepted))
    pool = tuple(i for i in rows if i < prefix_end and i not in fresh)
    room = max(0, budget - len(fresh))
    masks = tuple(i for i in rows if not accepted[i])[:room]
    backfill = tuple(i for i in rows if i >= prefix_end and accepted[i] and i not in fresh
                     )[:room - len(masks)]
    zero = tuple(sorted((*fresh, *masks, *backfill)))

    # Drop ordinary old rows before any extra MASK, taking the last in each
    # group to preserve the zero planner's priority among the remaining rows.
    evictable = backfill[::-1] + masks[:0:-1]
    slack_only = policy in SLACK_MODES
    # When zero is empty the adapter must take its normal verify/stop path.
    # An optional refresh must not create a new forward just to spend quota.
    capacity = len(evictable)
    if slack_only:
        capacity = max(0, budget - len(zero)) if zero else 0
    count = min(quota, len(pool), capacity)
    added = ()
    if count:
        if policy in ("hard_prefix_random", "hard_prefix_slack_random"):
            ranked = list(pool)
            Random(f"physical-prefix:{seed}:{block_start}:{clock}").shuffle(ranked)
            added = tuple(sorted(ranked[:count]))
        else:
            added = pool[-count:]
    evicted = () if slack_only else tuple(sorted(evictable[:count]))
    positions = tuple(sorted((set(zero) - set(evicted)).union(added)))
    return PrefixPlan(budget, quota, block_start, clock, prefix_end, tuple(sorted(fresh)), pool,
                      masks[0] if masks else None, zero, masks, backfill, added, evicted, positions)

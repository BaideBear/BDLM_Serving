"""Planning helper for the controlled KV-only prefix-refresh intervention.

The intervention consumes only physical-Q slack left unused by the original
hard-capped LLaDA2.1 IBS planner.  Injected rows are real query rows: the
forward refreshes their K/V, but their logits are discarded and they are never
sampled or T2T-edited.  No upstream controller state is modified.
"""
import numpy as np

ORDERS = ("right", "oldest")


def plan_prefix_refresh(*, prompt_len, block_start, cached_prefix_before,
                        old_mask, dirty, initialized, original_positions,
                        offer, quota, order="right"):
    """Return ``(extra_absolute, eligible_absolute, slack)``.

    ``extra_absolute`` is the ascending list of rows to add to the original
    picked rows (``"right"`` selects the highest eligible rows, ``"oldest"`` the
    lowest; the projection itself is always sorted ascending).
    ``eligible_absolute`` is the full eligible set in ascending order, kept for
    audit.  ``slack`` is the unused physical cap after the original selection.

    ``old_mask`` is block-local (non-zero means unmasked).  ``dirty`` and
    ``initialized`` contain absolute positions.  ``original_positions`` are the
    absolute positions picked by the unmodified planner.
    """
    if order not in ORDERS:
        raise ValueError(f"prefix_order must be one of {ORDERS}")
    quota = int(quota)
    if quota <= 0 or offer is None:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), 0
    original = [int(p) for p in original_positions]
    if not original:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), 0

    count = len(original)
    slack = max(0, int(offer) - count)
    if slack == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), 0

    block_start = int(block_start)
    lower = max(int(prompt_len), block_start)
    upper = block_start + int(cached_prefix_before)
    largest_original = max(original)
    original_set = set(original)
    eligible = []
    for pos in range(lower, upper):
        local = pos - block_start
        if local < 0 or local >= len(old_mask) or not old_mask[local]:
            continue
        if pos in dirty or pos not in initialized or pos in original_set:
            continue
        # The cache visible end is set by the largest original picked row.
        # Keeping every added row strictly earlier leaves that span unchanged.
        if pos >= largest_original:
            continue
        eligible.append(pos)
    if not eligible:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), slack

    take = min(quota, slack)
    if order == "right":
        # "right" selects the highest eligible rows; the projection itself is
        # always sorted ascending so absolute_positions is batch/audit friendly.
        chosen = sorted(eligible[-take:])
    else:
        chosen = eligible[:take]
    return (np.asarray(chosen, dtype=np.int64),
            np.asarray(eligible, dtype=np.int64), slack)

"""Fixed, logit-free B32 old-position ranking; see geometry-preregister.md."""

from bisect import bisect_left

__all__ = ["rank_old"]

_INF = float("inf")


def _mask_distance(i, masks):
    k = bisect_left(masks, i)
    return min(i - masks[k - 1] if k else _INF,
               masks[k] - i if k < len(masks) else _INF)


def rank_old(edit_pool, editable, resolved, fresh, last_write, last_query,
             clock, previous_strict):
    """Return a new list with the exact old candidate pool in priority order.

    Positions are block-local. ``resolved`` and both event arrays have block
    length. Clocks are in [-1, clock], where -1 means never. ``fresh`` and
    ``previous_strict`` contain editable, resolved positions. The latter
    describes strict edits from the previous denoise step, not a prediction.
    The caller owns allocation, forced refresh, and MASK coverage.
    """
    pool = list(edit_pool)
    if not pool:
        return pool
    block = len(resolved)
    if len(last_write) != block or len(last_query) != block:
        raise ValueError("resolved and both event clocks must have block length")
    if clock < 0 or any(t < -1 or t > clock for t in last_write) or any(
            t < -1 or t > clock for t in last_query):
        raise ValueError("expected nonnegative clock and event times in [-1, clock]")
    eligible = set(editable)
    if any(i < 0 or i >= block for i in eligible):
        raise ValueError("editable position outside block")
    old_or_fresh = {i for i in eligible if resolved[i]}
    fresh, strict = set(fresh), set(previous_strict)
    if not (fresh | strict) <= old_or_fresh:
        raise ValueError("fresh and previous_strict must be editable and resolved")
    if len(set(pool)) != len(pool) or not set(pool) <= old_or_fresh - fresh:
        raise ValueError("edit_pool must contain distinct editable non-fresh resolved positions")

    masks = sorted(eligible - old_or_fresh)
    anchors = [(j, last_write[j]) for j in fresh if last_write[j] >= 0]
    strict_anchors = [(j, last_write[j], clock - last_write[j])
                      for j in strict if last_write[j] >= 0]

    def key(i):
        last = last_query[i]
        dm = _mask_distance(i, masks)
        # At equal clocks the query happened before the write.
        df = min((abs(i-j) for j, write in anchors if write >= last), default=_INF)
        g = max(dm, df)
        s = min((abs(i-j) + age for j, write, age in strict_anchors
                 if i != j and write >= last), default=_INF)
        return min(g, s), g, s, dm, df, -last_write[i], last, i

    return sorted(pool, key=key)

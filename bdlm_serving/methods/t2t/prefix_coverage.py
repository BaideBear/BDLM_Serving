"""Choose the upper endpoint of each equal-mass bin in a paid prefix pool."""
from operator import index


def spread_prefix(pool, k):
    """Return k sorted positions, preserving the frontier when k is nonzero.

    Supply the entire eligible pool for ONE physical block, after excluding
    prompt, unpaid and mandatory rows. The caller owns both eligibility and
    quota; this function must not widen visibility or add sampler seats.
    """
    rows = tuple(sorted(map(index, pool)))
    k = index(k)
    n = len(rows)
    if not 0 <= k <= n or (n and rows[0] < 0) or len(set(rows)) != n:
        raise ValueError("Invalid eligible prefix pool or dose")
    return tuple(rows[(j * n - 1) // k] for j in range(1, k + 1))

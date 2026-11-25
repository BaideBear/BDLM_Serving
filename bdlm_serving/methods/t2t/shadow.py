"""Construct a one-seat counterfactual; only the diagnostic caller sees lookahead."""


def swap_pool(baseline, resolved, fresh, editable):
    """Preserve all fresh and at least the first MASK if one is selected."""
    base = set(baseline)
    pool = sorted(i for i in editable if resolved[i] and i not in base)
    old = sorted(i for i in base if resolved[i] and i not in fresh)
    masks = sorted(i for i in base if not resolved[i])
    victim = old[-1] if old else masks[-1] if len(masks) >= 2 else None
    return victim, pool


def screened_swap(baseline, resolved, fresh, editable, probe):
    """A union screen then an exact-width replay from the same pre-step cache.

    Misses are allowed. In particular, a union negative need not be a swap
    negative, and only the highest-confidence union hit is verified. No output
    produced by a diagnostic forward becomes the production token or cache.
    """
    victim, pool = swap_pool(baseline, resolved, fresh, editable)
    if victim is None or not pool:
        return baseline, []
    union = sorted(set(baseline) | set(pool))
    hits = probe(union, pool, "oracle_union")
    if not hits:
        return baseline, []
    candidate = min(hits, key=lambda i: (-hits[i], i))
    trial = sorted((set(baseline) - {victim}) | {candidate})
    if candidate not in probe(trial, [candidate], "oracle_trial"):
        return baseline, []
    return trial, [candidate]

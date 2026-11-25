"""Pure-CPU hard-refresh allocator for sparse Q.

The allocator never reads logits, token identities, or model state.  It uses
only the resolved mask, the mandatory fresh set, and two event clocks:

``last_write[i]``  last step at which position ``i`` was written/rewritten
                    (``-1`` if never).
``last_query[i]``  last step at which position ``i`` was actually queried
                    (``-1`` if never).

A write happens after the computation of its step.  Therefore a neighbour
written at step ``t`` is already visible to a candidate whose last query was
also ``t``: the frontier test is ``last_write[j] >= last_query[i]``.  Using
``>`` would drop every same-round compute-then-write neighbour.

For every non-fresh resolved edit candidate ``i``::

    score(i) = sum over j != i, |i-j| <= RADIUS,
                             last_write[j] >= 0,
                             last_write[j] >= last_query[i]
                  1.0 / (1.0 + (clock - last_write[j]) + |i-j|)

This is a cheap additive time-space decay: recent and nearby rewrites
contribute more.  ``RADIUS = 32`` keeps a B=256 block at O(B*R) instead of
O(B^2).  Ties break by ascending position, so the ranking is deterministic.

Policies
--------
``hard_frontier``
    Rank the edit pool by the score above.
``hard_random``
    Uniformly shuffle the *same* edit pool with ``f"{seed}:{clock}"`` and take
    the same number of seats.  The clock is part of the seed so each step gets
    a different permutation.  There is no history/confidence gate: a candidate
    with no write event is just as eligible as one with a recent rewrite.
``hard_noedit``
    Reserve no edit seats.  Every non-fresh seat goes to MASK, with the same
    shared backfill if MASK is exhausted.

Allocation order
----------------
All counts are capped at ``min(len(eligible), max(budget, len(fresh)))``:

1. every fresh position is hard-selected;
2. if an unselected MASK exists and at least one seat remains, exactly one
   MASK is selected first (lowest position);
3. up to ``edit_slots`` non-fresh resolved positions are reserved;
4. remaining seats go to MASK;
5. any still-unused seat backfills from the other pool in neutral ascending
   position order for every policy, so only the reserved set differs between
   policies.

Only fresh can overflow ``budget``.  The mandatory MASK never increases the
total beyond the cap.
"""

from __future__ import annotations

import random


__all__ = ["select_hard"]


RADIUS = 32
_POLICIES = ("hard_frontier", "hard_random", "hard_noedit", "hard_slack",
             "hard_trigger")


def _frontier_scores(edit_pool, last_write, last_query, clock, radius=RADIUS):
    """Return ``{position: score}`` for the non-fresh edit pool."""
    scores = {}
    block = len(last_write)
    for i in edit_pool:
        threshold = last_query[i]
        total = 0.0
        low = i - radius
        if low < 0:
            low = 0
        high = i + radius + 1
        if high > block:
            high = block
        for j in range(low, high):
            if j == i:
                continue
            write = last_write[j]
            if write >= 0 and write >= threshold:
                total += 1.0 / (1.0 + (clock - write) + abs(i - j))
        scores[i] = total
    return scores


def select_hard(editable, resolved, fresh, last_write, last_query, clock,
                budget, edit_slots, policy, seed, *, previous_strict=()):
    """Return ``(sorted_positions, reserved_extra_edit_positions)``.

    ``editable`` is the block-local eligible set.  ``resolved`` is a block-long
    boolean list.  ``fresh`` is the set of positions that must be hard-selected.
    ``last_write`` and ``last_query`` are block-long integer lists with ``-1``
    for "never".  ``clock`` is the current step; writes at ``clock`` are
    allowed for a mid-round caller.

    ``budget`` is the nominal Q.  ``edit_slots`` is the number of non-fresh
    resolved seats reserved for editing; ``hard_noedit`` requires zero.
    ``policy`` is one of ``hard_frontier``, ``hard_random``, ``hard_noedit``.
    ``seed`` is used only by ``hard_random``.

    The second return value contains only the explicitly reserved edit seats,
    not edit positions selected later by shared backfill.  This lets a caller
    measure the policy's edit precision separately from backfill.
    """
    eligible = sorted(set(editable))
    fresh = set(fresh)
    block = len(resolved)
    if len(last_write) != block or len(last_query) != block:
        raise ValueError("resolved, last_write and last_query must have block length")
    if any(i < 0 or i >= block for i in eligible):
        raise ValueError("editable position outside block")
    if not fresh.issubset(eligible):
        raise ValueError("every fresh position must be editable")
    if budget < 0 or edit_slots < 0:
        raise ValueError("budget and edit_slots must be non-negative")
    if policy not in _POLICIES:
        raise ValueError(f"unknown hard policy: {policy}")
    if policy == "hard_noedit" and edit_slots:
        raise ValueError("hard_noedit requires edit_slots=0")
    if policy != "hard_noedit" and not edit_slots:
        raise ValueError("hard_frontier/hard_random require at least one edit seat")
    if clock < 0:
        raise ValueError("clock must be non-negative")
    if any(write < -1 or write > clock for write in last_write):
        raise ValueError("last_write outside [-1, clock]")
    if any(query < -1 or query > clock for query in last_query):
        raise ValueError("last_query outside [-1, clock]")

    cap = min(len(eligible), max(budget, len(fresh)))
    selected = set(fresh)

    # 1. Mandatory MASK: one unselected unresolved position, lowest index.
    if len(selected) < cap:
        for i in eligible:
            if not resolved[i] and i not in selected:
                selected.add(i)
                break

    # 2. Reserve edit seats from the non-fresh resolved pool.
    edit_pool = [i for i in eligible
                 if resolved[i] and i not in fresh and i not in selected]
    if policy == "hard_noedit":
        ranked = []
    elif policy == "hard_random":
        ranked = list(edit_pool)
        random.Random(f"{seed}:{clock}").shuffle(ranked)
    else:
        scores = _frontier_scores(edit_pool, last_write, last_query, clock)
        ranked = sorted(edit_pool, key=lambda i: (-scores[i], i))

    if policy in ("hard_slack", "hard_trigger"):
        # Slack is measured before the first mandatory MASK was taken.
        # Both policies spend slack first, so it never displaces a MASK.
        masks = sum(not resolved[i] and i not in fresh for i in eligible)
        slack = max(0, cap - len(fresh) - masks)
        limit = min(edit_slots, slack)
        if policy == "hard_trigger" and slack == 0 and previous_strict:
            # Release the extra seat to MASK unless a previous real substitution
            # provides a dirty neighbour within four positions of an old row.
            ranked = [i for i in ranked if any(0 < abs(i-j) <= 4
                      and last_write[j] >= last_query[i] for j in previous_strict)]
            limit = min(edit_slots, 1)
        edit_slots = limit
    take = min(edit_slots, cap - len(selected), len(ranked))
    reserved = ranked[:take]
    selected.update(reserved)

    # 3. Give the remaining seats to MASK, in neutral positional order.
    if len(selected) < cap:
        for i in eligible:
            if len(selected) >= cap:
                break
            if not resolved[i] and i not in selected:
                selected.add(i)

    # 4. Shared backfill: outside the reserved seats, use neutral ascending
    #    positions for every policy.  This isolates the policy's reserved set.
    if len(selected) < cap:
        for i in eligible:
            if len(selected) >= cap:
                break
            if resolved[i] and i not in fresh and i not in selected:
                selected.add(i)

    return sorted(selected), sorted(reserved)

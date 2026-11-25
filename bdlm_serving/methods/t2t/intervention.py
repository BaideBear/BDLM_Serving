"""Single-seat, logit-free intervention on top of ``frontier.select_hard``.

The helper is pure CPU.  It reads only block-local positions, the resolved
mask, the mandatory fresh set and the two event clocks.  It never reads
logits, token identities, model weights or KV state.

``plan_seat`` builds three same-size Q sets from one no-edit baseline:

* ``base``     -- ``select_hard(..., edit_slots=0, policy="hard_noedit")``.
* ``targeted`` -- ``base`` with one seat swapped for a candidate that the
  targeted policy actually reserved outside ``base``.
* ``random``   -- the same swap, but the added seat is drawn uniformly from
  all non-fresh old positions outside ``base`` (it may equal ``candidate``).

The targeted policy is chosen by ``model``: LLaDA uses
``hard_slack_geometry`` with at most one reserved seat; Gemma uses
``hard_trigger`` with at most four.  The policy's own gate is applied by
``select_hard``; this module never forces a reservation when the gate is
closed.  If the targeted reservation is empty or entirely inside ``base``,
there is no physical difference and ``None`` is returned -- no event is
manufactured.

Deterministic rules
-------------------
Candidate: among ``targeted_reserved - base`` take the lowest position.
Victim: among ``base - targeted_raw`` prefer resolved, non-fresh old
positions and take the highest one (the shadow-swap rule); otherwise take the
highest MASK from that difference, but only if ``base`` has at least two
MASKs, so at least one MASK survives.  If neither is legal the result is
``None``.  All fresh positions are preserved by construction.

The random arm uses the same victim as the targeted arm and samples one
position uniformly from all non-fresh old positions outside ``base``.  The
sampling is seeded by ``f"seat-random:{seed}:{clock}"`` so a given request and
step are reproducible.
"""

from __future__ import annotations

import random
from numbers import Integral

from .frontier import select_hard


__all__ = ["plan_seat"]


# Model -> (targeted policy, maximum explicitly reserved edit seats).
# The caps are the preregistered LLaDA Q4 / Gemma Q32 intervention settings.
# ``select_hard`` still applies its own slack/trigger gate on top of the cap.
_TARGETED = {
    "llada": ("hard_slack_geometry", 1),
    "gemma": ("hard_trigger", 4),
}


def _targeted_for(model):
    """Return ``(policy, edit_slots)`` for a model name."""
    compact = "".join(ch for ch in str(model).strip().lower() if ch.isalnum())
    if "llada" in compact:
        return _TARGETED["llada"]
    if "gemma" in compact:
        return _TARGETED["gemma"]
    raise ValueError(f"unsupported model: {model!r}")


def _int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _position_set(values, name, block):
    """Validate and normalise an iterable of block-local positions."""
    out = set()
    for value in values:
        position = _int(value, f"{name} position")
        if position < 0 or position >= block:
            raise ValueError(f"{name} position outside block")
        out.add(position)
    return out


def _clock_list(values, name, block, clock):
    """Validate and normalise one block-long event clock."""
    values = list(values)
    if len(values) != block:
        raise ValueError(f"{name} must have block length")
    out = []
    for value in values:
        value = _int(value, f"{name} entry")
        if value < -1 or value > clock:
            raise ValueError(f"{name} outside [-1, clock]")
        out.append(value)
    return out


def plan_seat(eligible, resolved, fresh, last_write, last_query, clock,
              budget, model, seed, previous_strict=()):
    """Return a one-seat intervention plan, or ``None`` when nothing differs.

    Parameters are block-local and positional.  ``resolved`` is a block-long
    boolean sequence; ``last_write`` and ``last_query`` are block-long integer
    sequences with ``-1`` for "never"; ``fresh`` is the mandatory hard-refresh
    set; ``previous_strict`` holds the previous step's strict T2T positions.
    ``budget`` is the nominal Q and ``model`` selects the targeted policy.

    The result is ``None`` when the targeted policy reserves no seat outside
    the no-edit baseline, or when no legal victim exists.  Otherwise it is a
    dict with sorted ``base``/``targeted``/``random`` position lists, the
    swapped ``victim``, the added ``candidate`` and ``random_candidate``, the
    ``stratum`` (``"slack"`` for an old victim, ``"mask"`` for a MASK victim),
    the sorted ``fresh`` set, the sorted eligible unresolved ``masks`` and the
    passthrough ``clock``.
    """
    clock = _int(clock, "clock")
    budget = _int(budget, "budget")
    if clock < 0:
        raise ValueError("clock must be non-negative")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    policy, edit_slots = _targeted_for(model)

    resolved_list = [bool(value) for value in resolved]
    block = len(resolved_list)
    last_write_list = _clock_list(last_write, "last_write", block, clock)
    last_query_list = _clock_list(last_query, "last_query", block, clock)

    eligible_set = _position_set(eligible, "eligible", block)
    fresh_set = _position_set(fresh, "fresh", block)
    strict_set = _position_set(previous_strict, "previous_strict", block)
    if not fresh_set <= eligible_set:
        raise ValueError("every fresh position must be eligible")
    if any(not resolved_list[i] for i in fresh_set):
        raise ValueError("every fresh position must be resolved")
    if any(i not in eligible_set or not resolved_list[i] for i in strict_set):
        raise ValueError("previous_strict positions must be eligible and resolved")

    eligible_tuple = tuple(sorted(eligible_set))
    strict_tuple = tuple(sorted(strict_set))

    # 1. The no-edit baseline.  hard_noedit has no reserved edit seats.
    base_positions, base_reserved = select_hard(
        eligible_tuple, resolved_list, fresh_set, last_write_list,
        last_query_list, clock, budget, 0, "hard_noedit", seed,
        previous_strict=strict_tuple,
    )
    base_positions = list(base_positions)
    base_reserved = list(base_reserved)
    base_set = set(base_positions)
    if base_reserved:
        raise ValueError("hard_noedit returned reserved edit seats")
    if len(base_set) != len(base_positions):
        raise ValueError("hard_noedit returned duplicate positions")
    if not fresh_set <= base_set:
        raise ValueError("hard_noedit dropped mandatory fresh positions")

    # 2. The targeted policy, with its original gate and seat cap.
    targeted_positions, targeted_reserved = select_hard(
        eligible_tuple, resolved_list, fresh_set, last_write_list,
        last_query_list, clock, budget, edit_slots, policy, seed,
        previous_strict=strict_tuple,
    )
    targeted_positions = list(targeted_positions)
    targeted_reserved = list(targeted_reserved)
    targeted_set = set(targeted_positions)
    if len(targeted_set) != len(targeted_positions):
        raise ValueError("select_hard returned duplicate positions")
    if not fresh_set <= targeted_set:
        raise ValueError("select_hard dropped mandatory fresh positions")
    reserved_set = set(targeted_reserved)
    if len(reserved_set) != len(targeted_reserved):
        raise ValueError("select_hard returned duplicate reserved positions")
    if not reserved_set <= targeted_set:
        raise ValueError("select_hard reserved positions are not selected")
    if any(i in fresh_set for i in reserved_set):
        raise ValueError("select_hard reserved a mandatory fresh position")

    # 3. A candidate must be a real targeted reservation outside the baseline.
    #    An empty difference means there is no physical event to report.
    candidate_pool = sorted(reserved_set - base_set)
    if not candidate_pool:
        return None
    candidate = candidate_pool[0]

    # 4. Sacrifice a position that the targeted policy actually dropped.
    #    Prefer old rows; otherwise displace a MASK only when another MASK
    #    remains in the baseline.
    dropped = base_set - targeted_set
    old_victims = sorted(
        i for i in dropped if resolved_list[i] and i not in fresh_set
    )
    mask_victims = sorted(i for i in dropped if not resolved_list[i])
    base_masks = [i for i in base_set if not resolved_list[i]]
    if old_victims:
        victim = old_victims[-1]
        stratum = "slack"
    elif mask_victims and len(base_masks) >= 2:
        victim = mask_victims[-1]
        stratum = "mask"
    else:
        return None

    # 5. Uniform random control over every non-fresh old row outside base.
    #    The targeted candidate is deliberately eligible for collision.
    random_pool = sorted(
        i for i in eligible_set
        if resolved_list[i] and i not in fresh_set and i not in base_set
    )
    if not random_pool:
        return None
    rng = random.Random(f"seat-random:{seed}:{clock}")
    random_candidate = rng.choice(random_pool)

    targeted = sorted((base_set - {victim}) | {candidate})
    random_selection = sorted((base_set - {victim}) | {random_candidate})
    if len(targeted) != len(base_set) or len(random_selection) != len(base_set):
        raise RuntimeError("single-seat swap changed the Q size")

    return {
        "base": sorted(base_set),
        "targeted": targeted,
        "random": random_selection,
        "victim": victim,
        "candidate": candidate,
        "random_candidate": random_candidate,
        "stratum": stratum,
        "fresh": sorted(fresh_set),
        "masks": sorted(i for i in eligible_set if not resolved_list[i]),
        "clock": clock,
    }

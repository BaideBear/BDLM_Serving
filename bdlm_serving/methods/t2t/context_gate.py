"""One physical exchange at Q8; separate when to exchange from who enters."""
import math
import random

from .hazard import features

GATES = ("hard_context_slack", "hard_context_slack_random", "hard_context_gate",
         "hard_context_gate_random")
ONCE = tuple("hard_context_once_"+arm for arm in ("base", "rank", "random"))
CONTEXT = ("hard_context_fixed", *GATES, *ONCE)
CUTOFF = math.log(.05 / .95)


def select(valid, resolved, fresh, writes, strict, clock, q, mode, seed, block, score_rows):
    if q != 8 or mode not in GATES:
        raise ValueError("Context gate study uses Q8, E1")
    valid = sorted(valid)
    fresh = set(fresh)
    old = [i for i in valid if resolved[i] and i not in fresh]
    masks = [i for i in valid if not resolved[i]]
    size = min(len(valid), max(q, len(fresh)))
    free = size - len(fresh)
    base = sorted(fresh.union((masks + old)[:free]))
    event = dict(base=base, fresh=sorted(fresh), masks=masks, old_pool=old,
                 rank=None, score=None, victim=None, entrant=None)
    slack = free > len(masks)
    if len(fresh) > q:
        reason = "overflow"
    elif not free:
        reason = "no_room"
    elif not old:
        reason = "no_old"
    elif not slack and free < 2:
        reason = "protected_mask"
    elif not slack and mode.startswith("hard_context_slack"):
        reason = "no_slack"
    else:
        pos = [features(i, masks, fresh, strict, writes, clock) for i in old]
        scores = [float(v) for v in score_rows(old, pos)]
        if len(old) != len(scores) or not all(map(math.isfinite, scores)):
            raise ValueError("Invalid context risk scores")
        neg, top = min((-v, i) for i, v in zip(old, scores, strict=True))
        event.update(rank=top, score=-neg)
        if not slack and -neg < CUTOFF:
            reason = "below_threshold"
        elif top in base:
            reason = "already_selected"
        else:
            victim = max(set(base).intersection(old if slack else masks))
            entrant = (random.Random(f"context-swap:{seed}:{block}:{clock}").choice(
                sorted(set(old) - set(base))) if mode.endswith("_random") else top)
            event.update(reason="swap_old" if slack else "swap_mask", victim=victim, entrant=entrant)
            return sorted(set(base) - {victim} | {entrant}), [entrant], event
    event["reason"] = reason
    return base, [], event


def crowded_plan(move, seed, block, clock):
    """Only a real MASK displacement is an anchor; earlier slack is not spent."""
    if move["reason"] != "swap_mask":
        return None
    base, victim, top = move["base"], move["victim"], move["rank"]
    pool = sorted(set(move["old_pool"]) - set(base))
    other = random.Random(f"context-swap:{seed}:{block}:{clock}").choice(pool)
    def replace(i):
        return sorted(set(base) - {victim} | {i})
    return dict(base=base, targeted=replace(top), random=replace(other), victim=victim,
                candidate=top, random_candidate=other, fresh=move["fresh"], masks=move["masks"],
                stratum="mask", clock=clock, risk_score=move["score"])

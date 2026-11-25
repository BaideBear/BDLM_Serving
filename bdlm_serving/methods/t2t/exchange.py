"""B32: compare the predicted immediate event on each side of a physical swap."""
from bisect import bisect_left
import json
import math
from random import Random


MODES = {"hard_exchange_equal": 1., "hard_exchange_edit4": 4.}
PLACEBOS = {"hard_exchange_random_a": "a", "hard_exchange_random_b": "b"}
AGE_ARMS = {"hard_exchange_oldhalf_a": "a", "hard_exchange_oldhalf_b": "b"}
ONLINE_MODES = {**MODES, **dict.fromkeys(PLACEBOS, 4.), **dict.fromkeys(AGE_ARMS, 4.)}
SCHEMA = "llada-b32-exchange-v1"


def position_keys(valid, resolved, fresh, strict, writes, clock):
    masks = [i for i in valid if not resolved[i]]
    old = [i for i in valid if resolved[i] and i not in fresh]

    def near(i, group):
        return bisect_left((1, 3, 8), min((abs(i-j) for j in group), default=64))

    keys = {i: ("old", near(i, fresh), near(i, masks), int(near(i, strict) < 2),
                bisect_left((2, 4, 8), clock-writes[i] if writes[i] >= 0 else 64)) for i in old}
    keys.update({i: ("mask", bisect_left((0, 1, 3, 7), rank), near(i, fresh),
                     bisect_left((4, 12, 24), len(masks))) for rank,i in enumerate(masks)})
    return keys


def decode(artifact):
    if artifact["schema"] != SCHEMA:
        raise ValueError("Wrong exchange calibration")
    probs = {tuple(r["key"]):r["probability"] for r in artifact["rates"]}
    probs.update({(k,):p for k,p in artifact["fallback"].items()})
    if set(artifact["fallback"]) != {"old","mask"} or any(
            not math.isfinite(p) or not 0 <= p <= 1 for p in probs.values()):
        raise ValueError("Invalid exchange rates")
    return probs


def read_rates(path):
    if path is None:
        raise ValueError("Exchange requires explicit frozen rates")
    return decode(json.loads(path.read_text()))


def choose(valid, resolved, fresh, strict, writes, clock, q, slots, rates, edit_value):
    """Old backfill competes with old first. If none existed, protect the first MASK."""
    fresh = set(fresh)
    size = min(len(valid), max(q, len(fresh)))
    masks = [i for i in valid if not resolved[i]]
    old = [i for i in valid if resolved[i] and i not in fresh]
    base = set(fresh) | set((masks + old)[:size-len(fresh)])
    outside = [i for i in old if i not in base]
    backfill = [i for i in old if i in base]
    # A backfill round may replace only baseline old rows. Never displace MASK
    # after exhausting them. In a crowded round, replace the marginal MASK.
    victims = backfill or list(reversed([i for i in masks[1:] if i in base]))
    if slots <= 0 or not outside or not victims:
        return sorted(base), [], []
    keys = position_keys(valid, resolved, fresh, strict, writes, clock)
    def prob(i):
        k = keys[i]
        return rates.get(k, rates[(k[0],)])
    if backfill:
        victims = sorted(victims, key=lambda i:(prob(i), -i))
    outside.sort(key=lambda i:(-prob(i), i))
    offers, reserve = [], []
    for candidate,victim in zip(outside[:slots], victims):
        accepted = prob(candidate) * (1. if backfill else edit_value) > prob(victim)
        offers.append((candidate, victim, accepted))
        if not accepted:
            break
        base.remove(victim)
        base.add(candidate)
        reserve.append(candidate)
    assert len(base) == size and fresh <= base
    return sorted(base), sorted(reserve), offers


def oldest_half(pool, last_query, seed):
    """Uniform within the oldest half, including uniform boundary ties."""
    ordered = sorted(pool, key=last_query.__getitem__)
    m = (len(pool)+1)//2
    if last_query[ordered[0]] < 0:
        raise ValueError("Old candidate has no observed Q")
    cutoff = last_query[ordered[m-1]]
    if cutoff == last_query[ordered[0]] == last_query[ordered[-1]]:
        return pool
    older = [i for i in ordered if last_query[i] < cutoff]
    tied = [i for i in ordered if last_query[i] == cutoff]
    return sorted(older + Random(seed+":half-tie").sample(tied,m-len(older)))


def randomize_candidate(valid, resolved, fresh, ranked, reserved, offers, seed, *, last_query=None):
    """Keep every accepted offer's victim and width; only its replacement is randomized."""
    if not reserved:
        return ranked, reserved, None
    victims = sorted(v for _, v, accepted in offers if accepted)
    original = (set(ranked) - set(reserved)) | set(victims)
    pool = sorted(i for i in valid if resolved[i] and i not in fresh and i not in original)
    if not set(reserved) <= set(pool) or len(victims) != len(reserved):
        raise ValueError("Invalid ranked exchange")
    draw_pool = pool if last_query is None else oldest_half(pool,last_query,seed)
    if last_query is not None and len(reserved) != 1:
        raise ValueError("Cache-age experiment requires one entering seat")
    random_old = sorted(Random(seed).sample(draw_pool, len(reserved)))
    picked = sorted((original - set(victims)) | set(random_old))
    if len(picked) != len(ranked) or not set(fresh) <= set(picked):
        raise ValueError("Placebo changed width or displaced a mandatory row")
    audit = dict(base=sorted(original), ranked=list(ranked), executed=picked,
        victims=victims, pool=pool, fresh=sorted(fresh),
        masked_victims=[v for v in victims if not resolved[v]])
    if last_query is not None:
        audit.update(draw_pool=draw_pool, last_query=[last_query[i] for i in pool])
    return picked, random_old, audit

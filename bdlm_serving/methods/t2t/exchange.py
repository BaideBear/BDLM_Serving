"""B32: compare the predicted immediate event on each side of a physical swap."""
from bisect import bisect_left
import json
import math


MODES = {"hard_exchange_equal": 1., "hard_exchange_edit4": 4.}
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

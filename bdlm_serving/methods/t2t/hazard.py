"""Positional risk only. No decoded token ID, logits, confidence, KV or task ID."""
import json
import math


def features(i, masks, fresh, strict, write, clock):
    def proximity(group):
        d = min((abs(i-j) for j in group if i != j), default=64)
        return [float(d <= k) for k in (1, 2, 4, 8)]
    age = clock-write[i] if write[i] >= 0 else 64
    near_mask, near_fresh = proximity(masks), proximity(fresh)
    return [1., i/31., len(masks)/32., len(fresh)/32., *near_mask, *near_fresh,
            *proximity(strict), *[float(age <= k) for k in (2, 3, 4, 8)],
            float(any(0 < j-i <= 4 for j in fresh)),
            float(any(0 < i-j <= 4 for j in fresh)),
            near_mask[2]*near_fresh[2],
            sum(abs(i-j) <= 4 for j in fresh)/8.,
            float(bool(strict))]


def read_weights(path):
    if path is None:
        raise ValueError("Calibrated hazard requires explicit, frozen weights")
    artifact = json.loads(path.read_text())
    w = artifact["weights"]
    if artifact["schema"] != "llada-positional-hazard-v1" or len(w) != 25 or not all(map(math.isfinite,w)):
        raise ValueError("Unexpected positional calibration")
    return tuple(w)


def choose(valid, resolved, fresh, writes, strict, clock, q, cap, weights, threshold=.05):
    masks = [i for i in valid if not resolved[i]]
    old = [i for i in valid if resolved[i] and i not in fresh]
    size = min(len(valid), max(q, len(fresh)))
    free = size-len(fresh)
    old_cap = max(0,min(cap,free-int(bool(masks))))
    slack = max(0,free-len(masks))
    if old_cap:
        ranks = sorted(((-sum(v*w for v,w in zip(features(i,masks,fresh,strict,writes,clock),weights,strict=True)),i)
                        for i in old))
        cutoff = -math.inf if threshold == 0 else math.log(threshold/(1.-threshold))
        reserve = [i for j,(neg,i) in enumerate(ranks[:old_cap]) if j < slack or -neg >= cutoff]
    else:
        reserve = []
    selected = set(fresh) | set(reserve) | set(masks[:free-len(reserve)])
    for i in old:
        if len(selected) == size:
            break
        selected.add(i)
    return sorted(selected), sorted(reserve)

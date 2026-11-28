"""Paired, request-weighted one-decision response baseline. No GPU or future inputs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite, log2, sqrt
from statistics import mean, stdev


MENUS = {"llada21": (4, 8, 16, 32, 64), "gemma": (32, 64, 128, 256, 512)}
FEATURES = ("head_mask_rows", "dirty_rows", "resident_rows", "tail_open", "prev_q",
            "recent_m2t", "recent_t2t", "phase")
# Only native controller states known *before* selecting an offer.
PHASES = {"llada21": {"bootstrap", "rolling", "closing", "debt", "rebuild"},
          "gemma": {"sample", "closing", "verify", "rebuild"}}


def observed_state(state):
    # Old sealed receipts put an implementation-specific epoch inside the state.
    # Keep their digest intact; never require, read or use that field for distance.
    if set(state) not in (set(FEATURES), set(FEATURES) | {"epoch"}):
        raise ValueError("Wrong pre-decision feature schema")
    return {k: state[k] for k in FEATURES}


def regime(state):
    # A successor with no KV can already be labelled "rolling" by the controller.
    # Keep this apart from an ordinary partially denoised, resident head.
    return (state["phase"], state["tail_open"], state["resident_rows"] == 0)


def count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"Invalid {name}: {value!r}")
    return value


def vector(state, model):
    """The whitelist is deliberately exact: fail closed on unknown or missing features."""
    state = observed_state(state)
    block = MENUS[model][-2]
    for name in ("head_mask_rows", "dirty_rows", "resident_rows", "recent_m2t", "recent_t2t"):
        count(state[name], name)
    if (state["head_mask_rows"] > block or state["resident_rows"] > block
            or any(state[k] > 2*block for k in ("dirty_rows", "recent_m2t", "recent_t2t"))
            or type(state["tail_open"]) is not bool or not isinstance(state["phase"], str)
            or not state["phase"] or state["prev_q"] not in (*MENUS[model], None)):
        raise ValueError("Invalid pre-decision state")
    return (state["head_mask_rows"]/block, state["dirty_rows"]/(2*block),
            state["resident_rows"]/block, state["recent_m2t"]/(2*block),
            state["recent_t2t"]/(2*block),
            0 if state["prev_q"] is None else (1+log2(state["prev_q"]/MENUS[model][0]))/5)


@dataclass(frozen=True)
class Fork:
    family: tuple
    request: str
    digest: str
    regime: tuple
    x: tuple | None
    y: tuple
    quality: tuple


def read_fork(data):
    """One row is one verified complete five-arm GPU fork, not five unrelated requests."""
    model = data["model"]
    offers = MENUS[model]
    scope = (model, data["executor_id"], data["continuation_q"])
    req, digest = data["request_id"], data["state_digest"]
    if (not isinstance(scope[1], str) or not scope[1] or scope[2] not in offers
            or not isinstance(req, str) or not req or not isinstance(digest, str)
            or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("Invalid executor, continuation, request or state identity")
    s = observed_state(data["state"])
    missing = [k for k in FEATURES if k != "prev_q" and s[k] is None]
    # Validate all observed fields; missing telemetry never enters a distance.
    checked = dict(s)
    for k in missing:
        checked[k] = False if k == "tail_open" else "unobserved" if k == "phase" else 0
    v = vector(checked, model)
    x = None if missing or s["phase"] not in PHASES[model] else v
    n0 = count(data["pre_delivered_n"], "pre_delivered_n")
    arms = {a["q"]: a for a in data["arms"]}
    if set(arms) != set(offers) or len(data["arms"]) != len(offers):
        raise ValueError("Expected one complete matched fork; retain incomplete receipts separately")
    y, quality = [], []
    for q in offers:
        a = arms[q]
        if a["state_digest"] != digest or a["terminal"] not in ("eos", "truncated", "decode_failure"):
            raise ValueError("Unmatched state or unpriced endpoint; do not fit")
        n = count(a["final_n"], "final_n")
        f = count(a["suffix_decoder_f"], "suffix_decoder_f") + count(a["suffix_encoder_f"], "suffix_encoder_f")
        if (n < n0 or f == 0 or type(a["correct"]) is not bool
                or (a["terminal"] == "decode_failure" and a["correct"])):
            raise ValueError("Inconsistent endpoint")
        y.append((n-n0)/f)
        quality.append(a["correct"])
    return Fork(scope, req, digest, regime(s), x, tuple(y), tuple(quality))


def curve(train, family, state, *, allowed_q=(), neighbours=12, min_requests=4,
          max_distance=.25, exclude_request=None, static=False):
    """Nearest same-lifecycle state per request, or request-weighted state-blind comparator.

    Distance is mean absolute distance in normalized coordinates. The paired SEs
    describe dispersion of neighbour means, not a calibrated guarantee on this request.
    allowed_q must come from an external, independently validated quality gate.
    """
    offers = MENUS[family[0]]
    if (not 1 <= min_requests <= neighbours or not 0 <= max_distance <= 1
            or set(allowed_q) - set(offers)):
        raise ValueError("Invalid prediction settings")
    state = observed_state(state)
    # A first call may have no previous offer. Other missing values are unknown,
    # not evidence of zero debt/edits. Return the full menu without predictions.
    incomplete = any(state[k] is None for k in FEATURES if k != "prev_q")
    if not static and (incomplete or state["phase"] not in PHASES[family[0]]):
        return dict(status="unsupported", reason="incomplete_state" if incomplete else "unobserved_lifecycle",
                    distinct_requests=0,
                    points=[dict(q=q, expected_suffix_tpf=None, quality_qualified=q in allowed_q,
                                 eligible=False) for q in offers],
                    adjacent=[dict(lo=lo, hi=hi, upgrade_gain=None, downshift_loss=None, paired_se=None)
                              for lo, hi in zip(offers, offers[1:])])
    x = vector(state, family[0]) if not static else ()
    groups = defaultdict(list)
    seen = set()
    for r in train:
        if r.family != family or r.request == exclude_request:
            continue
        key = (r.request, r.digest)
        if key in seen:
            raise ValueError("Duplicate decision in training set")
        seen.add(key)
        if not static and (r.x is None or r.regime != regime(state)):
            continue
        d = 0. if static else mean(abs(a-b) for a, b in zip(r.x, x))
        if static or d <= max_distance:
            groups[r.request].append((d, r.digest, r))
    if static:
        ys = [tuple(map(mean, zip(*(r.y for _, _, r in vs)))) for vs in groups.values()]
    else:
        nearest = sorted((min(vs, key=lambda v: v[:2]) for vs in groups.values()),
                         key=lambda v: (v[0], v[1], v[2].request))
        ys = [r.y for _, _, r in nearest[:neighbours]]
    n = len(ys)
    supported = n >= min_requests
    p = list(map(mean, zip(*ys))) if supported else [None]*len(offers)
    points = [dict(q=q, expected_suffix_tpf=v, quality_qualified=q in allowed_q,
                   eligible=supported and q in allowed_q) for q, v in zip(offers, p)]
    margins = []
    for j, (lo, hi) in enumerate(zip(offers, offers[1:])):
        values = [v[j+1]-v[j] for v in ys]
        # Positive loss means that reducing Q is costly; may be negative.
        gain = mean(values) if supported else None
        margins.append(dict(lo=lo, hi=hi, upgrade_gain=gain, downshift_loss=gain,
                            paired_se=stdev(values)/sqrt(n) if supported and n > 1 else None))
    return dict(status="supported" if supported else "unsupported", distinct_requests=n,
                points=points, adjacent=margins)


def recommend_adjacent(result, current_q, feasible_q, *, se_margin=1.0):
    """Optional single-request comparison; the caller owns all hard budget decisions.

    If the current tier becomes infeasible, do not use an unqualified point as a
    fallback. A positive SE margin is an engineering guard, not a confidence level.
    """
    points = result["points"]
    tiers = {p["q"] for p in points}
    feasible = set(feasible_q)
    if current_q not in tiers or feasible-tiers or not isfinite(se_margin) or se_margin < 0:
        raise ValueError("Invalid current offer, budget menu or guard")
    allowed = feasible & {p["q"] for p in points if p["quality_qualified"]}
    if current_q not in allowed:
        return {"q": None, "reason": "outer_or_quality_constraint"}
    if result["status"] != "supported":
        return {"q": current_q, "reason": "no_state_support"}
    choices = []
    for pair in result["adjacent"]:
        lo, hi = pair["lo"], pair["hi"]
        if current_q not in (lo, hi) or pair["paired_se"] is None:
            continue
        next_q = hi if current_q == lo else lo
        gain = pair["upgrade_gain"] if current_q == lo else -pair["downshift_loss"]
        guard = gain - se_margin*pair["paired_se"]
        if next_q in allowed and guard > 0:
            choices.append((guard, -next_q))
    return ({"q": -max(choices)[1], "reason": "paired_gain"} if choices else
            {"q": current_q, "reason": "no_supported_improvement"})

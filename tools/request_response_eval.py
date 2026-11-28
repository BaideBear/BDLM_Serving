"""Leave-one-request-out diagnostic on complete, audited GPU fork JSONL records."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from bdlm_serving.methods.t2t.request_response import curve, read_fork


def errors(truth, result):
    pred = [p["expected_suffix_tpf"] for p in result["points"]]
    # No quality authorization here: regret is a diagnostic over the measured menu.
    j = max(range(len(pred)), key=pred.__getitem__)
    margins = list(zip([b-a for a, b in zip(truth, truth[1:])],
                       [b-a for a, b in zip(pred, pred[1:])]))
    non_ties = [(y, p) for y, p in margins if y != 0]
    return dict(mae=mean(abs(y-p) for y, p in zip(truth, pred)),
                marginal_mae=mean(abs(y-p) for y, p in margins), regret=max(truth)-truth[j],
                non_ties=len(non_ties), sign_hits=sum(y*p > 0 for y, p in non_ties))


def summarize(scores):
    # Multiple histories / anchored states on one request do not give it extra votes.
    if not scores:
        return None
    out = {k: mean(mean(v[k] for v in vals) for vals in scores.values())
           for k in ("mae", "marginal_mae", "regret")}
    hits, denom = [sum(sum(v[k] for v in vals) for vals in scores.values())
                   for k in ("sign_hits", "non_ties")]
    return dict(out, differing_measured_margins=denom,
                adjacent_direction_accuracy=hits/denom if denom else None)


def evaluate(records, **settings):
    families = defaultdict(list)
    seen = set()
    for data in records:
        r = read_fork(data)
        key = (r.family, r.request, r.digest)
        if key in seen:
            raise ValueError("Duplicate matched fork")
        seen.add(key)
        families[r.family].append((r, data["state"]))
    reports = []
    for family, entries in sorted(families.items()):
        scores, base = defaultdict(list), defaultdict(list)
        train = [r for r, _ in entries]
        for r, s in entries:
            conditioned = curve(train, family, s, exclude_request=r.request, **settings)
            if conditioned["status"] != "supported":
                continue
            comparator = curve(train, family, s, exclude_request=r.request, static=True, **settings)
            scores[r.request].append(errors(r.y, conditioned))
            base[r.request].append(errors(r.y, comparator))
        reports.append(dict(model=family[0], executor_id=family[1], continuation_q=family[2],
                            total_decisions=len(entries), supported_decisions=sum(map(len, scores.values())),
                            total_requests=len({r.request for r in train}), supported_requests=len(scores),
                            state_model=summarize(scores), budget_only_on_same_support=summarize(base)))
    return reports


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl", nargs="+", type=Path)
    args = p.parse_args()
    records = (json.loads(line) for path in args.jsonl for line in path.read_text().splitlines() if line.strip())
    print(json.dumps(evaluate(records), indent=2, allow_nan=False))

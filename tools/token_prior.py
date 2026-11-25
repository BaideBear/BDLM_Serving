"""Rank already accepted rows by a calibration-only token correction to positional risk."""
import argparse
from collections import Counter, defaultdict
import json
from math import log
from pathlib import Path

import numpy as np
from scipy.special import expit

from bdlm_serving.evaluation.frozen import MANIFEST, ROOT, sha
from tools.edit_opportunity import load
from tools.lexical_hazard import BUDGETS, check_split, measurements, write_rows


def residual_rates(paths, weights):
    expected, observed = Counter(), Counter()
    for path in paths:
        for stage, _, _, old, _, _, _, features, labels in path:
            if stage != "selected" or not old:
                continue
            for i, token, p in zip(old, features[:, -1], expit(features[:, :-1] @ weights), strict=True):
                expected[int(token)] += p
                observed[int(token)] += int(i in labels)
    return {t: log((1 + observed[t]) / (1 + e)) for t, e in expected.items()}


def study(cal, native, out):
    frozen = ROOT / "experiments/t2t/calibrated/weights.json"
    if sha(cal / "weights.json") != sha(frozen):
        raise ValueError("Wrong calibration artifact")
    if json.loads((native/"verified.json").read_text()) != {"model": "llada21", "matched_requests": 30}:
        raise ValueError("Missing frozen30 native qualification")
    artifact = json.loads(frozen.read_text())
    weights = np.asarray(artifact["weights"])
    train, test = artifact["samples"], json.loads(MANIFEST.read_text())["samples"]
    check_split(train, test)

    def read(folder, task, encode):
        return load(folder / f"{task['index']:02d}.trace.jsonl.gz", task, old_token_features=encode)

    rates = residual_rates((read(cal, t, lambda token: (token,)) for t in train), weights)
    sums, counts, rows = defaultdict(Counter), defaultdict(Counter), []
    for t in test:
        path = read(native, t, lambda token: (rates.get(token, 0.0),))
        base = dict(measurements(path, np.r_[weights, 0.]))
        for metric, candidate in measurements(path, np.r_[weights, 1.]):
            ref = base[metric]
            if (ref["targets"], ref["seats"]) != (candidate["targets"], candidate["seats"]):
                raise ValueError("Unmatched denominator")
            row = dict(base_hits=ref["hits"], token_hits=candidate["hits"],
                       targets=ref["targets"], seats=ref["seats"])
            rows.append(dict(index=t["index"], dataset=t["dataset"], metric=metric, **row))
            sums[metric].update(row)
            d = candidate["hits"] - ref["hits"]
            counts[metric]["requests_up" if d > 0 else "requests_down" if d < 0 else "requests_equal"] += 1
    summary = [dict(metric=m, **s, **{k: counts[m][k] for k in
                    ("requests_up", "requests_equal", "requests_down")}) for m, s in sums.items()]
    if any(s["targets"] != 191 for s in sums.values()):
        raise ValueError("Wrong native target cohort")
    out.mkdir(exist_ok=True, parents=True)
    write_rows(out / "requests.csv", rows)
    write_rows(out / "summary.csv", summary)
    changes = [sums[m]["token_hits"] - sums[m]["base_hits"] for m in (f"Q{q}/{e}" for q, e in BUDGETS)]
    print("Budget hit differences:", changes, "Advance:", min(changes) >= 0 and sum(v > 0 for v in changes) >= 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("cal", "native", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    study(**vars(parser.parse_args()))

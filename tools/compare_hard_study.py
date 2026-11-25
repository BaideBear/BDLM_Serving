"""Compare completed hard-v1 arms with the original same-window IBS/OBS."""
from collections import defaultdict
import csv
from fractions import Fraction
from math import isclose
from pathlib import Path
from statistics import mean

from render_hard_study import MODELS, HARD_POLICY


ROOT = Path(__file__).resolve().parents[1]


def table(path, rows):
    """Keep this analysis of published CSVs independent of model/scoring packages."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def by_request(rows, manifest):
    keys = [(int(r["index"]), r["dataset"], r["task_id"]) for r in rows]
    if len(keys) != 30 or set(keys) != manifest:
        raise ValueError("Comparison requires the same 30 frozen requests")
    return {key: (int(r["correct"]), Fraction(int(r["N"]), int(r["F"])))
            for key, r in zip(keys, rows, strict=True)}


def compare(arm, ref, a, b):
    ac, bc = (sum(c for c, _ in x.values()) for x in (a, b))
    at, bt = (mean(t for _, t in x.values()) for x in (a, b))
    for c, t, row in ((ac, at, arm), (bc, bt, ref)):
        if c != int(row["correct"]) or not isclose(float(t), float(row["tpf"]), rel_tol=1e-12):
            raise ValueError(f"Request/summary mismatch: {row['model']} {row['policy']}")
    quality = [a[k][0] - b[k][0] for k in a]
    delta_tpf = [a[k][1] - b[k][1] for k in a]
    dc, dt = ac - bc, mean(delta_tpf)
    outcome = ("both_improve" if dc > 0 and dt > 0 else
               "quality_tie_tpf_improves" if dc == 0 and dt > 0 else
               "tradeoff" if dc * dt < 0 else "no_joint_gain")
    both = sum(a[k][0] and b[k][0] for k in a)
    return dict(model=arm["model"], nominal_q=int(arm["nominal_q"]), policy=arm["policy"],
                reference=ref["policy"], reference_run=ref["run"], requests=30,
                correct=ac, reference_correct=bc, delta_correct=dc,
                tpf=arm["tpf"], reference_tpf=ref["tpf"], delta_tpf=float(dt),
                token_efficiency=arm["token_efficiency"], reference_token_efficiency=ref["efficiency"],
                q_per_f=arm["q_per_f"], reference_q_per_f=ref["q_per_f"],
                both_correct=both, only_policy_correct=quality.count(1),
                only_reference_correct=quality.count(-1), neither_correct=30-ac-bc+both,
                tpf_positive=sum(d > 0 for d in delta_tpf), tpf_equal=delta_tpf.count(0),
                tpf_negative=sum(d < 0 for d in delta_tpf), outcome=outcome)


def main():
    import json
    manifest_rows = json.loads((ROOT / "benchmarks/frozen30/manifest.json").read_text())["samples"]
    manifest = {(r["index"], r["dataset"], r["task_id"]) for r in manifest_rows}
    hard, baselines = ROOT / "experiments/t2t/hard", ROOT / "experiments/baselines"
    baseline_summary = defaultdict(list)
    baseline_requests = defaultdict(list)
    for row in read_rows(baselines / "summary.csv"):
        if row["role"] == "baseline" and row["dataset"] == "all":
            baseline_summary[row["model"], row["policy"]].append(row)
    for row in read_rows(baselines / "requests.csv"):
        if row["role"] == "baseline":
            baseline_requests[row["model"], row["policy"], row["run"]].append(row)

    result = []
    for model, (_, width) in MODELS.items():
        summary = hard / f"{model}-online-summary.csv"
        if not summary.is_file():
            continue
        requests = defaultdict(list)
        for row in read_rows(hard / f"{model}-online-requests.csv"):
            if row["model"] == model:
                requests[row["policy"]].append(row)
        for arm in sorted(read_rows(summary), key=lambda r: (int(r["nominal_q"]), r["policy"])):
            if (arm["model"] != model or arm["dataset"] != "all"
                    or not HARD_POLICY.search(arm["policy"])):
                continue
            q = int(arm["nominal_q"])
            if q > width:
                continue
            a = by_request(requests[arm["policy"]], manifest)
            for policy in (f"ibs_{q}", f"obs_{q}"):
                refs = baseline_summary[model, policy]
                if not refs:
                    continue
                ref, = refs
                b = by_request(baseline_requests[model, policy, ref["run"]], manifest)
                result.append(compare(arm, ref, a, b))
    if not result:
        raise ValueError("No completed hard-v1 arms with same-window IBS/OBS references")
    path = hard / "macro-paired.csv"
    table(path, result)
    print(f"{path}: {len(result)} verified pairs")


if __name__ == "__main__":
    main()

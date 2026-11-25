"""Publish all measured logit-free single-block policies vs their same-Q noedit control."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.render_hard_study import ROOT, all_rows
from tools.t2t_results import table


def request_key(row):
    return int(row["index"]), row["dataset"], row["task_id"]


def stratified_draws(keys, draws=10000, seed=20261126):
    groups = defaultdict(list)
    for idx, (_, dataset, _) in enumerate(keys):
        groups[dataset].append(idx)
    if sorted(map(len, groups.values())) != [10, 10, 10] or len(set(keys)) != 30:
        raise ValueError("Expected 10 distinct frozen requests in each of 3 datasets")
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(group, (draws, len(group)), replace=True)
                           for group in groups.values()], axis=1)


def request_metrics(records, keys, summary):
    by_key = {request_key(r): r for r in records}
    if len(records) != 30 or set(keys) != set(by_key):
        raise ValueError(f"Missing, duplicate, or mismatched request in {summary['policy']}")
    a = np.array([[int(by_key[key][field]) for field in ("correct", "N", "F", "physical_Q")]
                  for key in keys])
    if np.any(a[:, 2:] <= 0) or not np.array_equal(
            a.sum(axis=0), [int(summary[k]) for k in ("correct", "N", "F", "physical_Q")]):
        raise ValueError(f"Physical accounting mismatch in {summary['policy']}")
    if not np.isclose((a[:, 1] / a[:, 2]).mean(), float(summary["tpf"]), atol=1e-12, rtol=0):
        raise ValueError(f"Request TPF mismatch in {summary['policy']}")
    return a


def paired_uncertainty(a, b, draws):
    quality = (a[:, 0] - b[:, 0])[draws].sum(axis=1)
    tpf = (a[:, 1] / a[:, 2] - b[:, 1] / b[:, 2])[draws].mean(axis=1)
    def pooled(x):
        return x[:, 1][draws].sum(axis=1) / x[:, 3][draws].sum(axis=1)
    diff = a[:, 0] - b[:, 0]
    out = dict(only_policy_correct=int((diff > 0).sum()), only_control_correct=int((diff < 0).sum()))
    for name, vals in (("delta_correct", quality), ("delta_TPF", tpf),
                       ("delta_eff_pp", 100 * (pooled(a) - pooled(b)))):
        out[f"{name}_lo"], out[f"{name}_hi"] = np.quantile(vals, [.025, .975])
    return out


def main():
    root = ROOT / "experiments/t2t"
    keys = sorted(request_key(r) for r in json.loads(
        (ROOT / "benchmarks/frozen30/manifest.json").read_text())["samples"])
    draws = stratified_draws(keys)
    rows, overview, uncertainty = [], [], []
    for model, budgets in (("llada21",(4,8,16,32,64)), ("gemma",(16,32,64,128,256))):
        measured, request_data = {}, {}
        for d in ("hard", "adaptive", "placement", "placement/geometry-online", "calibrated", "refresh", "exchange"):
            path = root / d / f"{model}-online-summary.csv"
            if not path.exists():
                continue
            by_name = defaultdict(list)
            with (path.parent / f"{model}-online-requests.csv").open(newline="") as f:
                for record in csv.DictReader(f):
                    if record["model"] == model:
                        by_name[record["policy"]].append(record)
            for r in all_rows(path):
                name = r["policy"]
                if "_hard_" not in name or name.endswith("oracle1"):
                    continue
                if name in measured or int(r["requests"]) != 30:
                    raise ValueError(f"Duplicate or incomplete {name}")
                measured[name] = r
                request_data[name] = request_metrics(by_name[name], keys, r)
        for q in budgets:
            base = measured[f"t2t_q{q}_f0_m{q}_hard_noedit"]
            candidates = []
            for name, r in measured.items():
                if int(r["nominal_q"]) != q:
                    continue
                mode = name[name.index("hard_"):]
                delta = int(r["correct"])-int(base["correct"])
                dt = float(r["tpf"])-float(base["tpf"])
                datum = dict(model=model, q=q, policy=name, mode=mode, old_cap=int(r["edit"]),
                    correct=int(r["correct"]), tpf=float(r["tpf"]), efficiency=float(r["token_efficiency"]),
                    actual_Q_F=float(r["q_per_f"]), N=int(r["N"]), F=int(r["F"]), Q=int(r["physical_Q"]),
                    delta_correct=delta, delta_TPF=dt, delta_Q=int(r["physical_Q"])-int(base["physical_Q"]))
                rows.append(datum)
                uncertainty.append(dict(model=model, q=q, policy=name, reference=base["policy"],
                    delta_correct=delta, delta_TPF=dt, delta_eff_pp=100 * (
                        float(r["token_efficiency"])-float(base["token_efficiency"])),
                    **paired_uncertainty(request_data[name], request_data[base["policy"]], draws)))
                if mode != "hard_noedit" and "placebo" not in mode and "random" not in mode and delta>0 and dt>=0:
                    candidates.append(datum)
            winner = sorted(candidates, key=lambda r:(-r["correct"],-r["tpf"],r["Q"],r["policy"]))[:1]
            overview.append(dict(model=model, nominal_q=q,
                control_correct=base["correct"], control_tpf=base["tpf"], control_eff=base["token_efficiency"],
                control_Q=base["physical_Q"], both_up_count=len(candidates),
                illustrative_policy=winner[0]["policy"] if winner else "",
                correct=winner[0]["correct"] if winner else "", tpf=winner[0]["tpf"] if winner else "",
                efficiency=winner[0]["efficiency"] if winner else "", physical_Q=winner[0]["Q"] if winner else "",
                note="single B32 saturation; see dual64/ for real two-block Q64" if (model,q)==("llada21",64)
                else "single block saturated" if (model,q) in (("llada21",32),("gemma",256)) else ""))
    table(root/"budget-index.csv", rows)
    table(root/"budget-verdicts.csv", overview)
    table(root/"budget-paired-intervals.csv", uncertainty)


if __name__ == "__main__":
    main()

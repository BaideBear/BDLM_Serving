"""Rebuild the fixed-seat context comparison from scored, token-free tables."""
import argparse
import csv
import json
from pathlib import Path

from tools.budget_index import (paired_uncertainty, request_key, request_metrics,
                                stratified_draws)
from tools.render_hard_study import (ROOT, all_rows, point, read_baselines, render,
                                     window_controls)
from tools.t2t_results import table


def bundle(folder, policy, keys):
    summary, = [s for s in all_rows(folder / "llada21-online-summary.csv") if s["policy"] == policy]
    with (folder / "llada21-online-requests.csv").open() as f:
        requests = [r for r in csv.DictReader(f) if r["policy"] == policy]
    return point(summary, "token_efficiency"), request_metrics(requests, keys, summary)


def report(out, root=ROOT):
    exp = root / "experiments/t2t"
    keys = sorted(request_key(r) for r in json.loads(
        (root / "benchmarks/frozen30/manifest.json").read_text())["samples"])
    draws = stratified_draws(keys)
    base = read_baselines(root / "experiments/baselines/summary.csv")
    groups, contrasts = [], []
    for q, cap in ((4, 1), (8, 1), (16, 2)):
        by = {
            "No edit reserve": bundle(exp / "hard", f"t2t_q{q}_f0_m{q}_hard_noedit", keys),
            "Position only": bundle(exp / "calibrated" if q == 16 else out,
                                    f"t2t_q{q}_f0_m{q-cap}_hard_hazard_fixed", keys),
            "Position + input context": bundle(out, f"t2t_q{q}_f0_m{q-cap}_hard_context_fixed", keys),
        }
        rows = [(k, style, by[k][0]) for k, style in zip(by, ("hard", "adaptive", "geometry"), strict=True)]
        groups.append((q, rows + window_controls("llada21", q, 32, base)))
        p, x = by["Position + input context"]
        for name in ("Position only", "No edit reserve"):
            b, y = by[name]
            contrasts.append(dict(q=q, cap=cap, reference=name,
                correct=p.correct, control_correct=b.correct, tpf=p.tpf, control_tpf=b.tpf,
                efficiency=p.efficiency, control_efficiency=b.efficiency, physical_Q=int(x[:, 3].sum()),
                control_physical_Q=int(y[:, 3].sum()), delta_correct=p.correct-b.correct,
                delta_tpf=p.tpf-b.tpf, delta_eff_pp=100*(p.efficiency-b.efficiency),
                **paired_uncertainty(x, y, draws)))
    table(out / "matched.csv", contrasts)
    render(out / "rollout.svg", "llada21", 32, groups, adaptive=True,
           study_title="Does input-context ranking improve delivery at fixed seats?")
    primary = [r for r in contrasts if r["reference"] == "Position only"]
    return sum(r["delta_correct"] > 0 and r["delta_tpf"] >= 0 for r in primary) >= 2 and all(
        r["delta_correct"] >= 0 for r in primary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "rollout")
    print("Passes matched-position preregistration:", report(p.parse_args().out))

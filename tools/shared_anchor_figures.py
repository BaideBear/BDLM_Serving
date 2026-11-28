"""Publish both models' sealed fixed-offer anchor curves, cohort CSV and a compact result table."""
import argparse
import csv
import json
from pathlib import Path

from experiments.t2t.serving_slots.fixed_request_comparison import MAIN, draw
from experiments.t2t.serving_slots.shared_fixed_shard import identity
from tools.frozen90_six_shards import MANIFEST
from tools.shared_anchor_report import audit, comparisons


def make_rows(report, baseline_csv):
    if report["seen"] != 900 or report["deferred_incomplete_lines"] or len(report["points"]) != 10:
        raise ValueError("Both models need five sealed 90-case offers before releasing this figure")
    comparisons(report, baseline_csv)
    with baseline_csv.open(newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            if row["method"] not in MAIN:
                continue
            for k in ("budget", "n", "correct", "N", "F", "history_Q"):
                row[k] = int(row[k]) if row[k] else None
            row["macro_TPF"] = float(row["macro_TPF"])
            row["token_efficiency"] = float(row["token_efficiency"]) if row["token_efficiency"] else None
            rows.append(row)
    for p in report["points"]:
        for group, cohort in p["cohorts"].items():
            rows.append(dict(model=p["model"], budget=p["offer"], method="Shared planner",
                             cohort="frozen90" if group=="all90" else group, n=cohort["requests"],
                             correct=cohort["correct"], macro_TPF=cohort["macro_TPF"],
                             token_efficiency=cohort["pooled_TE"], N=cohort["N"], F=cohort["F"],
                             history_Q=cohort["Q"], source=report["version"]))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "registration", "baselines", "out"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    reg = json.loads(a.registration.read_bytes())
    report = audit(a.root, json.loads(MANIFEST.read_bytes())["samples"], reg)
    rows = make_rows(report, a.baselines)
    a.out.mkdir(parents=True, exist_ok=True)
    cols = ("model","cohort","budget","method","n","correct","macro_TPF","token_efficiency","N","F","history_Q","source")
    with (a.out/"metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    (a.out/"summary.json").write_text(json.dumps(report, allow_nan=False, indent=2)+"\n")
    draw(rows, shared_output=a.out/"frozen90.png")
    md = ["# Shared fixed-budget planner: formal90 anchor", "",
          f"Formal registration `{identity(reg)}`; execution revision `{reg['commit']}`.",
          "Engineering reference, not a tuned winner. All rows are independent, fixed-offer requests.",
          "Each metric cell is **correct / macro TPF / pooled token efficiency (%)**.",
          "A dash marks an untested budget, `n/a` an unaudited efficiency denominator.",
          "[Figure](frozen90.png) | [All source values](metrics.csv) | [Deltas, physical Q and calibration split](summary.json).",
          "", "| Model | Q | Shared planner | " + " | ".join(MAIN) + " |",
          "|---|---:|"+"---:|"*(len(MAIN)+1)]
    lookup = {(r["model"],r["budget"],r["method"]):r for r in rows if r["cohort"]=="frozen90"}
    for point in report["points"]:
        m, q = point["model"], point["offer"]
        cells = []
        for method in ("Shared planner",*MAIN):
            r = lookup.get((m,q,method))
            if r is None:
                cells.append("-")
                continue
            te = r["token_efficiency"]
            cells.append(f"{r['correct']}/{r['n']} / {r['macro_TPF']:.3f} / " +
                         ("n/a" if te is None else f"{100*te:.2f}%"))
        md.append(f"| {m} | {q} | "+" | ".join(cells)+" |")
    md.extend(["", "At 2B, nominal IBS can saturate at one block. Flow is shown at B and 2B only.",
               "TPF counts actual post-prefill forwards; efficiency pools final nonspecial tokens over selected history-reading decoder rows.", ""])
    (a.out/"README.md").write_text("\n".join(md))


if __name__=="__main__":
    main()

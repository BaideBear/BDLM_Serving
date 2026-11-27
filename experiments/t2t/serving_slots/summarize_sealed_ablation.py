"""Summarize complete fixed-budget ablations alongside the sealed native reference."""
import argparse
import csv
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("gemma", "llada21"))
    parser.add_argument("complete_json", type=Path)
    parser.add_argument("native_csv", type=Path)
    args = parser.parse_args()
    data = json.loads(args.complete_json.read_text())
    offers = (32, 64, 128, 256, 512) if args.model == "gemma" else (4, 8, 16, 32, 64)
    arms = ("00", "10", "11")
    expected = {f"{i:02}-q{q}-{arm}.json.gz" for i in range(90) for q in offers for arm in arms}
    if data["rows"] != len(expected) or set(data["cases"]) != expected:
        raise ValueError("not a complete frozen90 matrix; do not summarize it as final")
    native_policy = "native256" if args.model == "gemma" else "native32"
    with args.native_csv.open() as stream:
        native = [r for r in csv.DictReader(stream) if r["scope"] == "all"
                  and r["arm_policy"] == r["baseline_policy"] == native_policy
                  and r["model"] == args.model]
    out = csv.writer(sys.stdout, lineterminator="\n")
    out.writerow(("model", "cohort", "q", "arm", "correct", "n", "tokens", "forwards",
                  "actual_q", "macro_tpf", "pooled_tpf", "efficiency_pct", "q_per_forward"))
    for n in (90, 30):
        expected_cohort = "all-0-89" if n == 90 else "original-0-29"
        ref, = [r for r in native if r["cohort"] == expected_cohort and int(r["requests"]) == n]
        N, F, Q = (int(ref[k]) for k in ("arm_N", "arm_F", "arm_physical_Q"))
        out.writerow((args.model, f"frozen{n}", offers[-2], native_policy, ref["arm_correct"], n,
                      N, F, Q, f"{float(ref['arm_macro_tpf']):.4f}", f"{N/F:.4f}",
                      f"{100*N/Q:.4f}", f"{Q/F:.4f}"))
        for q in offers:
            for arm in arms:
                rows = [data["cases"][f"{i:02}-q{q}-{arm}.json.gz"] for i in range(n)]
                N, F, Q = (sum(r[k] for r in rows) for k in ("N", "F", "Q"))
                out.writerow((args.model, f"frozen{n}", q, arm, sum(r["correct"] for r in rows),
                              n, N, F, Q, f"{sum(r['TPF'] for r in rows)/n:.4f}",
                              f"{N/F:.4f}", f"{100*N/Q:.4f}", f"{Q/F:.4f}"))


if __name__ == "__main__":
    main()

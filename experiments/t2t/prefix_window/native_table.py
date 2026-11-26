"""Export native-only results once the two frozen90 reference bundles verify."""
import argparse
import csv
import json
from pathlib import Path

from tools.t2t_results import table


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def build(references, scores, out):
    if not (references / "verified.json").is_file():
        raise ValueError("Both native reference bundles must be verified")
    rows = []
    for model in ("llada21", "gemma"):
        captured = json.loads((references / f"capture-{model}.json").read_text())
        if captured["count"] != 60 or captured["status"] != "captured":
            raise ValueError(f"Missing native capture for {model}")
        measured = read(scores / f"native-{model}.csv")
        expected = {(n, scope) for n in (30, 60, 90) for scope in ("all", "gsm8k", "math500", "humaneval")}
        if len(measured) != len(expected) or {(int(r["window"]), r["scope"]) for r in measured} != expected:
            raise ValueError(f"Unexpected native score table for {model}")
        for r in measured:
            if (r["model"] != model or r["baseline_policy"] != r["arm_policy"]
                    or int(r["failed"]) or int(r["rescued"]) or int(r["lost"])
                    or r["baseline_correct"] != r["arm_correct"] or float(r["macro_tpf_delta"])):
                raise ValueError("Expected a complete native self-comparison")
            rows.append(dict(model=model, policy=r["baseline_policy"], cohort=r["cohort"],
                             dataset=r["scope"], requests=r["requests"], correct=r["baseline_correct"],
                             macro_tpf=r["baseline_macro_tpf"], token_efficiency=r["baseline_pooled_token_efficiency"],
                             N=r["baseline_N"], F=r["baseline_F"], Q=r["baseline_physical_Q"]))
    table(out, rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("references", "scores", "out"):
        p.add_argument("--"+name, type=Path, required=True)
    args = p.parse_args()
    build(args.references, args.scores, args.out)

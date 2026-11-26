"""Join private per-arm paired scores and audited exposure; emit only aggregates.

Only completed, scored slack-right/random policies appear. Never fill missing
arms with zero. A prefix share is pooled physical rows, not mean per-request
percent. Pass a private run root for input and an explicit CSV destination.
"""
import argparse
import csv
from pathlib import Path

from .exposure import parse_policy
from .grid import MODELS, ratio_caps


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def rows(data):
    exposure = {}
    for path in sorted(data.glob("exposure-gemma-*.csv")):
        for item in read(path):
            key = (item["policy"], item["cohort"])
            if key in exposure and item != exposure[key]:
                raise ValueError(f"Conflicting exposure for {key}")
            exposure[key] = item
    for q in MODELS["gemma"][1]:
        for k in sorted({1, 2, 4, *ratio_caps(q)}):
            for selector in ("right", "random"):
                path = data / f"gemma-q{q}-slack-k{k}-{selector}.csv"
                if not path.is_file():
                    continue
                for score in read(path):
                    if score["scope"] != "all":
                        continue
                    n = int(score["window"])
                    name = {30: "original30", 60: "added60", 90: "all90"}[n]
                    p = score["arm_policy"]
                    if parse_policy(p) != dict(budget=q, fresh=0, mask=q-k,
                                               mode=f"hard_prefix_slack_{selector}"):
                        raise ValueError(f"Wrong arm in {path}")
                    ex = exposure[p, name]
                    if int(ex["requests"]) != n or int(score["requests"]) != n or \
                            int(ex["mask_displaced"]) or int(ex["ordinary_displaced"]):
                        raise ValueError(f"Invalid slack audit: {path}, {name}")
                    add, events = int(ex["added_rows"]), int(ex["events"])
                    selected = round(events * float(ex["mean_q_rows"]))
                    physical = int(score["arm_physical_Q"])
                    if not (0 <= add <= selected <= physical and events > 0 and selected > 0):
                        raise ValueError("Exposure and generation ledger disagree")
                    yield dict(q=q, k=k, selector=selector, cap_over_q=k/q,
                               full_slack=int(k == q-1), cohort=name, requests=n,
                               zero_correct=score["baseline_correct"], correct=score["arm_correct"],
                               rescued=score["rescued"], lost=score["lost"],
                               quality_delta_ci_low=score["quality_delta_ci_low"],
                               quality_delta_ci_high=score["quality_delta_ci_high"],
                               zero_tpf=score["baseline_macro_tpf"], tpf=score["arm_macro_tpf"],
                               tpf_delta_ci_low=score["macro_tpf_delta_ci_low"],
                               tpf_delta_ci_high=score["macro_tpf_delta_ci_high"],
                               zero_efficiency=score["baseline_pooled_token_efficiency"],
                               efficiency=score["arm_pooled_token_efficiency"], selected_events=events,
                               affected_events=events-int(ex["noop_events"]),
                               affected_requests=n-int(ex["untouched_requests"]),
                               events_reaching_k=ex["saturated_events"], added=add, selected_q=selected,
                               full_generation_q=physical, share_selected=add/selected,
                               share_all_generation=add/physical)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    result = list(rows(args.data))
    if not result:
        raise ValueError("No scored and audited Gemma arms")
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=result[0], lineterminator="\n")
        w.writeheader()
        w.writerows(result)

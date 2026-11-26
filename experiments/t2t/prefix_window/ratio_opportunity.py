"""On frozen Gemma zero trajectories, count physical prefix rows each cap could add.

No simulated accuracy, latency or TPF: the next state is always the recorded zero
state. Only sparse selected forwards admit prefix rows; all generation Q includes
bootstrap/verify. Results are aggregate by cohort and do not contain prompts.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from bdlm_serving.methods.t2t.protocol import verify_work
from . import exposure
from .grid import MODELS, ratio_caps


def tally(hist, k):
    """Histogram maps simultaneous capacity (eligible rows AND empty seats) to events."""
    return (sum(min(k, s) * n for s, n in hist.items()),
            sum(n for s, n in hist.items() if s >= k and k > 0),
            sum(n for s, n in hist.items() if s > k))


def opportunities(data):
    cases_path = data / "samples.json"
    cases = exposure.load_cases(cases_path)
    for q in MODELS["gemma"][1]:
        run = data / ("gemma-q32" if q == 32 else "gemma-grid-k1")
        manifest, kind, width, digest = exposure.load_run(run, cases_path)
        policy = f"t2t_q{q}_f0_m{q}_hard_prefix_zero"
        if (kind, manifest["suite"], manifest["seed_offset"]) != ("gemma", "frozen90", 0) or \
                policy not in manifest["policies"] or json.loads(
                    (run / "qualified.json").read_text()).get("matched") != 90:
            raise ValueError(f"Unqualified zero reference for Q{q}")
        records = exposure.load_requests(run, policy, cases, "gemma")
        hist = {name: Counter() for name, _ in exposure.COHORTS}
        total = {name: Counter() for name, _ in exposure.COHORTS}
        for index, request in records.items():
            verify_work(request)
            physical = request["generation_physical_padded_rows"]
            phases = request["q_by_phase"]
            if (physical != sum(phases.values()) or physical != sum(
                    event["q"] for event in request["events"]) + phases.get("reencode", 0)):
                raise ValueError(f"Physical generation ledger does not close at Q{q}, {index}")
            selected = [exposure.classify(kind, width, policy, index, e)
                        for e in request["events"] if e["stage"] == "selected"]
            capacities = Counter(min(e["slack"], e["pool"]) for e in selected)
            for name, indices in exposure.COHORTS:
                if index in indices:
                    hist[name].update(capacities)
                    total[name].update(all_q=physical, selected_q=sum(e["physical"] for e in selected),
                                       touched_requests=int(any(capacities)))
        for name, indices in exposure.COHORTS:
            h, t = hist[name], total[name]
            for k in sorted({0, 1, 2, 4, *ratio_caps(q)}):
                add, reach, remaining = tally(h, k)
                yield dict(q=q, k=k, full_slack=int(k == q-1), cap_over_q=k/q, cohort=name,
                           requests=len(indices), samples_sha256=digest, selected_events=sum(h.values()),
                           opportunity_events=sum(n for s, n in h.items() if s > 0),
                           opportunity_requests=t["touched_requests"], events_reaching_k=reach,
                           events_with_unused_opportunity=remaining, zero_selected_q=t["selected_q"],
                           zero_generation_q=t["all_q"], hypothetical_added=add,
                           share_selected=add/(t["selected_q"]+add) if t["selected_q"] else 0,
                           share_all_generation=add/(t["all_q"]+add))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    rows = list(opportunities(a.data))
    with a.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

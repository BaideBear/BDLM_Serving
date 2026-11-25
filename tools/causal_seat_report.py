"""Verify first-opportunity one-seat interventions against measured physical events."""
import argparse
import csv
import json
from collections import Counter
from itertools import zip_longest
from pathlib import Path

from tools.t2t_results import table

MODES = ("base", "rank", "random")


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def first_difference(a, b, key):
    """Relative denoise step; -1 = identical series. Includes stage/block alignment."""
    def series(ev):
        return [(x["block"], x["stage"], x[key]) for x in ev if key in x]
    return next((i for i, (x, y) in enumerate(zip_longest(series(a), series(b))) if x != y), -1)


def inspect(raw):
    paths = [r["events"] for r in raw]
    anchors = [[j for j, e in enumerate(ev) if "intervention" in e] for ev in paths]
    idx = raw[0]["index"]
    require(anchors[0] == anchors[1] == anchors[2] and len(anchors[0]) <= 1,
            f"Expected zero or one identical anchor index, task {idx}")
    anchor = anchors[0][0] if anchors[0] else -1
    plans = []
    keys = ("base","targeted","random","victim","candidate","random_candidate","fresh","masks",
            "stratum","clock","pre_sha256","post_rng_sha256")
    for t, path in enumerate(paths):
        require(raw[t]["status"] == "success" and all(raw[t][k] == raw[0][k] for k in
            ("index","dataset","seed","task_id","model")), f"Request identity mismatch on {idx}")
        if anchor < 0:
            require(path == paths[0] and raw[t]["output_ids"] == raw[0]["output_ids"] and all(
                raw[t][k] == raw[0][k] for k in ("generation_forwards", "generation_query_rows")),
                f"Unmatched no-op on {idx}")
            continue
        require(path[:anchor] == paths[0][:anchor], f"Prefixes differ on {idx}")
        e = path[anchor]
        p = e["intervention"]
        plans.append(p)
        require(all(p[k] == plans[0][k] for k in keys), f"Unmatched plan, pre-state or RNG on {idx}")
        require(all(len(p[k]) == 64 for k in ("pre_sha256", "post_rng_sha256", "after_sha256"))
                and p["after_sha256"] == e["trajectory_sha256"], "Missing state evidence")
        expected = p[("base", "targeted", "random")[t]]
        actual = e["positions"] if "positions" in e else [i-32*e["block"] for i in e["absolute_positions"]]
        base, picked = set(p["base"]), set(expected)
        masks, fresh = set(p["masks"]), set(p["fresh"])
        require(e["stage"] == "selected" and sorted(picked) == actual == p["executed"]
                and len(base) == len(expected) == e.get("physical_q", e.get("q"))
                and fresh <= picked and (not base & masks or picked & masks),
                f"Physical Q contract broken on {idx}")
        if t:
            extra = p["candidate" if t == 1 else "random_candidate"]
            require(base-picked == {p["victim"]} and picked-base == {extra}
                    and extra not in fresh|masks and p["victim"] not in fresh
                    and ((p["victim"] in masks) == (p["stratum"] == "mask")), "Expected exactly one swap")
    if plans and plans[0]["candidate"] == plans[0]["random_candidate"]:
        require(paths[1] == paths[2] and raw[1]["output_ids"] == raw[2]["output_ids"],
                "Identical placements must replay exactly")
    return anchor, plans


def main(run, out):
    meta = json.loads((run / "run.json").read_text())
    model, policies = meta["model"], meta["policies"]
    policy = [next(s for s in policies if s.endswith("hard_once_"+m)) for m in MODES]
    marks = [json.loads((run / f"marks-{p}.json").read_text()) for p in policy]
    rows, phases = [], []
    for i in range(30):
        raw = [json.loads((run / f"{i:02d}-{p}.json").read_text()) for p in policy]
        at, plans = inspect(raw)
        for a, b in ((1,0), (2,0), (1,2)):
            x, y = raw[a], raw[b]
            start = max(at, 0)
            all_diff, acc_diff = [first_difference(x["events"][start:], y["events"][start:], k)
                for k in ("trajectory_sha256", "accepted_sha256")]
            pair = f"{MODES[a]}_vs_{MODES[b]}"
            dx = dict(model=model, index=i, dataset=x["dataset"], task_id=x["task_id"], pair=pair,
                anchor=at, stratum=plans[0]["stratum"] if plans else "none",
                delta_admissions=(plans[a]["after_admissions"]-plans[b]["after_admissions"] if plans else 0),
                first_canvas_diff=all_diff, first_accepted_diff=acc_diff,
                output_changed=int(x["output_ids"] != y["output_ids"]),
                correct=int(marks[a][str(i)]), ref_correct=int(marks[b][str(i)]),
                delta_F=x["generation_forwards"]-y["generation_forwards"],
                delta_Q=x["generation_query_rows"]-y["generation_query_rows"],
                delta_N=x["output_tokens"]-y["output_tokens"], delta_TPF=x["tpf"]-y["tpf"])
            rows.append(dx)
            for phase in sorted(x["forwards_by_phase"].keys() | y["forwards_by_phase"].keys()):
                phases.append(dict(model=model, index=i, pair=pair, phase=phase,
                    delta_F=x["forwards_by_phase"].get(phase,0)-y["forwards_by_phase"].get(phase,0),
                    delta_Q=x["q_by_phase"].get(phase,0)-y["q_by_phase"].get(phase,0)))
    out.mkdir(parents=True, exist_ok=True)
    table(out/f"{model}-causal-requests.csv", rows)
    table(out/f"{model}-causal-phases.csv", phases)
    summary = []
    for pair in sorted({x["pair"] for x in rows}):
        for group in ("all", "slack", "mask", "none"):
            rr = [r for r in rows if r["pair"] == pair and (group == "all" or r["stratum"] == group)]
            summary.append(dict(model=model,pair=pair,stratum=group,n=len(rr),
                anchors=sum(r["anchor"]>=0 for r in rr), final_changed=sum(r["output_changed"] for r in rr),
                immediate_canvas=sum(r["first_canvas_diff"]==0 for r in rr),
                immediate_accepted=sum(r["first_accepted_diff"]==0 for r in rr),
                later_accepted=sum(r["first_accepted_diff"]>0 for r in rr),
                wins=sum(r["correct"]>r["ref_correct"] for r in rr), losses=sum(r["correct"]<r["ref_correct"] for r in rr),
                **{k:sum(r[k] for r in rr) for k in ("delta_N","delta_F","delta_Q")}))
    table(out/f"{model}-causal-summary.csv", summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("run","out"):
        p.add_argument("--"+k,required=True,type=Path)
    a = p.parse_args()
    main(a.run,a.out)

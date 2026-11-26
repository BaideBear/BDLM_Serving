"""Audit a completed frozen90 external-slot experiment and score both phases."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from bdlm_serving.evaluation.frozen import sha
from experiments.t2t.prefix_window.score_run import snapshot
from tools.prefix_cohorts import score_files


def audit(row, pattern):
    prev = None
    c = Counter(requests=1, switched_requests=0, up=0, down=0, selections=0,
                mandatory=0, masks=0, backfill=0, prefix=0, charged_other=0,
                overflow_rows=0, underfill_rows=0, offered_rows=0)
    ll = row["model"] == "llada21"
    if ll:
        work = [(e["stage"], e["physical_q"]) for e in row["events"] if e["phase"] != "prefill"]
    else:
        raw = row["physical_events"]
        if raw[0] != ["encoder", row["prefill_rows"]]:
            raise ValueError("Missing observed Gemma prompt prefill")
        if [q for tag,q in raw[1:] if tag == "decoder"] != [e["q"] for e in row["events"]]:
            raise ValueError("Gemma decoder work disagrees with observer")
        work = raw[1:]
    for e in row["events"]:
        if e["stage"] != "selected":
            if "slot_tick" in e or "slot_offer" in e:
                raise ValueError("Consumed a budget on a non-selected event")
            continue
        p = e["prefix_plan"]
        q = pattern[c["selections"] % len(pattern)]
        if (e["slot_tick"] != c["selections"] or e["slot_offer"] != q or p["budget"] != q):
            raise ValueError("Slot clock, phase or offer changed")
        obs = ([pos-p["block_start"] for pos in e["absolute_positions"]] if ll else e["positions"])
        sets = [set(p[k]) for k in ("fresh_rows", "zero_mask_rows", "zero_backfill_rows", "added_rows")]
        actual = len(obs)
        if (p["evicted_rows"] or sum(map(len, sets)) != actual or set.union(*sets) != set(obs)
                or actual != p["actual_width"] or obs != p["selected_rows"]
                or len(sets[0]) != e["mandatory_fresh"] or
                len(sets[1]) != e["selected_masked_before" if ll else "selected_mask"]
                or (ll and (not e["layer_physical_q"] or any(v != actual for v in e["layer_physical_q"])))
                or e["physical_q" if ll else "q"] != actual
                or max(0, len(sets[0])-q) != p["overflow"] or max(0, actual-q) != p["overflow"]):
            raise ValueError("Selected physical rows or protected quota disagree")
        for key, s in zip(("mandatory", "masks", "backfill", "prefix"), sets):
            c[key] += len(s)
        c.update(selections=1, up=int(prev is not None and q > prev),
                 down=int(prev is not None and q < prev), overflow_rows=max(0, actual-q),
                 underfill_rows=max(0, q-actual), offered_rows=q)
        prev = q
    c["switched_requests"] = int(c["up"] + c["down"] > 0)
    selected = sum(c[k] for k in ("mandatory","masks","backfill","prefix"))
    f, total = len(work), sum(v for _, v in work)
    if (selected != row["q_by_phase"].get("selected", 0) or
            f != row["generation_forwards"] or total != row["generation_physical_padded_rows"]
            or total != row["generation_query_rows"] or
            selected != c["offered_rows"] + c["overflow_rows"] - c["underfill_rows"]):
        raise ValueError("Physical work mismatch")
    c.update(charged_other=total-selected, N=row["output_tokens"], F=f, physical_Q=total)
    return c


def report(run, cases, out):
    meta = json.loads((run/"run.json").read_text())
    model = meta["model"]
    a,b = {"gemma": (32,64), "llada21": (8,16)}[model]
    phases = ((a,), (b,), (a,a,b,b), (b,b,a,a))
    labels = ["slottrace_" + "_".join(map(str,t)) for t in phases]
    if meta["policies"] != labels:
        raise ValueError("Unregistered slot experiment")
    out.mkdir(exist_ok=True, parents=True)
    cache = out / ".inputs"
    dirs, summary = [], []
    for label, pattern in zip(labels, phases):
        folder = snapshot(run, label, sha(cases), cache)
        dirs.append(folder)
        counts = []
        for i in range(90):
            raw = json.loads((folder / f"{i:02d}-{label}.json").read_text())
            if (raw["index"], raw["policy"], raw["model"]) != (i, label, model):
                raise ValueError("Request identity mismatch")
            counts.append(audit(raw, pattern))
        for cohort, idx in (("original30", slice(30)), ("added60", slice(30,None)), ("all90", slice(None))):
            sums = Counter()
            for r in counts[idx]:
                sums.update(r)
            summary.append(dict(model=model, policy=label, cohort=cohort, **sums))
    for ctl, arm in ((0,1),(0,2),(1,2),(0,3),(1,3),(2,3)):
        path = out/f"{model}-{labels[ctl]}-vs-{labels[arm]}.csv"
        score_files(dirs[ctl], dirs[arm], cases, path, baseline_policy=labels[ctl], arm_policy=labels[arm])
    with (out/f"{model}-slot-work.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run","cases","out"):
        parser.add_argument("--"+key, type=Path, required=True)
    report(**vars(parser.parse_args()))

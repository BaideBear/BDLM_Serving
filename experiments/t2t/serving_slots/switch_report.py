"""Audit a completed frozen90 external-slot experiment and score both phases."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from bdlm_serving.evaluation.frozen import sha
from bdlm_serving.methods.t2t.slot_trace import parse_offer
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
    width = 32 if ll else 256
    slots = [parse_offer(str(field),width) for field in pattern]
    full = 0
    for e in row["events"]:
        if e["stage"] not in ("selected","native_offer"):
            if "slot_tick" in e or "slot_offer" in e:
                raise ValueError("Consumed a budget on a non-selected event")
            continue
        s = slots[c["selections"] % len(pattern)]
        q,k,mode = s.plan(width)
        if e["slot_tick"] != c["selections"] or e["slot_offer"] != q:
            raise ValueError("Slot clock, phase or offer changed")
        up, down = int(prev is not None and q > prev), int(prev is not None and q < prev)
        prev = q
        if e["stage"] == "native_offer":
            if (ll or not s.native_full or e["q"] != width or e["positions"] != list(range(width))
                    or e["mandatory_fresh"] != e["fresh_selected"] or e["available_fresh"] != e["mandatory_fresh"]
                    or e["overflow"] or "prefix_plan" in e):
                raise ValueError("Explicit native budget did not execute a complete canvas")
            full += width
            c.update(selections=1,up=up,down=down,offered_rows=q,native_offers=1,native_rows=width)
            continue
        p = e["prefix_plan"]
        if (s.native_full or p["quota"] != k or p["budget"] != q
                or e.get("slot_prefix_mode") != (mode if k else None)):
            raise ValueError("Wrong allocation for this slot")
        obs = ([pos-p["block_start"] for pos in e["absolute_positions"]] if ll else e["positions"])
        sets = [set(p[k]) for k in ("fresh_rows", "zero_mask_rows", "zero_backfill_rows", "added_rows")]
        actual = len(obs)
        if (not 0 <= p["prefix_end"] <= width or any(not 0 <= i < width for i in obs)
                or len(sets[3]) > k or not sets[3] <= set(p["eligible_prefix_rows"])
                or any(not 0 <= i < p["prefix_end"] for i in sets[3])
                or p["evicted_rows"] or sum(map(len, sets)) != actual or set.union(*sets) != set(obs)
                or actual != p["actual_width"] or obs != p["selected_rows"]
                or len(sets[0]) != e["mandatory_fresh"] or
                len(sets[1]) != e["selected_masked_before" if ll else "selected_mask"]
                or (ll and (not e["layer_physical_q"] or any(v != actual for v in e["layer_physical_q"])))
                or e["physical_q" if ll else "q"] != actual
                or max(0, len(sets[0])-q) != p["overflow"] or max(0, actual-q) != p["overflow"]):
            raise ValueError("Selected physical rows or protected quota disagree")
        for key, s in zip(("mandatory", "masks", "backfill", "prefix"), sets):
            c[key] += len(s)
        c.update(selections=1, up=up, down=down, overflow_rows=max(0, actual-q),
                 underfill_rows=max(0, q-actual), offered_rows=q)
    c["switched_requests"] = int(c["up"] + c["down"] > 0)
    selected = sum(c[k] for k in ("mandatory","masks","backfill","prefix"))
    f, total = len(work), sum(v for _, v in work)
    if (selected != row["q_by_phase"].get("selected", 0) or
            f != row["generation_forwards"] or total != row["generation_physical_padded_rows"]
            or total != row["generation_query_rows"] or
            full != row["q_by_phase"].get("native_offer",0) or
            selected + full != c["offered_rows"] + c["overflow_rows"] - c["underfill_rows"]):
        raise ValueError("Physical work mismatch")
    c.update(charged_other=total-selected-full, N=row["output_tokens"], F=f, physical_Q=total)
    return c


def report(run, cases, out, reference_switch_run=None):
    meta = json.loads((run/"run.json").read_text())
    model = meta["model"]
    a,b = {"gemma": (32,64), "llada21": (8,16)}[model]
    base = ((a,), (b,), (a,a,b,b), (b,b,a,a))
    def labels_for(phases):
        return ["slottrace_" + "_".join(map(str,t)) for t in phases]
    native = ((128,),("256n",),(128,128,"256n","256n"),("256n","256n",128,128))
    dose = ((32,), (64,), ("32r8",), ("32r8","32r8",64,64), (64,64,"32r8","32r8"),
            (32,)*8+(64,)*8, (64,)*8+(32,)*8)
    options = [base] + ([native,dose] if model == "gemma" else [])
    phases = next((t for t in options if labels_for(t) == meta["policies"]),None)
    if phases is None or (phases == dose) != bool(reference_switch_run):
        raise ValueError("Unregistered slot experiment")
    labels = labels_for(phases)
    sources = [run] * len(labels)
    pairs = [(0,1),(0,2),(1,2),(0,3),(1,3),(2,3)]
    if phases == dose:
        ref = json.loads((reference_switch_run/"run.json").read_text())
        if (ref["model"],ref["policies"]) != (model,labels_for(base)):
            raise ValueError("Expected both original K0 switch phases as references")
        phases += base[2:]
        labels += labels_for(base[2:])
        sources += [reference_switch_run]*2
        pairs = [(0,2),(7,3),(8,4),(7,5),(8,6),(3,4),(5,6)]
        pairs += [(endpoint,mixed) for endpoint in (0,1) for mixed in (3,4,5,6)]
    out.mkdir(exist_ok=True, parents=True)
    cache = out / ".inputs"
    dirs, summary = [], []
    for label, pattern, source in zip(labels, phases, sources, strict=True):
        folder = snapshot(source, label, sha(cases), cache)
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
    for ctl, arm in pairs:
        path = out/f"{model}-{labels[ctl]}-vs-{labels[arm]}.csv"
        score_files(dirs[ctl], dirs[arm], cases, path, baseline_policy=labels[ctl], arm_policy=labels[arm])
    with (out/f"{model}-slot-work.csv").open("w", newline="") as f:
        fields = dict.fromkeys(k for r in summary for k in r)
        w = csv.DictWriter(f, fieldnames=list(fields), lineterminator="\n",restval=0)
        w.writeheader()
        w.writerows(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run","cases","out"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--reference-switch-run",type=Path)
    report(**vars(parser.parse_args()))

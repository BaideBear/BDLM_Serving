"""Audit sealed same-device joint-close pairs without loading torch or model data."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from .gemma_joint_verify_eval import QUAL, check, pair, path
from .gemma_rejected_noise_report import interval


B = 256
COHORTS = (("original30", range(30)), ("new60", range(30, 90)), ("frozen90", range(90)))
FIELDS = ("N", "correct", "F", "decoder_F", "encoder_F", "decoder_Q", "encoder_Q")
REG = ("source", "gpu", "checkpoint", "samples_sha256", "scorer_sha256", "torch", "transformers",
       "layers", "projections")


def tpf(r):
    return r["N"] / r["F"]


def efficiency(r):
    return r["N"] / (r["decoder_Q"] + r["encoder_Q"])


def bill(r):
    hits = r["hits"]
    h = sum(len(e["head_dirty"]) for e in hits)
    t = sum(len(e["tail_dirty"]) for e in hits)
    check(len({e["ordinal"] for e in hits}) == len(hits), "Repeated terminal intervention")
    check(all(e["applied"] == r["on"] for e in hits), "Mislabelled close")
    check(not r["on"] or r["route_Q"].get("verify_joint_paid", 0) == B*len(hits)+t,
          "Measured joint row bill does not reconcile")
    return dict(events=len(hits), exposed=int(bool(hits)), blocks=1+r["encoder_F"],
                with_tail_debt=sum(bool(e["tail_dirty"]) for e in hits),
                head_dirty_Q=h, tail_dirty_Q=t)


def pair_details(a, b):
    row = dict(index=a["index"], dataset=a["dataset"], seed=a["seed"], history_exact=1,
               same_output=a["ids"] == b["ids"])
    x, y = a["hits"], b["hits"]
    row.update(first_tick=x[0]["tick"] if x else None, first_ordinal=x[0]["ordinal"] if x else None,
               first_seal_same=(x[0]["sealed"] == y[0]["sealed"]) if x else None)
    off_seals = {e["ordinal"]:e["sealed"] for e in x}
    on_seals = {e["ordinal"]:e["sealed"] for e in y}
    common = sorted(off_seals.keys() & on_seals.keys())
    row["common_event_ordinals"] = len(common)
    row["event_ordinal_sets_equal"] = off_seals.keys() == on_seals.keys()
    row["first_unequal_common_seal_ordinal"] = next(
        (o for o in common if off_seals[o] != on_seals[o]), None)
    events = []
    for label, r in (("off", a), ("on", b)):
        row.update({label+"_"+k: r[k] for k in (*FIELDS, "max_decoder_Q")})
        row.update({label+"_"+k: v for k, v in bill(r).items()})
        row[label+"_TPF"] = round(tpf(r), 8)
        row[label+"_efficiency"] = round(efficiency(r), 8)
        for n, e in enumerate(r["hits"]):
            hd, td = len(e["head_dirty"]), len(e["tail_dirty"])
            events.append(dict(index=r["index"], arm=label, event=n, ordinal=e["ordinal"], tick=e["tick"],
                               head_dirty=hd, tail_dirty=td, physical_F=1 if r["on"] else 2,
                               physical_Q=B+td+(0 if r["on"] else hd),
                               first_added_head_row=next(
                                   (i for i in range(B) if i not in e["head_dirty"]), None),
                               drain_row_sha=e["drain_rows"], verify_row_sha=e["verify_rows"],
                               joint_row_sha=e["joint_rows"]))
    # Residual after subtracting only the on-trajectory mechanical saving.
    row["F_residual"] = b["F"] - a["F"] + len(y)
    row["decoder_Q_residual"] = b["decoder_Q"] - a["decoder_Q"] + row["on_head_dirty_Q"]
    return row, events


def cohort(name, ids, pairs, rows):
    ps = [pairs[i] for i in ids if i in pairs]
    out = dict(cohort=name, expected=len(ids), matched=len(ps), missing=len(ids)-len(ps))
    if not ps:
        return out
    for j, label in enumerate(("off", "on")):
        arm = [p[j] for p in ps]
        out.update({label+"_"+k: sum(r[k] for r in arm) for k in FIELDS})
        out.update({label+"_"+k: sum(rows[r["index"]][label+"_"+k] for r in arm) for k in bill(arm[0])})
        out[label+"_macro_TPF"] = round(mean(map(tpf, arm)), 6)
        out[label+"_macro_efficiency"] = round(mean(map(efficiency, arm)), 8)
        n, dq, eq = (out[label+"_"+k] for k in ("N", "decoder_Q", "encoder_Q"))
        out[label+"_pooled_decoder_efficiency"] = round(n/dq, 8)
        out[label+"_pooled_physical_efficiency"] = round(n/(dq+eq), 8)
    for k, metric in (("macro_TPF", tpf), ("mean_F", lambda r:r["F"]),
                      ("accuracy", lambda r:int(r["correct"])), ("macro_efficiency", efficiency)):
        out["delta_"+k] = round(mean(metric(b)-metric(a) for a,b in ps), 8)
        out[k+"_lo95"], out[k+"_hi95"] = interval(ps, metric) or (None, None)
    out.update(changed_outputs=sum(a["ids"] != b["ids"] for a,b in ps),
               first_seal_differences=sum(rows[a["index"]]["first_seal_same"] is False for a,b in ps),
               rescues=sum(b["correct"] and not a["correct"] for a,b in ps),
               losses=sum(a["correct"] and not b["correct"] for a,b in ps))
    out.update({k:sum(rows[a["index"]][k] for a,b in ps) for k in ("F_residual", "decoder_Q_residual")})
    return out


def table(p, rows):
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(p)


def gather(root, history, out=None):
    pairs, rows, events, reg = {}, {}, [], None
    for i in range(90):
        if not all(path(root,i,on).exists() for on in (False, True)):
            continue
        a, b = pairs[i] = pair(root, history, i)
        sig = {k:a[k] for k in REG}
        check(reg in (None, sig), "Mixed registration")
        reg = sig
        rows[i], sub = pair_details(a, b)
        events.extend(sub)
    summary = [cohort(n, ids, pairs, rows) for n, ids in COHORTS]
    if out:
        check(len(pairs)==90 and len({a["task_id"] for a,b in pairs.values()})==90,
              "Missing/duplicate frozen case; no compact export")
        for name, ids in COHORTS:
            check(Counter(rows[i]["dataset"] for i in ids)==
                  dict.fromkeys(("gsm8k", "math500", "humaneval"), len(ids)//3), "Unbalanced cohort")
        q = json.loads((root/"qualification.json").read_text())
        check(q["indices"]==list(QUAL) and q["source"]==reg["source"] and
              q["hits"]==sum(rows[i]["on_events"] for i in QUAL)>0, "Unqualified snapshot")
        out.mkdir(parents=True, exist_ok=True)
        table(out/"pairs.csv", list(rows.values()))
        table(out/"events.csv", events)
        table(out/"cohorts.csv", summary)
        tmp = out/"provenance.tmp"
        tmp.write_text(json.dumps(dict(**reg, root=str(root), history=str(history), qualification=list(QUAL),
                                       qualification_hits=q["hits"], matched=len(pairs)),
                                  sort_keys=True, indent=2)+"\n")
        tmp.replace(out/"provenance.json")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "history", "out"):
        ap.add_argument("--"+key, type=Path, required=key!="out")
    args = ap.parse_args()
    print(json.dumps(gather(args.root,args.history,args.out)))

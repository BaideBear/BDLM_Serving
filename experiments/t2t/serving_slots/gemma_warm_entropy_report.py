"""Reconcile the 90 paired A1 receipts on the cluster. Export scalars and digests only."""
import argparse
import csv
import json
from hashlib import sha256
from pathlib import Path
from statistics import mean

from .gemma_joint_verify_eval import check, digest
from .gemma_rejected_noise_report import interval
from .gemma_warm_entropy_eval import A1, BILL


PHASES = ("cold_sample", "warm_sample", "drain", "verify")
PHASE_KEYS = tuple(p + "_" + u for p in PHASES for u in ("F", "Q"))
AUDIT_KEYS = ("N", "F", "decoder_F", "decoder_Q", "encoder_F", "encoder_Q")
METRICS = (*AUDIT_KEYS, "total_Q", *PHASE_KEYS, "warm_admitted", "warm_revoked")


def bill(x):
    phases = x["phases"]
    check(not set(phases).difference((*PHASE_KEYS, "warm_admitted", "warm_revoked")),
          "Unknown phase")
    for unit in ("F", "Q"):
        check(sum(phases.get(f"{p}_{unit}", 0) for p in PHASES)
              == x["decoder_"+unit] == sum(x["route_"+unit].values()), "Unreconciled decoder")
    check(x["F"] == x["decoder_F"] + x["encoder_F"] and x["encoder_Q"] == 256*x["encoder_F"]
          and x["max_decoder_Q"] <= 128 and not x["joint_F"], "Invalid physical bill")
    check(phases["verify_Q"] == 256*(1+x["encoder_F"]) and phases["verify_F"]*128 == phases["verify_Q"],
          "Incomplete terminal sweeps")
    out = {k: x[k] for k in AUDIT_KEYS}
    out.update(total_Q=x["decoder_Q"]+x["encoder_Q"], correct=int(x["correct"]),
               ids_sha256=digest(x["output_ids"]))
    out.update({k: phases.get(k, 0) for k in (*PHASE_KEYS, "warm_admitted", "warm_revoked")})
    return out


def compare(r, h, task):
    check(r["schema"] == "warm-entropy-pair-v1" and r["q"] == h["budget"] == 128
          and r["gpu"] == A1 and h["fuse"] is True and h["slack"] is False and h["gate"] == "off"
          and h["prefix"] == [1, 4]
          and all(r[k] == h[k] == task[k] for k in ("index", "task_id", "dataset", "seed"))
          and all(r[k] == h[k] for k in ("input_sha256", "samples_sha256", "scorer_sha256")),
          "Mismatched history or manifest")
    off, on = r["off"], r["on"]
    check(all(off[k] == h[k] for k in (*BILL, "output_ids", "N", "correct")), "Replay failed")
    for x in off, on:
        check(type(x["correct"]) is bool and 0 < x["N"] <= len(x["output_ids"]) <= 2048, "Bad delivery")
    first = r["first"]
    if first:
        extra = first["extra"]
        check(0 <= first["tick"] < min(off["decoder_F"],on["decoder_F"])
              and 0 <= first["ordinal"] <= min(off["encoder_F"],on["encoder_F"])
              and 0 <= first["age"] < 48 and 0 < len(extra) <= first["measured"] <= 128
              and sorted(set(extra)) == extra and 0 <= min(extra) <= max(extra) < 256
              and len(bytes.fromhex(first["predecessor_sha256"])) == 32, "Bad first-difference certificate")
    else:
        check(off == on, "Unexplained divergence")
    row = {k: r[k] for k in ("index", "dataset", "task_id", "seed")}
    row.update(q=128, first_tick=first["tick"] if first else None,
               first_block=first["ordinal"] if first else None, first_age=first["age"] if first else None,
               first_measured_Q=first["measured"] if first else None, first_extra=len(first["extra"]) if first else 0,
               predecessor_sha256=first["predecessor_sha256"] if first else None,
               matched_ids=int(off["output_ids"] == on["output_ids"]))
    for arm, x in (("off", off), ("on", on)):
        row.update({arm+"_"+k: v for k,v in bill(x).items()})
    return row


def summarize(rows):
    n = len(rows)
    out = dict(pairs=n, exposed=sum(x["first_tick"] is not None for x in rows),
               matched_ids=sum(x["matched_ids"] for x in rows),
               rescues=sum(x["on_correct"] > x["off_correct"] for x in rows),
               losses=sum(x["off_correct"] > x["on_correct"] for x in rows),
               lower_F=sum(x["on_F"] < x["off_F"] for x in rows),
               same_F=sum(x["on_F"] == x["off_F"] for x in rows))
    for arm in ("off", "on"):
        for k in (*METRICS, "correct"):
            key = f"{arm}_{k}"
            out[key] = sum(x[key] for x in rows)
        out[arm+"_macro_TPF"] = round(mean(x[arm+"_N"]/x[arm+"_F"] for x in rows), 6)
        out[arm+"_pooled_TPF"] = round(out[arm+"_N"]/out[arm+"_F"], 6)
    pairs = [tuple(dict(dataset=x["dataset"], tpf=x[a+"_N"]/x[a+"_F"], correct=x[a+"_correct"])
                   for a in ("off", "on")) for x in rows]
    for metric in ("tpf", "correct"):
        lo, hi = interval(pairs, lambda x: x[metric])
        out.update({f"delta_{metric}_lower95": lo, f"delta_{metric}_upper95": hi})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("root","history","manifest","ibs-matrix","out"):
        ap.add_argument("--"+k, type=Path, required=True)
    a = ap.parse_args()
    manifest = json.loads(a.manifest.read_bytes())
    by_id = {x["index"]: x for x in manifest["samples"]}
    check(set(by_id) == set(range(90)) and len(manifest["samples"]) == 90, "Not frozen90")
    with a.ibs_matrix.open(newline="") as f:
        ibs = {x["cohort"]:x for x in csv.DictReader(f)
               if (x["model"],x["method"],x["setting"]) == ("gemma","ibs","128")}
    rows, cohorts, files = [], [], []
    signature = None
    for i in range(90):
        p = a.root/f"C128-pair-{i:02}.json"
        blob = p.read_bytes()
        r = json.loads(blob)
        hb = (a.history/f"C128-profile-fuse1-{i:02}.json").read_bytes()
        sig = (r["source"],r["scorer_sha256"],r["samples_sha256"])
        check(r["ref_sha256"] == sha256(hb).hexdigest() and r["samples_sha256"] == manifest["samples_sha256"]
              and (signature is None or sig == signature), "Mixed references or registration")
        signature = sig
        files.append([p.name, sha256(blob).hexdigest()])
        rows.append(compare(r, json.loads(hb), by_id[i]))
    for k, alt, subset in (("original30","original30",rows[:30]),("new60","added60",rows[30:]),
                           ("frozen90","all90",rows)):
        baseline = ibs[alt]
        check(int(baseline["requests"]) == len(subset), "Bad reference denominator")
        cohorts.append(dict(cohort=k, **summarize(subset),
                            **{"ibs_"+key:baseline[key] for key in ("correct","F","N","physical_Q","macro_tpf")}))
    a.out.mkdir(parents=True, exist_ok=True)
    for name, data in (("cohorts.csv", cohorts), ("pairs.csv", rows)):
        with (a.out/name).open("w", newline="") as f:
            writer = csv.DictWriter(f, data[0], lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    prov = dict(schema="warm-entropy-compact-v1", q=128, pairs=len(rows), source=signature[0],
                scorer_sha256=signature[1], samples_sha256=signature[2], raw=str(a.root), history=str(a.history),
                gpu=A1, pair_files_sha256=digest(sorted(files)),
                ibs_matrix_sha256=sha256(a.ibs_matrix.read_bytes()).hexdigest())
    (a.out/"provenance.json").write_text(json.dumps(prov, sort_keys=True, indent=2)+"\n")


if __name__ == "__main__":
    main()

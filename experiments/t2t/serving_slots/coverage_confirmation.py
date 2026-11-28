"""Summarize complete GPU pairs; interim comparisons use identical IDs at every offer."""
import argparse
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
from statistics import mean

from .coverage_span_report import ANCHORS, KEYS, TIERS, audit, check, decide
from .llada_seat_report import paired
from .risk_ribbon_report import bill


def split_tpf(a, b):
    # Symmetric exact decomposition of b.N/b.F - a.N/a.F, not causal attribution.
    length = (b["N"]-a["N"]) * (1/b["F"]+1/a["F"])/2
    calls = (b["N"]+a["N"]) * (1/b["F"]-1/a["F"])/2
    return length, calls


def summarize(rows, arm, *, intervals=False):
    a, b = bill(rows, "live"), bill(rows, arm)
    pieces = [split_tpf(r["live"], r[arm]) for r in rows]
    length, calls = map(mean, zip(*pieces))
    change = b["macro_TPF"]-a["macro_TPF"]
    check(isclose(length+calls, change, abs_tol=1e-10), "TPF accounting failed")
    out = dict(n=len(rows), live=a, candidate=b, delta_macro_TPF=change,
               output_length_term=length, forward_count_term=calls,
               per_arm_two_block_F={k:sum(r[k]["two_block_F"] for r in rows) for k in ("live", arm)},
               rescues=sum(r[arm]["correct"] and not r["live"]["correct"] for r in rows),
               losses=sum(r["live"]["correct"] and not r[arm]["correct"] for r in rows))
    if intervals:
        out["paired_stratified_ci95"] = paired(rows, "live", arm)
    return out


def report(root, manifest, *, interim=False, refs=None):
    pilot = audit(root, manifest)
    arm = decide(pilot)["expand"]
    check(arm is not None, "no candidate qualified")
    m = json.loads(manifest.read_bytes())["samples"]
    order = [i+j for i in range(30) for j in (0,30,60) if i+j not in ANCHORS]
    inputs, records, counts, digests, completed, reg_checks = {}, {}, {}, {}, {}, set()
    for q in TIERS:
        p = root/f"llada-coverage-span-v1-q{q}"
        d = root/f"llada-coverage-span-v1-confirm-{arm}-q{q}"
        reg = json.loads((d/"run.json").read_bytes())
        check(all(reg[k]==v for k,v in pilot[q]["registration"].items()) and
              reg["extension"] is True and reg["winner"]==arm and reg["confirm_indices"]==order and
              reg["anchors_excluded"]==list(ANCHORS) and reg["pilot_receipt_sha256"]==pilot[q]["receipt_sha256"],
              "changed confirmation registration")
        reg_checks.add((reg["code_sha256"],reg["checkpoint"]))
        files = sorted(d.glob("[0-9][0-9].json"))
        ids = {int(f.stem) for f in files}
        check(ids==set(order[:len(ids)]), "non-prefix confirmation IDs")
        marker = d/"complete.json"
        completed[q] = marker.is_file()
        if completed[q]:
            check(ids==set(order) and json.loads(marker.read_bytes())==dict(q=q,indices=order,winner=arm),
                  "invalid completion marker")
        elif not interim:
            raise ValueError(f"Q{q} not sealed; use --interim for a registered snapshot")
        records[q], counts[q], digest = {}, len(ids), sha256()
        for f in [*(p/f"{i:02}.json" for i in ANCHORS), *files]:
            raw = f.read_bytes()
            r = json.loads(raw)
            i = r["index"]
            check(f.name==f"{i:02}.json" and i not in records[q] and
                  all(r[k]==m[i][k] for k in KEYS) and
                  inputs.setdefault(i,r["input"])==r["input"] and len(r["input"])==64,
                  "prompt, seed or identity mismatch")
            if refs is not None:
                ref = json.loads((refs/f"{i:02d}-native32.json").read_bytes())
                check(ref["index"]==i and sha256(json.dumps(ref["prompt_ids"],separators=(",",":")).encode()).hexdigest()==r["input"],
                      "prompt differs from original native input")
            for k in ("live",arm):
                v = r[k]
                check(type(v["correct"]) is bool and len(v["output_digest"])==64 and
                      0<v["max_decoder_Q"]<=q and 0<v["F"]<=v["Q"]<=v["F"]*v["max_decoder_Q"] and
                      0<=v["two_block_F"]<=v["F"] and 0<=v["N"]<=2048,"invalid final bill")
            digest.update(sha256(raw).digest())
            records[q][i]=r
        digests[q]=digest.hexdigest()
    check(len(reg_checks)==1,"conflicting checkpoint or confirmation runner across tiers")
    common=set.intersection(*(set(r) for r in records.values()))-set(ANCHORS)
    sealed=all(completed.values())
    groups={"confirm81" if sealed else "interim_common":sorted(common)}
    if sealed:
        groups.update(original30=list(range(30)),new60=list(range(30,90)),frozen90=list(range(90)))
    return dict(candidate=arm, complete=sealed, observed_new_pair_counts=counts, cohort_indices=groups,
                receipt_digests=digests, results={
                    q:{key:summarize([recs[i] for i in g],arm,intervals=sealed)
                       for key,g in groups.items() if g} for q,recs in records.items()})


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root",type=Path)
    ap.add_argument("manifest",type=Path)
    ap.add_argument("--reference",type=Path)
    ap.add_argument("--interim",action="store_true")
    a=ap.parse_args()
    print(json.dumps(report(a.root,a.manifest,interim=a.interim,refs=a.reference),indent=2))

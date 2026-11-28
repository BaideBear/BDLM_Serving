"""Audit compact GPU receipts on the host; do not export generated text."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

from .llada_risk_ribbon import ANCHORS, KEYS, RISK, SAMPLES, SCORER_SHA, TIERS


def bill(rows, arm):
    vals=[r[arm] for r in rows]
    sums={k:sum(r[k] for r in vals) for k in ("N","F","Q","correct")}
    return dict(**sums, failures=sum(bool(r["failure"]) for r in vals),
                macro_TPF=mean(r["N"]/r["F"] for r in vals),TE=sums["N"]/sums["Q"])


def audit(root,manifest,tiers=TIERS,*,full=False):
    m=json.loads(manifest.read_bytes())
    assert m["samples_sha256"]==SAMPLES
    result={}
    for q in tiers:
        def read(p):
            return json.loads(p.read_bytes())
        d=root/f"llada-risk-ribbon-v1-q{q}"
        reg=read(d/"run.json")
        assert reg["q"]==q and reg["protocol"]=="llada-risk-ribbon-v1"
        assert reg["risk"]==RISK and reg["samples"]==SAMPLES and reg["scorer"]==SCORER_SHA
        assert tuple(reg["indices"])==ANCHORS and read(d/"complete.json")==dict(indices=list(ANCHORS),q=q)
        folders=[d]
        if full:
            extra=root/f"llada-risk-ribbon-v1-extension-q{q}"
            expected=set(range(90))-set(ANCHORS)
            more=read(extra/"run.json")
            assert more["device"]==reg["device"] and more["samples"]==reg["samples"] and more["risk"]==RISK
            assert set(read(extra/"complete.json")["indices"])==expected
            folders.append(extra)
        docs=[p for folder in folders for p in sorted(folder.glob("[0-9][0-9].json"))]
        ids=set(); rows=[]; h=sha256()
        for p in docs:
            r=read(p); i=r["index"]; assert p.name==f"{i:02}.json" and i not in ids
            assert all(r[k]==m["samples"][i][k] for k in KEYS)
            assert r["first"] or r["live"]==r["ribbon"]
            assert all(0<a["max_decoder_Q"]<=q and 0<a["F"]<=a["Q"]<=a["F"]*q and a["N"]>=0
                       for a in (r["live"],r["ribbon"]))
            h.update(sha256(p.read_bytes()).digest()); ids.add(i); rows.append(r)
        assert ids==(set(range(90)) if full else set(ANCHORS))
        groups={}
        for label,part in (("original",[r for r in rows if r["index"]<30]),("additional",[r for r in rows if r["index"]>=30]),("all",rows)):
            groups[label]=dict(n=len(part),exposed=sum(bool(r["first"]) for r in part),
                rescues=sum(not r["live"]["correct"] and r["ribbon"]["correct"] for r in part),
                losses=sum(r["live"]["correct"] and not r["ribbon"]["correct"] for r in part),
                mean_paired_TPF_delta=mean(r["ribbon"]["N"]/r["ribbon"]["F"]-r["live"]["N"]/r["live"]["F"] for r in part),
                live=bill(part,"live"),ribbon=bill(part,"ribbon"))
        result[q]=dict(registration=reg, receipt_sha256=h.hexdigest(), cohorts=groups)
    return result


def gate(data):
    if set(data)!=set(TIERS):
        raise ValueError("Do not select an arm from partial tier coverage")
    cohorts=[v["cohorts"]["all"] for v in data.values()]
    if any(g["n"]!=len(ANCHORS) for g in cohorts):
        raise ValueError("Expected registered anchors, not validation data")
    return (any(g["exposed"] for g in cohorts)
            and all(g["ribbon"]["correct"]>=g["live"]["correct"]-1 and
                    not g["ribbon"]["failures"] for g in cohorts)
            and sum(g["mean_paired_TPF_delta"]>0 for g in cohorts)>=3)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root",type=Path)
    ap.add_argument("manifest",type=Path)
    ap.add_argument("--full", action="store_true")
    args=ap.parse_args()
    data=audit(args.root,args.manifest,full=args.full)
    print(json.dumps(dict(tiers=data,expand=gate(data) if not args.full else None),indent=2))

"""Score strict fixed GPU references once; pair only complete same-budget GPU cohorts."""
import argparse
from collections import defaultdict
import csv
import json
from math import isclose
from pathlib import Path
from statistics import fmean
import sys

from bdlm_serving.evaluation import scoring
from bdlm_serving.evaluation.frozen import sha
from . import fixed_family_references as refs
from .fixed_family_report import COHORTS, DATA, MODELS, report
from .llada_all_slots90 import require
from .mask_pair_report import efficiency_interval
from tools.prefix_cohorts import load_cases, normalize_request, paired_ci, quality_ci


SUMMARY=refs.HERE/"controls/fixed-family-references.csv"
CACHE=DATA/"fixed-family-reference-case-ledger-v1.json"
LEDGER_SHA="ff9caccbf776416f0237ddcc73af774fbe567c57929b359583f70fba128d64ce"
SCORER="a72cec90206f69616b9de436d052f6df54cf075ba9abeac4a75a11e46587d692"
ID=("index","dataset","task_id","seed")


def from_reference(raw,case):
    r=normalize_request(raw,case,"strict fixed-budget reference")
    require(r["Q"]==r["physical_Q"],"Reference logical and physical costs differ")
    return dict(**{k:case[k] for k in ID},correct=bool(r["correct"]),N=r["N"],F=r["F"],Q=r["physical_Q"])


def validate(ledger, tasks):
    require(ledger["samples"]==sha(DATA/"samples.json") and
            ledger["baseline_table"]==sha(SUMMARY) and ledger["scorer"]==sha(Path(scoring.__file__))==SCORER,
            "Different frozen data, scorer, or baseline revision")
    with SUMMARY.open(newline="") as f:
        archived=list(csv.DictReader(f))
    require(len(archived)==60 and len(ledger["arms"])==20,"Incomplete references")
    for s in archived:
        arm=ledger["arms"][s["model"]+":"+s["raw_policy"]]
        idx=dict(refs.COHORTS)[s["cohort"]]
        require(arm["q"]==int(s["budget"]) and arm["method"]==s["method"] and len(arm["items"])==90,
                "Wrong assignment")
        subset=arm["items"][idx]
        for a,b in zip(subset,tasks[idx],strict=True):
            require(all(a[k]==b[k] for k in ID) and type(a["correct"]) is bool
                    and a["N"]>=0 and a["F"]>0 and a["Q"]>=a["F"],"Invalid item")
        require(len(subset)==int(s["requests"]) and sum(x["correct"] for x in subset)==int(s["correct"])
                and all(sum(x[k] for x in subset)==int(s[k]) for k in ("N","F","Q"))
                and isclose(fmean(x["N"]/x["F"] for x in subset),float(s["macro_TPF"]),rel_tol=1e-12),
                f"Per-item rescore differs from published aggregate: {s['model']} {s['method']} {s['budget']} {s['cohort']}")


def prepare():
    require(not CACHE.exists(),"Reference ledger already exists")
    path=DATA/"samples.json"
    tasks=load_cases(path)
    arms={}
    for model,q,method,policy,group,_,_ in refs.sources():
        rows=[]
        for case in tasks:
            raw=json.loads((group/f"{case['index']:02}-{policy}.json").read_bytes())
            require(raw["policy"]==policy,"Wrong reference policy")
            rows.append(from_reference(raw,case))
        arms[model+":"+policy]=dict(q=q,method=method,items=rows)
        print(f"Rescored {model} {policy} on frozen90",file=sys.stderr,flush=True)
    ledger=dict(samples=sha(path),baseline_table=sha(SUMMARY),scorer=sha(Path(scoring.__file__)),arms=arms)
    validate(ledger,tasks)
    tmp=CACHE.with_suffix(".tmp")
    with tmp.open("x") as f:
        json.dump(ledger,f,allow_nan=False,separators=(",",":"))
    # Do not overwrite another scorer's sealed ledger.
    CACHE.hardlink_to(tmp)
    tmp.unlink()


def pair(control,variant):
    require(len(control)==len(variant)>0,"Empty or unpaired sample")
    cts=defaultdict(lambda:[0,0,0,0])
    dt=defaultdict(list)
    for a,b in zip(control,variant,strict=True):
        require(all(a[k]==b[k] for k in ID),"Paired identity differs")
        cts[a["dataset"]][(not a["correct"])*2+(not b["correct"])]+=1
        dt[a["dataset"]].append(b["N"]/b["F"]-a["N"]/a["F"])
    dq,_,ql,qh,_=quality_ci(cts)
    dt,_,tl,th=paired_ci(dt)
    measures={}
    tables=[]
    for side,rows in (("baseline",control),("candidate",variant)):
        n,f,q=(sum(x[k] for x in rows) for k in ("N","F","Q"))
        measures.update({side+"_"+k:v for k,v in dict(correct=sum(x["correct"] for x in rows),
            macro_TPF=fmean(x["N"]/x["F"] for x in rows),efficiency=n/q,N=n,F=f,Q=q).items()})
        tables.append({x["index"]:dict(N=x["N"],physical_Q=x["Q"]) for x in rows})
    el,eh=efficiency_interval(control,*tables)
    return dict(requests=len(control),**measures,rescued=sum(x[2] for x in cts.values()),
        lost=sum(x[1] for x in cts.values()),delta_quality=dq,quality_ci_low=ql,quality_ci_high=qh,
        delta_TPF=dt,TPF_ci_low=tl,TPF_ci_high=th,
        delta_efficiency=measures["candidate_efficiency"]-measures["baseline_efficiency"],
        efficiency_ci_low=el,efficiency_ci_high=eh)


def paired_rows(ledger):
    validate(ledger,load_cases(DATA/"samples.json"))
    for model in MODELS:
        data=report(DATA,model,"cuda",include_items=True)
        for name,arm in ledger["arms"].items():
            if name.split(":")[0]!=model:
                continue
            q=arm["q"]
            for cohort,idx in COHORTS.items():
                if q not in data["cohorts"][cohort]:
                    continue
                a=[arm["items"][i] for i in sorted(idx)]
                b=[data["items"][q][i] for i in sorted(idx)]
                yield dict(model=model,device="cuda",budget=q,cohort=cohort,
                           baseline=arm["method"],law="1/8",**pair(a,b))


def smoke_rows(ledger,model):
    """A diagnostic on three predeclared items, without inferential intervals."""
    validate(ledger,load_cases(DATA/"samples.json"))
    data=report(DATA,model,"cuda",include_items=True)
    idx=(0,10,20)
    require(set(data["protocol_smoke3"])==set(MODELS[model][1]),"Three complete cases at all tiers required")
    for arm_name,arm in ledger["arms"].items():
        if arm_name.split(":")[0]!=model:
            continue
        q=arm["q"]
        a=[arm["items"][i] for i in idx]
        b=[data["items"][q][i] for i in idx]
        require(all(x[k]==y[k] for x,y in zip(a,b,strict=True) for k in ID),"Unpaired smoke item")
        row=dict(model=model,device="cuda",q=q,baseline=arm["method"],indices="0,10,20",law="1/8")
        for label,items in (("baseline",a),("candidate",b)):
            n,f,paid=(sum(x[k] for x in items) for k in ("N","F","Q"))
            row.update({label+"_"+k:v for k,v in dict(correct=sum(x["correct"] for x in items),
                       macro_TPF=fmean(x["N"]/x["F"] for x in items),efficiency=n/paid,N=n,F=f,paid_Q=paid).items()})
        yield row


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    op=ap.add_mutually_exclusive_group()
    op.add_argument("--prepare",action="store_true",help="Create the cluster-only per-item reference ledger")
    op.add_argument("--smoke3",choices=MODELS,help="Pair all fixed CUDA tiers on the predeclared initial three cases only")
    args=ap.parse_args()
    if args.prepare:
        prepare()
    else:
        require(sha(CACHE)==LEDGER_SHA,"Changed case ledger; refuse to pair")
        ledger=json.loads(CACHE.read_bytes())
        rows=list(smoke_rows(ledger,args.smoke3) if args.smoke3 else paired_rows(ledger))
        if rows:
            writer=csv.DictWriter(sys.stdout,fieldnames=list(rows[0]),lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        else:
            print("No complete fixed-family GPU cohort; no paired results.",file=sys.stderr)

"""Exact per-request TPF accounting for a sealed LLaDA window and capped IBS."""
import argparse
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
from statistics import mean


def require(ok,why):
    if not ok: raise ValueError(why)


def terms(control,trial,q):
    factors=[]
    for r in (control,trial):
        n,f,w=(r[k] for k in ("N","F","Q"))
        require(n>=0 and 0<f<=w<=q*f,"invalid physical bill")
        factors.append((w/f,n/w))
    (w0,e0),(w1,e1)=factors
    width=(w1-w0)*(e1+e0)/2
    yield_per_q=(e1-e0)*(w1+w0)/2
    delta=trial["N"]/trial["F"]-control["N"]/control["F"]
    require(isclose(delta,width+yield_per_q,abs_tol=1e-12),"per-request product did not reconcile")
    return dict(baseline_width=w0,window_width=w1,baseline_n_per_q=e0,window_n_per_q=e1,
                width_term=width,n_per_q_term=yield_per_q,delta_tpf=delta)


def report(root,ibs_file,sealed):
    ref=ibs_file.read_bytes()
    require(sha256(ref).hexdigest()=="7c7d94d0a090c234b18322c339361b3b0d0d76a5ba83f03083d8659a6805ffde","wrong IBS export")
    refs=json.loads(ref)
    meta=json.loads((root/"run.json").read_bytes())
    q=meta["q"]
    require(meta==sealed["registration"] and meta["protocol"]=="llada-fixed-width-window-v1"
            and q in (4,8,16) and sorted(meta["order"])==list(range(90)) and
            meta["samples"]==refs["samples"] and meta["scorer"]==refs["scorer"],"wrong control/study")
    rows=[];dig=sha256()
    for i,reference in enumerate(refs["arms"][str(q)]):
        blob=(root/"rows"/f"{i:02d}.json").read_bytes()
        dig.update(sha256(blob).digest())
        r=json.loads(blob)
        require(r["index"]==i and all(r[k]==reference[k] for k in ("task_id","seed","dataset","index","input")),
                "not the same request")
        rows.append((reference,r["window"],terms(reference,r["window"],q)))
    require(len(rows)==90 and dig.hexdigest()==sealed["rows_sha256"],"not the sealed ninety requests")
    cohorts={}
    for name,group in (("original30",rows[:30]),("new60",rows[30:]),("frozen90",rows)):
        c=cohorts[name]=dict(n=len(group))
        v=sealed["cohorts"][name]
        for j,key in enumerate(("ibs","window")):
            stats={t:sum(r[j][t] for r in group) for t in ("N","F","Q","correct")}
            if j:
                require(all(v["arms"][key][t]==s for t,s in stats.items()),"wrong sealed trajectory")
            c[key]=dict(**stats,pooled_width=stats["Q"]/stats["F"],TE=stats["N"]/stats["Q"],
                        macro_TPF=mean(r[j]["N"]/r[j]["F"] for r in group))
        c["macro"]=m={k:mean(r[-1][k] for r in group) for k in group[0][-1]}
        c["narrower_requests"]=sum(r[-1]["window_width"]<r[-1]["baseline_width"] for r in group)
        require(isclose(m["delta_tpf"],v["window_minus_external_capped_IBS"]["delta_macro_TPF"],abs_tol=1e-12)
                and isclose(m["delta_tpf"],m["width_term"]+m["n_per_q_term"],abs_tol=1e-12),
                "cohort decomposition did not reconcile")
    return dict(protocol="paired-tpf-product-v1",q=q,rows_sha256=dig.hexdigest(),
                ibs_ledger=refs["ledger"],cohorts=cohorts)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("root","ibs","sealed"): ap.add_argument(name,type=Path)
    a=ap.parse_args()
    print(json.dumps(report(a.root,a.ibs,json.loads(a.sealed.read_bytes())),indent=2,sort_keys=True))

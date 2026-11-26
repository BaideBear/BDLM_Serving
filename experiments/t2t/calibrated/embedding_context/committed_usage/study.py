"""Fixed committed-prefix usage ablation, calibrated on separate 24 native requests."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from bdlm_serving.evaluation.frozen import MANIFEST, ROOT, sha
from bdlm_serving.methods.t2t.committed_usage import CommittedUsage
from bdlm_serving.methods.t2t.embedding_context import CALIBRATION, Projector, context, input_table, read_context_weights
from tools.edit_opportunity import load, evaluate, fit, shortlist
from tools.lexical_hazard import BUDGETS, check_split, write_rows

RAW=ROOT.parent/"t2t-native-1125"
HERE=Path(__file__).resolve().parent


def run(out, checkpoint):
    if out.resolve().is_relative_to(ROOT):
        raise ValueError("Choose private output")
    cal, native=RAW/"calibration24-v1-llada21",RAW/"llada21"
    if (sha(cal/"weights.json")!=sha(CALIBRATION) or
            json.loads((native/"verified.json").read_text())!=dict(model="llada21",matched_requests=30)):
        raise ValueError("Unverified calibration or native evaluation")
    train=json.loads(CALIBRATION.read_text())["samples"]
    test=json.loads(MANIFEST.read_text())["samples"]
    check_split(train,test)
    timings={"commit":[], "features":[]}

    class Timed(CommittedUsage):
        def commit(self, ids):
            t=perf_counter_ns()
            super().commit(ids)
            timings["commit"].append(perf_counter_ns()-t)
        def features(self, old, before, mask):
            t=perf_counter_ns()
            rows=super().features(old,before,mask)
            timings["features"].append(perf_counter_ns()-t)
            return rows

    out.mkdir(parents=True,exist_ok=True)
    original=read_context_weights(RAW/"context-fixed-private/calibration24.json")
    rows=[]
    with input_table(checkpoint) as table:
        projector=Projector(table)
        def path(folder,t,factory):
            return load(folder/f"{t['index']:02d}.trace.jsonl.gz",t,context_features=lambda *args:context(
                projector.unit,*args),prefix_factory=factory)
        trained=fit([path(cal,t,Timed) for t in train])
        if trained.shape!=(34,) or not np.isfinite(trained).all():
            raise ValueError("Invalid fit")
        for t in test:
            steps=path(native,t,CommittedUsage)
            for method,w in (("context",original),("committed_usage",trained)):
                subset=[(*step[:-2],step[-2][:,:len(w)] if step[3] else step[-2],step[-1]) for step in steps]
                curve=shortlist(subset,w)
                metrics=[(f"top-{k}",curve[k]["hits"],curve[k]["seats"],curve[k]["positives"]) for k in (1,4,8,16)]
                for q,e in BUDGETS:
                    c=evaluate(subset,w,q,e,"hazard")
                    metrics.append((f"Q{q}/{e}",c["sparse_reserved_hits"],c["sparse_seats"],c["selected_old"]))
                rows.extend(dict(index=t["index"],dataset=t["dataset"],method=method,
                                 metric=name,hits=h,seats=s,targets=n) for name,h,s,n in metrics)
    by={(r["method"],r["metric"],r["index"]):r for r in rows}
    published={r["metric"]:r for r in csv.DictReader((HERE.parent/"comparison.csv").read_text().splitlines())}
    summary=[]
    for metric in sorted({r["metric"] for r in rows}):
        pairs=[(by["context",metric,i],by["committed_usage",metric,i]) for i in range(30)]
        if any((a["targets"],a["seats"])!=(b["targets"],b["seats"]) for a,b in pairs):
            raise ValueError("Unequal same-state seats or target sets")
        v=Counter({k:sum(a[k] for a,_ in pairs) for k in ("hits","seats","targets")})
        if v["targets"]!=191 or (metric in published and any(int(published[metric][k])!=v[j]
                for k,j in (("context_hits","hits"),("seats","seats"),("targets","targets")))):
            raise ValueError("Archive parity failed")
        ds=[b["hits"]-a["hits"] for a,b in pairs]
        hits=v["hits"]+sum(ds)
        summary.append(dict(metric=metric,seats=v["seats"],targets=v["targets"],context_hits=v["hits"],
            usage_hits=hits,delta_hits=sum(ds),requests_better=sum(d>0 for d in ds),
            requests_worse=sum(d<0 for d in ds),usage_precision=hits/v["seats"],usage_recall=hits/v["targets"]))
    write_rows(out/"requests.csv",rows)
    write_rows(out/"summary.csv",summary)
    write_rows(out/"timing.csv",[dict(operation=k,cohort="calibration24",calls=len(v),
        median_us=float(np.median(v))/1e3,p95_us=float(np.percentile(v,95))/1e3) for k,v in timings.items()])
    (out/"weights.json").write_text(json.dumps(dict(calibration_sha256=sha(CALIBRATION),
        schema="llada-committed-usage-v1",weights=trained.tolist()),allow_nan=False)+"\n")
    gains=[r["delta_hits"] for r in summary if r["metric"].startswith("Q")]
    print("Advance to sparse:", min(gains)>=0 and sum(x>0 for x in gains)>=2)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=RAW/"committed-usage-v1")
    p.add_argument("--checkpoint",type=Path,default=Path("/data0/ysy/models/LLaDA2.1-mini"))
    a=p.parse_args()
    run(a.out,a.checkpoint)

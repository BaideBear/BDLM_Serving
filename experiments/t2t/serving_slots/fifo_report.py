"""Score the closed-loop FIFO run; wall-clock units are ms, not TPF."""
import argparse
from bisect import bisect_right
import csv
from itertools import combinations
import json
from math import ceil
from pathlib import Path
from statistics import mean, median

from bdlm_serving.evaluation.frozen import sha
from bdlm_serving.evaluation.frozen90 import load_tasks
from experiments.t2t.prefix_window.score_run import link_requests
from .fifo import MODES, SLOTS, Pressure, arrival_trace
from tools.prefix_cohorts import score_files


def validate(times,trace,mode,slots):
    seq = [r["enqueued"] for r in times]
    if len(times)!=len(trace) or sorted(seq)!=seq:
        raise ValueError("Incomplete or reordered FIFO arrivals")
    previous = -1
    state = Pressure(mode,slots)
    for rank,(r,(i,target)) in enumerate(zip(times,trace,strict=True)):
        a,s,f = r["enqueued"],r["start"],r["finish"]
        if (r["index"]!=i or r["planned"]!=target or not target<=a<=s<f or s<previous):
            raise ValueError("FIFO schedule, interval or order mismatch")
        previous,prior = f,s
        for j,(tick,t,depth,q) in enumerate(r["offers"]):
            if (tick!=j or not prior<=t<=f or depth!=bisect_right(seq,t)-rank-1
                    or state.offer(depth)!=q):
                raise ValueError("Round used the wrong arrival snapshot or offered budget")
            prior = t


def p95(values):
    return sorted(values)[ceil(.95*len(values))-1]


def clock_quality(times,unit):
    span = times[-1]["planned"] - times[0]["planned"]
    lags = [r["enqueued"]-r["planned"] for r in times]
    gap_error = [abs((v["enqueued"]-u["enqueued"])-(v["planned"]-u["planned"]))
                 for j,(u,v) in enumerate(zip(times,times[1:])) if j%6 != 5]
    span_error = abs(lags[-1]-lags[0])/span
    healthy = p95(lags)<=unit/80 and p95(gap_error)<=unit/80 and span_error<=.01
    return dict(clock_pass=healthy,arrival_span_relerr=span_error,
                in_burst_gap_abs_error_p95_ms=p95(gap_error)/1e6,
                achieved_requests_s=len(times)*1e9/(times[-1]["finish"]-times[0]["planned"]))


def summarize(run,cases,out):
    read=lambda p: json.loads(p.read_text())
    meta = read(run/"run.json")
    tasks = load_tasks(cases)
    slots = SLOTS[meta["model"]]
    trace = arrival_trace(tasks,meta["unit_ns"])
    if (len(trace)!=90 or len({i for i,_ in trace})!=90 or meta["suite"]!="frozen90" or
            meta["samples_sha256"]!=sha(cases) or list(map(tuple,meta["trace"]))!=trace or
            meta["policies"]!=list(MODES) or meta["batch_size"]!=1 or meta["slots"]!=list(slots)):
        raise ValueError("Wrong FIFO metadata")
    out.mkdir(exist_ok=True,parents=True)
    dirs, summary = {}, []
    for mode in MODES:
        if read(run/f"complete-{mode}.json")!=dict(policy=mode,requests=90):
            raise ValueError("Incomplete FIFO arm")
        dirs[mode]=link_requests(run,mode,out/".inputs")
        clock = read(run/f"clock-{mode}.json")
        if clock["mode"]!=mode:
            raise ValueError("Wrong clock arm")
        times = clock["requests"]
        validate(times,trace,mode,slots)
        clock_stats = clock_quality(times,meta["unit_ns"])
        request_rows = {}
        for t in times:
            i=t["index"]
            raw=read(dirs[mode]/f"{i:02d}-{mode}.json")
            if [e["slot_offer"] for e in raw["events"] if "slot_offer" in e]!=[d[-1] for d in t["offers"]]:
                raise ValueError("Backend and clock offers disagree")
            request_rows[i]=raw
        for scope,start,end in (("original30",0,30),("added60",30,90),("all90",0,90)):
            rs=[t for t in times if start<=t["index"]<end]
            def ms(x): return x/1e6
            result=dict(model=meta["model"],mode=mode,scope=scope,n=len(rs),unit_ms=ms(meta["unit_ns"]))
            for metric,a,b in (("inject_lag","planned","enqueued"),("wait","enqueued","start"),
                               ("service","start","finish"),("response","enqueued","finish"),
                               ("scheduled_response","planned","finish")):
                vals=[ms(r[b]-r[a]) for r in rs]
                result[metric+"_mean_ms"]=mean(vals)
                result[metric+"_p50_ms"]=median(vals)
                result[metric+"_p95_ms"]=p95(vals)
            result["max_inject_lag_ms"]=max(ms(r["enqueued"]-r["planned"]) for r in rs)
            result["low_offers"]=sum(d[-1]==slots[0] for r in rs for d in r["offers"])
            result["high_offers"]=sum(d[-1]==slots[1] for r in rs for d in r["offers"])
            result["requests_switched"]=sum(len({d[-1] for d in r["offers"]})>1 for r in rs)
            if scope=="all90":
                result.update(clock_stats)
            else:
                result.update(dict.fromkeys(clock_stats,""))
            result["N"],result["F"],result["physical_Q"]=(sum(request_rows[r["index"]][k] for r in rs)
                for k in ("output_tokens","generation_forwards","generation_physical_padded_rows"))
            summary.append(result)
    for ctl,arm in combinations(MODES,2):
        score_files(dirs[ctl],dirs[arm],cases,out/f"{ctl}-vs-{arm}.csv",baseline_policy=ctl,arm_policy=arm)
    with (out/"fifo-clock.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator="\n")
        w.writeheader()
        w.writerows(summary)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("run","cases","out"):
        p.add_argument("--"+k,type=Path,required=True)
    summarize(**vars(p.parse_args()))

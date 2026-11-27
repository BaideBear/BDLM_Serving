"""Reconcile archived strict Optimus / fixed FlowBlock GPU measurements with frozen90 scores."""
import csv
import json
from math import isclose
from pathlib import Path
from statistics import fmean
import sys

from bdlm_serving.evaluation.frozen import sha
from bdlm_serving.evaluation.frozen90 import load_tasks
from .llada_all_slots90 import DATA, require

HERE = Path(__file__).resolve().parent
COHORTS = (("original-0-29",slice(0,30)),("added-30-89",slice(30,90)),("all-0-89",slice(0,90)))


def sources():
    for model, tiers in (("llada21",(4,8,16,32)),("gemma",(32,64,128,256))):
        for q in tiers:
            for mode in ("ibs","obs"):
                if model=="llada21":
                    p=f"optcap_{mode}_{q}"
                    group={4:"low",8:"endpoints",16:"endpoints",32:"anchor"}[q]
                    folder=DATA/("llada21-optimus-hardcap-"+group)
                    src=(HERE/"controls"/(f"llada21-q{q}-{mode}-original-vs-capped.csv" if q<32
                                         else "llada21-original-vs-capped-q32.csv"))
                else:
                    p=f"{mode}_cap_{q}"
                    folder=(DATA/"gemma-fixed-caps-q32q64-v1" if q<128 else
                            DATA.parent/"private-optimus-cap90-v1/gemma-optimus-cap90-v1")
                    src=HERE/"controls"/("gemma-fixed-caps-paired.csv" if q<128 else "gemma-original-vs-capped-fixed.csv")
                yield model,q,mode,p,folder,src,"arm_"
        b=32 if model=="llada21" else 256
        for w in (1,2):
            p=f"flow_w{w}"
            folder=DATA/("llada21-reference90-wave" if model=="llada21" else "gemma-wave-reference90")
            src=HERE/"controls"/({"llada21":("llada21-native32-vs-flow-w1.csv","llada21-dual64-vs-flow-w2.csv"),
                    "gemma":("gemma-wave-vs-fixed.csv","gemma-fixed512-active-vs-flow.csv")}[model][w-1])
            yield model,b*w,p,p,folder,src,"baseline_"


def reconcile(model,q,label,policy,group,src,side,tasks,suite_sha):
    read=lambda p: json.loads(p.read_bytes())
    meta=read(group/"run.json")
    require(meta["samples_sha256"]==suite_sha and policy in meta["policies"] and
            read(group/f"complete-{policy}.json")==dict(policy=policy,requests=90),"Different frozen run")
    if not policy.startswith("flow"):
        require(meta["hard_q_cap"] and meta["cap_clock"]=="all_decoder_forwards_request_local","Not a fixed cap")
    tape=[]
    for i,task in enumerate(tasks):
        row=read(group/f"{i:02}-{policy}.json")
        require(row["policy"]==policy and all(row[k]==task[k] for k in ("index","dataset","task_id","seed")),
                "Task mismatch")
        if model=="llada21":
            events=row["events"]
            counts=[e.get("physical_forwards",1) for e in events]
            require(counts==[1]*len(events) and len(events)==len(row["calls"]),"Multi-call event")
            gen=[call["physical_rows"] for e,call in zip(events,row["calls"],strict=True) if e["phase"]!="prefill"]
            require(all(c["batch"]==1 and c["physical_rows"]==e["q_tokens"] for e,c in
                        zip(events,row["calls"],strict=True)),"LLaDA batch/width mismatch")
            enc=[]
        else:
            prompt,*calls=row["events"]
            require(prompt[0]=="encoder" and set(x[0] for x in calls)<={"encoder","decoder"},
                    "Invalid Gemma prefill or event")
            gen=[w for typ,w in calls if typ=="decoder"]
            enc=[w for typ,w in calls if typ=="encoder"]
            require([e["q"] for e in row["trace"]]==gen,"Gemma decoder trace width mismatch")
        n=row["output_tokens"]
        f=len(gen)+len(enc)
        paid=sum(gen)+sum(enc)
        require(gen and min(gen)>0 and max(gen)<=q and row["generation_forwards"]==f
                and row["generation_physical_padded_rows"]==paid and row["tpf"]==n/f,
                f"Cap / cost mismatch: {policy} index {i}")
        tape.append((n,f,paid,len(gen),sum(gen),max(gen),row["status"]!="success"))
    with src.open(newline="") as stream:
        scores=[r for r in csv.DictReader(stream) if r[side+"policy"]==policy and r["scope"]=="all"]
    out=[]
    for cohort,indices in COHORTS:
        rows=tape[indices]
        n,f,paid,df,dq,_,fail=map(sum,zip(*rows,strict=True))
        tpf=fmean(r[0]/r[1] for r in rows)
        matches=[r for r in scores if r["cohort"]==cohort]
        require(matches and all(len(rows)==int(r["requests"]) and
                (n,f,paid)==tuple(int(r[side+k]) for k in ("N","F","physical_Q")) and
                isclose(tpf,float(r[side+"macro_tpf"]),rel_tol=1e-8) and
                isclose(n/paid,float(r[side+"pooled_token_efficiency"]),rel_tol=1e-8) for r in matches),
                f"Published score/work does not match raw archive: {src.name} {policy} {cohort}")
        correct,={int(r[side+"correct"]) for r in matches}
        out.append(dict(model=model,budget=q,method=label,cohort=cohort,requests=len(rows),correct=correct,
               macro_TPF=tpf,efficiency=n/paid,N=n,F=f,Q=paid,decoder_F=df,decoder_Q=dq,encoder_Q=paid-dq,
               mean_decoder_Q=dq/df,max_decoder_Q=max(r[5] for r in rows),failures=fail,
               raw_group=group.name,raw_policy=policy,score_source=src.name))
    return out


def main():
    samples=DATA/"samples.json"
    tasks=load_tasks(samples)
    rows=[row for arm in sources() for row in reconcile(*arm,tasks,sha(samples))]
    writer=csv.DictWriter(sys.stdout,fieldnames=list(rows[0]),lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


if __name__=="__main__":
    main()

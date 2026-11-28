"""Physical H4 six only. `report` never loads a checkpoint."""
import argparse
import os
from pathlib import Path
import time
from unittest.mock import patch
from adapter import install
from replay import (D, ROOT, DATA, audit, digest, filehash, json, need, read)

from bdlm_serving.evaluation import scoring
from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.methods.optimus import fixed, upstream
from experiments.t2t.serving_slots import ibs_prefix_pair as pair, shared_llada_worker as worker
from experiments.t2t.serving_slots.shared_fixed_shard import check_board

ARMS=("none","right","change")
CASES=[7,13,25,33,55,71]
PRIOR=D.parent/"llada-ibs-q8-prefix-change-rank-h4-six-v1"
RETAINED=PRIOR/"07-none.json"
RETAINED_SHA="4c9aaab631241e268aa5b1e0b5e8f90af603da24f8967b18f5df2bdd82ac613a"
def save(path,obj):
    with path.open("x") as f:
        print(json.dumps(obj,sort_keys=True,separators=(",",":")),file=f)


def control_ref(r,h,cert):
    n=cert["event"]+1 if cert else len(h["events"])
    for k in ("events","calls","forward_inputs"):
        need(r[k][:n]==h[k][:n],f"historical ordinary {k}")
    need(r["output_ids"]==h["output_ids"] and len(r["events"])==n+bool(cert and cert["due"]),
         "historical result/paid close")
    if cert and cert["due"]:
        following=dict(zip(h["events"][n]["absolute_positions"],h["forward_inputs"][n],strict=True))
        need([following[a] for a in cert["due"]]==r["forward_inputs"][-1],"historical debt inputs")


def verify_mask(masks,r):
    for (pq,kv,all_true,block_causal),e,c in zip(masks,r["events"],r["calls"],strict=True):
        need(pq==e["absolute_positions"] and kv==c["kv_rows"] and
             (block_causal if e["phase"]=="prefill" else all_true), "unregistered physical visibility")


def first_pair(right,change):
    a,b=right["row"],change["row"]
    off=int(a["events"][0]["phase"]=="prefill")
    for n,(rt,ct) in enumerate(zip(right["tape"],change["tape"])):
        j=n+off
        need(rt["plan"]==ct["plan"],"versions diverged before the first different set")
        ea,eb=a["events"][j],b["events"][j]
        for k in ("selected_local","candidate_local","dirty_before","selected_masked","cached_prefix_before",
                  "block","step","visible_kv","physical_cap"):
            need(ea[k]==eb[k],"original selection differs before fork")
        za,zb=rt["plan"]["right"],ct["plan"]["change"]
        if za!=zb:
            need(len(za)==len(zb) and ea["q_tokens"]==eb["q_tokens"],"unequal count at fork")
            aa,bb=(dict(zip(e["absolute_positions"],r["forward_inputs"][j],strict=True))
                   for e,r in ((ea,a),(eb,b)))
            need(all(aa[x]==bb[x] for x in set(aa)&set(bb)),"incoming token drift")
            return dict(tick=n,block=ea["block"],ordinary=ea["prefix_refresh_original_absolute"],
                        dirty=ea["dirty_before"],right=za,change=zb,physical_Q=ea["q_tokens"],
                        candidates=rt["plan"]["candidates"],score_units=rt["plan"]["scores"],
                        writes_right=rt["changes"],writes_change=ct["changes"])
        need(rt==ct and a["forward_inputs"][j]==b["forward_inputs"][j] and
             {k:v for k,v in ea.items() if k!="prefix_refresh_order"}==
             {k:v for k,v in eb.items() if k!="prefix_refresh_order"},"pre-fork trajectory differs")
    need(len(right["tape"])==len(change["tape"]) and a["output_ids"]==b["output_ids"],"unexplained endpoint")
    return None


def main(mode):
    reg=read(D/"registration.json")
    tag=filehash(D/"registration.json")
    hx=read(D/"exposure.json")
    need(hx["registration"]==tag and hx["gate"] and hx["different_sets"]>0 and
         reg["cases"]==CASES and tuple(reg["arms"])==ARMS,"gate or panel")
    hroot=DATA/reg["history"]
    src=ROOT/"runtime/formal90-7a38ffc/src"
    run=read(hroot/"run.json")
    need(filehash(hroot/"run.json")==reg["historical_run_sha256"] and
         filehash(DATA/"samples.json")==reg["panel_sha256"] and
         all(filehash(src/k)==v for k,v in run["sources"].items()),"shared source or inputs")
    tasks=load_tasks(DATA/"samples.json")
    for a in ARMS:
        install(fixed,a,[])
    if mode=="check":
        return print("exact sources and three executable request-local methods verified")
    if mode=="run":
        need(not list(D.glob("??-*.json")),"refuse rerun")
        pre=read(D/"preflight.json")
        need(pre["same_raw_sha"]==filehash(RETAINED)==RETAINED_SHA and pre["passed"],"preflight missing")
        check_board("n164:H4")
        import torch
        model_handle=worker.load(Path("/data0/ysy/models/LLaDA2.1-mini"))
        model,tok=model_handle["model"],model_handle["tok"]
        need(model.device.type=="cuda","no CPU inference")
    result=[]
    controls=ROOT/"sources/optimus-controls"
    with patch.object(upstream,"control_directory",lambda _:controls):
        for k,i in enumerate(CASES):
            task=tasks[i]
            refs={}
            for a,label in zip(("none","right"),("0","2"),strict=True):
                hp=hroot/f"{i:02}-{run['arms'][label]}.json"
                x,=[v for v in hx["records"] if v["index"]==i and v["arm"]==a]
                need(x["raw_sha"]==filehash(hp),"historical bytes")
                refs[a]=read(hp)
            need(all(filehash(controls/f)==v for f,v in refs["none"]["extra"]["source_hashes"].items()),
                 "upstream source")
            arms,stats={},{}
            order=ARMS[k%3:]+ARMS[:k%3]
            for a in order:
                path=D/f"{i:02}-{a}.json"
                if (i,a)==(7,"none"):
                    path=RETAINED
                    need(filehash(path)==RETAINED_SHA,"retained v1 bytes changed")
                elif mode=="run":
                    need(time.time()<reg["deadline"]-90,"deadline")
                    prompt,_=worker.canonical_prompt(tok,task,refs["none"]["prompt_ids"])
                    torch.manual_seed(task["seed"])
                    masks,tape=[],[]
                    def mask_hook(_,args,kw):
                        pos,m=kw["position_ids"][0],kw["attention_mask"]
                        need(m.dtype==torch.bool and m.shape[:3]==(1,1,len(pos)),"physical input mask")
                        causal=False
                        if not masks and len(pos)==m.shape[-1]:
                            causal=torch.equal(m[0,0],pos[:,None]//32>=pos[None,:]//32)
                        masks.append((pos.tolist(),m.shape[-1],bool(m.all()),causal))
                    with model.register_forward_pre_hook(mask_hook,with_kwargs=True), \
                         patch.object(fixed.FixedOptimus,"run",install(fixed,a,tape)):
                        r=pair.generate(model,tok,task,prompt,2048,0 if a=="none" else 2)
                    if a=="change":
                        r["policy"]="ibs_q8_spare2_change_kvonly"
                        r["extra"]["prefix_order"]="change"
                    save(path,dict(row=r,arm=a,tape=tape,masks=masks,registration=tag))
                raw=arms[a]=read(path)
                r=raw["row"]
                need(raw["arm"]==a and raw["registration"]==tag and
                     all(r[v]==refs["none"][v] for v in ("index","seed","dataset","task_id","prompt_ids")),
                     "wrong identity")
                verify_mask(raw["masks"],r)
                audited,_=audit(r,a,live=True,tape=raw["tape"])
                if a in refs: control_ref(r,refs[a],audited["certificate"])
                cert=audited["certificate"]
                stats[a]=dict(N=r["output_tokens"],F=r["generation_forwards"],Q=r["generation_query_rows"],
                    correct=r["status"]=="success" and bool(scoring.score(r["output_text"],task)["correct"]),
                    status=r["status"],eos=r["eos"],ids_sha=digest(r["output_ids"]),
                    roles=audited["counts"],certificate=cert,raw_sha=filehash(path),prefill_Q=r["prefill_rows"])
            first=first_pair(arms["right"],arms["change"])
            row=dict(index=i,seed=task["seed"],task_id=task["task_id"],dataset=task["dataset"],order=order,
                     registration=tag,first=first,arms=stats,history_replayed=True)
            print(json.dumps(row,sort_keys=True,separators=(",",":")),flush=True)
            result.append(row)
    if mode=="run":
        save(D/"complete.json",dict(cases=CASES,requests=18,new_requests=17,prior_raw_sha=RETAINED_SHA,
                                    registration=tag,sha=digest(result),
                                    exposure=filehash(D/"exposure.json"),extension=False))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("check","run","report"))
    main(p.parse_args().mode)

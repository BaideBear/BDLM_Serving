"""Paired registered Q32/Q64 frontier admission; raw results stay on the GPU host."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import mean
import time

from bdlm_serving.evaluation import scoring
from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.methods.t2t.budget_family import Law
from .llada_agreement_eval import sha
from .llada_close_factor import ANCHORS,SAMPLES
from .llada_fixed_family_eval import CAP,KEYS,ORDER,SCORER_SHA,evaluate_fixed,require
from .llada_frontier_accept import check_first,frontier_class
from .llada_seat_report import paired


PARENTS={32:("llada-close-1128","bb1d1aa65beb2ee1eb71ce1a8b823392e3c398438007e99de647f632269e80c8"),
         64:("llada-q64-paid-agree-1128","7a25e37235717782a931067144053c2a5bd1f5e1dd2da54c3e93cf4a51d8718a")}
PROTOCOL="llada-fixed-warm-head-frontier-v1"


def compact(r):
    return dict(**{k:r[k] for k in ("N","F","Q","failure","correct","max_decoder_Q")},
                output_digest=sha256(json.dumps(r["output_ids"]).encode()).hexdigest())


def run(a):
    import torch
    from . import llada_frontier_accept, llada_agreement
    from .cross_race_llada import load_gpu,save
    from .llada_hfull_single_factor import request_classes
    q=a.q; device=os.environ["CUDA_VISIBLE_DEVICES"]
    parent=Path("/gs/results")/PARENTS[q][0]
    reg=json.loads((parent/"run.json").read_bytes())
    files=sorted((parent/"rows").glob("*.json"))
    require(sha(a.samples)==SAMPLES and sha(Path(scoring.__file__))==SCORER_SHA
            and sha256(b"".join(bytes.fromhex(sha(p)) for p in files)).hexdigest()==PARENTS[q][1]
            and device.startswith("GPU-") and torch.cuda.device_count()==1 and reg["device"]==device
            and reg["checkpoint"]==str(a.checkpoint) and len(files)==90,"unsealed parent or GPU")
    solo=a.scope=="solo"
    require(not solo or q==64,"solo follow-up is registered at Q64")
    tasks=load_tasks(a.samples)
    a.out.mkdir(exist_ok=False); (a.out/"rows").mkdir()
    save(a.out/"run.json",dict(protocol=PROTOCOL,device=device,q=q,parent=PARENTS[q],scope=a.scope,
         samples=SAMPLES,scorer=SCORER_SHA,checkpoint=str(a.checkpoint),order=ORDER,
         sources={p.name:sha(p) for p in map(Path,(__file__,llada_frontier_accept.__file__,
                                                    llada_agreement.__file__))},pid=os.getpid()))
    tok,model=load_gpu(a,lambda:None);mask=tok.convert_tokens_to_ids("<|mask|>")
    _,base,_=request_classes()
    if q==64:
        base=llada_agreement.request_class(base)
        kwargs=dict(cross=("quarter",False),coverage="head",warm_head_sample=True)
    else:
        kwargs=dict(cross=("head",False),close_policy="edit_once")
    classes={"old":base,"frontier":frontier_class(base,solo=solo)};added=excluded=0
    for j,i in enumerate(ORDER):
        require(time.time()<a.deadline-60,"deadline")
        native=json.loads((a.reference/f"{i:02d}-native32.json").read_bytes())
        require(native["index"]==i and native["generation_cap"]==CAP,"different prompt source")
        hist=json.loads(files[i].read_bytes())
        arms={k:evaluate_fixed(model,tok,tasks[i],native["prompt_ids"],q,Law(),
                              request_cls=classes[k],audit_admission=True,**kwargs)
              for k in (("old","frontier") if j%2==0 else ("frontier","old"))}
        old=arms["old"]; actual=compact(old)
        oldref=hist["repeat"] if q==64 else hist["arms"]["edit_once"]
        oldref=dict(oldref) if q==64 else compact({**oldref,"output_ids":oldref["ids"]})
        require(actual==oldref and all(all(r[k]==hist[k] for k in KEYS)
                and r["input_sha256"]==hist["input"] for r in arms.values()),"reference did not replay")
        blocked=[]
        first,n,calls=check_first(old,arms["frontier"],mask,q,solo=solo,excluded=blocked)
        added+=n;excluded+=sum(blocked)
        save(a.out/"rows"/f"{i:02d}.json",dict(**{k:old[k] for k in KEYS},input=hist["input"],
             reference_equal=True,first=first,added=n,changed_F=calls,
             **(dict(joint_withheld=sum(blocked)) if solo else {}),
             old=actual,frontier=compact(arms["frontier"])))
        print(i,"additional admissions",n,flush=True)
        if j+1==len(ANCHORS) and (not added or (solo and not excluded)):
            print("No distinct intervention on anchors; stopping.",flush=True)
            return


def report(root, manifest):
    meta=json.loads((root/"run.json").read_bytes()); q=meta["q"]
    solo=meta.get("scope","all")=="solo"
    require(meta.get("scope","all") in ("solo","all") and (not solo or q==64)
            and meta["protocol"]==PROTOCOL and meta["parent"]==list(PARENTS[q])
            and meta["order"]==list(ORDER) and meta["samples"]==manifest["samples_sha256"]==SAMPLES
            and meta["scorer"]==SCORER_SHA,"invalid registration")
    rows={}; files=sorted((root/"rows").glob("*.json"))
    for p in files:
        r=json.loads(p.read_bytes()); i=r["index"]
        require(p.name==f"{i:02d}.json" and all(r[k]==manifest["samples"][i][k] for k in KEYS)
                and r["reference_equal"] and bool(r["first"])==bool(r["added"])==bool(r["changed_F"])
                and 0<=r["changed_F"]<=r["added"] and (r["first"] or r["old"]==r["frontier"])
                and (not solo or r["joint_withheld"]>=0)
                and all(0<v["max_decoder_Q"]<=q and v["F"]<=v["Q"]<=v["F"]*q and v["N"]>=0
                        for v in (r["old"],r["frontier"])),"invalid receipt")
        rows[i]=r
    require(set(rows)==set(ORDER[:len(rows)]) and (len(rows)<=3 or
            any(rows[i]["added"] for i in ANCHORS) and (not solo or any(rows[i]["joint_withheld"] for i in ANCHORS))),
            "invalid exposure or order")
    cohorts={}
    for name,idx in (("original30",range(30)),("new60",range(30,90)),("frozen90",range(90))):
        if not set(idx)<=rows.keys():continue
        part=[rows[i] for i in idx]
        g=cohorts[name]=dict(n=len(part),exposed=sum(bool(r["added"]) for r in part),
                added=sum(r["added"] for r in part),ci=paired(part,"old","frontier"),
                rescued=sum(r["frontier"]["correct"] and not r["old"]["correct"] for r in part),
                lost=sum(r["old"]["correct"] and not r["frontier"]["correct"] for r in part))
        if solo:
            g["joint_withheld"]=sum(r["joint_withheld"] for r in part)
        for k in ("old","frontier"):
            sums={key:sum(r[k][key] for r in part) for key in ("correct","N","F","Q")}
            g[k]=dict(**sums,failed=sum(bool(r[k]["failure"]) for r in part),
                      TE=sums["N"]/sums["Q"],macro_TPF=mean(r[k]["N"]/r[k]["F"] for r in part))
    return dict(registration=meta,finished=len(rows),cohorts=cohorts,
                digest=sha256(b"".join(bytes.fromhex(sha(p)) for p in files)).hexdigest())


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__); cmd=ap.add_subparsers(dest="action",required=True)
    live=cmd.add_parser("run")
    for field in ("checkpoint","samples","reference","out"):
        live.add_argument("--"+field,type=Path,required=True)
    live.add_argument("--q",choices=PARENTS,type=int,required=True)
    live.add_argument("--scope",choices=("all","solo"),default="all")
    live.add_argument("--deadline",type=int,required=True)
    rep=cmd.add_parser("report");rep.add_argument("root",type=Path);rep.add_argument("manifest",type=Path)
    a=ap.parse_args()
    if a.action=="run":run(a)
    else:print(json.dumps(report(a.root,json.loads(a.manifest.read_bytes())),indent=2,sort_keys=True))

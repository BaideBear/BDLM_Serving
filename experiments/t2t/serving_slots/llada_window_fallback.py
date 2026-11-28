"""Fixed H0 Q8: keep cold fallback when paid cold and waiting MASKs mix."""
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
from .llada_close_factor import ANCHORS, SAMPLES
from .llada_fixed_family_eval import CAP, KEYS, ORDER, SCORER_SHA, evaluate_fixed, require
from .llada_seat_report import paired


def protected(paid):
    class ColdFallback(paid):
        def _fallback_group(self,idx,group):
            if not self.extra_window_mask:
                return group
            # extra_window_mask is exclusively resident non-dirty waiting MASKs.
            candidates=group & ~((idx[...,None]==idx.new_tensor(self.extra_window_mask)).any(-1))
            require(bool(candidates.any()),"mixed Q8 fallback lost every cold MASK")
            return candidates
    return ColdFallback


def decision(e,mask,protect):
    """Independently reconstruct actual writes from measured proposals."""
    measured=list(zip(e["sampled"],e.get("proposals",[])))
    require(len(measured)==len(e["sampled"]) and len({p//32 for p,_ in measured})<=1,"not a measured head")
    unresolved=[(p,v) for p,v in measured if v[0]==mask]
    chosen=[(p,v) for p,v in unresolved if v[4]]
    if unresolved and not chosen:
        pool=unresolved
        if protect and e.get("paid_waiting_mask"):
            pool=[(p,v) for p,v in pool if p in e["cold"]]
            require(bool(pool),"no cold candidate in mixed fallback")
        chosen=[max(pool,key=lambda x:x[1][2])]
    return sorted(p for p,v in chosen if v[1]!=mask)


def compare(a,b,mask):
    first=None
    switches=0
    for key,arm in (("paid",a),("protected",b)):
        for e in arm["trace"]:
            d=decision(e,mask,key=="protected")
            require(d==sorted(e["admitted"]),"paid proposals do not reproduce admissions")
            if key=="protected" and d!=decision(e,mask,False):
                switches+=1
    for x,y in zip(a["trace"],b["trace"]):
        if decision(x,mask,False)==decision(x,mask,True):
            require(x==y,"trajectory diverged before first actual fallback exchange")
            continue
        require(all(x[k]==y[k] for k in ("pre_state","rows","sampled","proposals","dirty_before",
                    "layer_physical_q","sample_clock","phase","cold","q","edited","revoked","paid_waiting_mask"))
                and y["admitted"]==decision(x,mask,True),"intervention is not isolated")
        first=dict(tick=x["tick"],old=x["admitted"],new=y["admitted"],width=x["q"])
        break
    if not first:
        require(len(a["trace"])==len(b["trace"]) and all(a[k]==b[k] for k in
                ("output_ids","failure","score","F","N","Q")),"unexplained no-exposure result")
    require(bool(first)==bool(switches),"missing fallback exchange")
    return first,switches


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def run(a):
    import torch
    from bdlm_serving.methods.t2t import llada_budget_request as base
    from .cross_race_llada import load_gpu,save
    from .llada_hfull_single_factor import request_classes
    from . import llada_window_sample as sample
    sealed=json.loads(Path(__file__).with_name("controls").joinpath(
        "llada-slot-allocation","c8-paid-frozen90.json").read_bytes())
    parent=json.loads((a.parent/"run.json").read_bytes())
    raw=sorted((a.parent/"rows").glob("*.json"))
    uuid=os.environ["CUDA_VISIBLE_DEVICES"]
    require(sealed["registration"]==parent and parent["device"]==uuid and
            parent["checkpoint"]==str(a.checkpoint) and torch.cuda.device_count()==1 and
            [p.name for p in raw]==[f"{i:02d}.json" for i in range(90)] and
            sha256(b"".join(bytes.fromhex(sha(p)) for p in raw)).hexdigest()==sealed["rows_sha256"] and
            sha(a.samples)==parent["samples"]==SAMPLES and sha(Path(scoring.__file__))==SCORER_SHA and
            parent["sampling_sha256"]==sha(Path(sample.__file__)),"parent/input/device mismatch")
    tasks=load_tasks(a.samples)
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/"rows").mkdir()
    save(a.out/"run.json",dict(protocol="llada-q8-cold-fallback-v1",q=8,device=uuid,
         order=ORDER,samples=SAMPLES,scorer=SCORER_SHA,parent=sealed["rows_sha256"],
         sources={Path(p).name:sha(Path(p)) for p in (__file__,base.__file__,sample.__file__)},pid=os.getpid()))
    def ready():
        (a.handoff/"ready").touch(exist_ok=False)
        while not (a.handoff/"go").exists():
            require(time.time()<a.deadline,"expired")
            time.sleep(.2)
    tok,model=load_gpu(a,ready)
    _,paid=sample.variants(request_classes()[1])
    mask=tok.convert_tokens_to_ids("<|mask|>")
    variants={"paid":paid,"protected":protected(paid)}
    exposed=0
    for j,i in enumerate(ORDER):
        ref=json.loads((a.reference/f"{i:02d}-native32.json").read_bytes())
        old=json.loads(raw[i].read_bytes())
        require(time.time()<a.deadline-60 and i==ref["index"] and ref["generation_cap"]==CAP,"stop")
        res={k:evaluate_fixed(model,tok,tasks[i],ref["prompt_ids"],8,Law(),cross=("head",False),
             audit_admission=True,request_cls=variants[k]) for k in
             (tuple(variants) if j%2==0 else tuple(reversed(variants)))}
        first,swaps=compare(*[res[k] for k in variants],mask)
        row=dict(first=first,swaps=swaps,input=old["input"],**{k:tasks[i][k] for k in KEYS})
        for k,v in res.items():
            require(all(row[t]==v[t]==old[t] for t in KEYS) and row["input"]==v["input_sha256"]
                    and 0<v["max_decoder_Q"]<=8 and v["Q"]==sum(e["q"] for e in v["trace"])
                    and all(e["layer_physical_q"]==[e["q"]]*20 for e in v["trace"]),"unmatched bill")
            row[k]={t:v[t] for t in ("N","F","Q","failure","correct","max_decoder_Q")}
            row[k]["output_digest"]=sha256(json.dumps(v["output_ids"]).encode()).hexdigest()
        require(all(old["paid"][k]==v for k,v in row["paid"].items()),"same-device control differs")
        save(a.out/"rows"/f"{i:02d}.json",row)
        exposed+=bool(first)
        print(i,first,swaps,flush=True)
        if j+1==len(ANCHORS):
            require(exposed>0,"no actual fallback exchange on registered anchors")


def report(root,manifest):
    reg=json.loads((root/"run.json").read_bytes())
    require(reg["protocol"]=="llada-q8-cold-fallback-v1" and reg["samples"]==SAMPLES==manifest["samples_sha256"]
            and reg["order"]==list(ORDER) and reg["scorer"]==SCORER_SHA,"wrong run")
    rows={}
    digest=sha256()
    for path in sorted((root/"rows").glob("*.json")):
        blob=path.read_bytes(); r=json.loads(blob); i=r["index"]
        require(path.name==f"{i:02d}.json" and all(r[k]==manifest["samples"][i][k] for k in KEYS)
                and bool(r["first"])==bool(r["swaps"]) and (r["first"] or r["protected"]==r["paid"])
                and all(0<v["F"]<=v["Q"]<=8*v["F"] and (not v["correct"] or not v["failure"])
                        for v in [r["paid"],r["protected"]]),"wrong receipt")
        rows[i]=r; digest.update(sha256(blob).digest())
    require(set(rows)==set(ORDER[:len(rows)]),"unexpected rows")
    cohorts={}
    for k,ids in (("original30",range(30)),("new60",range(30,90)),("frozen90",range(90))):
        if not set(ids)<=rows.keys(): continue
        part=[rows[i] for i in ids]
        c=cohorts[k]=dict(n=len(part),exposed=sum(bool(r["first"]) for r in part),
            swaps=sum(r["swaps"] for r in part),ci95=paired(part,"paid","protected"),
            rescued=sum(not r["paid"]["correct"] and r["protected"]["correct"] for r in part),
            lost=sum(r["paid"]["correct"] and not r["protected"]["correct"] for r in part))
        for arm in ("paid","protected"):
            vals=[r[arm] for r in part]
            s={t:sum(v[t] for v in vals) for t in ("N","F","Q","correct")}
            c[arm]=dict(**s,macro_TPF=mean(v["N"]/v["F"] for v in vals),TE=s["N"]/s["Q"],
                        failed=sum(bool(v["failure"]) for v in vals))
    return dict(registration=reg,finished=len(rows),rows_sha256=digest.hexdigest(),cohorts=cohorts)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__); subs=ap.add_subparsers(dest="mode",required=True)
    ev=subs.add_parser("run")
    for k in ("out","samples","checkpoint","reference","handoff","parent"):
        ev.add_argument("--"+k,type=Path,required=True)
    ev.add_argument("--deadline",type=int,required=True)
    re=subs.add_parser("report");re.add_argument("root",type=Path);re.add_argument("manifest",type=Path)
    a=ap.parse_args()
    if a.mode=="run": run(a)
    else: print(json.dumps(report(a.root,json.loads(a.manifest.read_bytes())),indent=2,sort_keys=True))

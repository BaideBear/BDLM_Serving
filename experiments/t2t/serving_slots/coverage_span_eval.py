"""Same-card three-arm replay on nine fixed, independent requests per offer."""
import argparse
import json
import os
from hashlib import sha256
from pathlib import Path

from bdlm_serving.evaluation import scoring
from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.methods.t2t.budget_family import Law
from . import coverage_span
from .cross_race_llada import load_gpu, save
from .llada_fixed_family_eval import CAP,KEYS,SCORER_SHA,TIERS,evaluate_fixed,require
from .llada_hfull_single_factor import request_classes
from .llada_risk_ribbon import arms, compact, SAMPLES

ANCHORS=(1,11,21,31,41,51,61,71,81)


def qualify(old,new,mask):
    # Reconstruct the first difference before autoregressive trajectories diverge.
    for i,(a,b) in enumerate(zip(old["trace"],new["trace"])):
        require(all(a[k]==b[k] for k in ("rows","sampled","q","pre_state","dirty_before","edited","revoked",
                                         "layer_physical_q","sample_clock","physical_full_head"))
                and a.get("proposals",[])==b.get("proposals",[]),"nonisolated acceptance gate")
        if a["admitted"]==b["admitted"]: continue
        require(b["physical_full_head"],"additional acceptance without full current-head projection")
        h=b["pre_state"]["head"]
        paid={pos for pos,(t,d,_,low,_) in zip(b["sampled"],b["proposals"])
              if h<=pos<h+32 and t==mask and d!=mask and low}
        require(set(b["admitted"])==set(a["admitted"])|paid,"unpaid difference")
        return i
    require(compact(old)==compact(new),"no exposure but different result")
    return None


def run(a):
    import torch
    sha=lambda p:sha256(Path(p).read_bytes()).hexdigest()
    require(sha(a.samples)==SAMPLES and sha(scoring.__file__)==SCORER_SHA
            and torch.cuda.device_count()==1 and os.environ["CUDA_VISIBLE_DEVICES"].startswith("GPU-"),
            "unexpected manifest, scorer or GPU")
    tasks=load_tasks(a.samples)
    a.out.mkdir(exist_ok=False)
    save(a.out/"run.json",dict(protocol="coverage-span-v1",offer=a.q,indices=ANCHORS,samples=SAMPLES,scorer=SCORER_SHA,
         device=os.environ["CUDA_VISIBLE_DEVICES"],sources={p.name:sha(p) for p in map(Path,(__file__,coverage_span.__file__))}))
    tok,model=load_gpu(a,lambda:None)
    live,livekw=arms(a.q)
    base=request_classes()[1]
    kw=dict(cross=("quarter" if a.q==64 else "head",False))
    specs=dict(live=(live["live"],livekw),span=(coverage_span.request(base,False),kw),
               covered=(coverage_span.request(base,True),kw))
    for j,i in enumerate(ANCHORS):
        ref=json.loads((a.reference/f"{i:02d}-native32.json").read_bytes())
        require(ref["index"]==i and ref["generation_cap"]==CAP,"wrong native prompt")
        order=list(specs); shift=j%len(order); vals={}
        for k in order[shift:]+order[:shift]:
            cls,kwarg=specs[k]
            vals[k]=evaluate_fixed(model,tok,tasks[i],ref["prompt_ids"],a.q,Law(),
                                   request_cls=cls,audit_admission=True,**kwarg)
        require(all(all(r[k]==tasks[i][k] for k in KEYS) and
                    r["input_sha256"]==vals["live"]["input_sha256"] and r["max_decoder_Q"]<=a.q
                    for r in vals.values()),"input or physical cap mismatch")
        first=qualify(vals["span"],vals["covered"],tok.convert_tokens_to_ids("<|mask|>"))
        row=dict(**{k:tasks[i][k] for k in KEYS}, input=vals["live"]["input_sha256"],first_covered=first)
        for k,r in vals.items():
            row[k]=dict(**compact(r),two_block_F=r["stats"]["dual_block_F"] if "stats" in r and "dual_block_F" in r["stats"]
                        else sum(e["head_q"]>0 and e["tail_q"]>0 for e in r["trace"]))
        save(a.out/f"{i:02}.json",row)
        print(i,first,[vals[k]["F"] for k in specs],flush=True)
    save(a.out/"complete.json",dict(indices=ANCHORS,q=a.q))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("checkpoint","samples","reference","out"):
        ap.add_argument("--"+k,type=Path,required=True)
    ap.add_argument("--q",type=int,choices=TIERS,required=True)
    run(ap.parse_args())

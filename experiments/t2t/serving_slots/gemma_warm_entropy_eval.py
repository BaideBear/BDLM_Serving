"""One registered A1 factor. Compare both arms on-device and check fixed replay."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time

from .gemma_joint_verify_eval import QUAL, ORDER, check, digest
from .layout_sweep import write

A1="GPU-861c9b5d-1e73-2550-f832-b73988bc64ff"
BILL=("decoder_F","decoder_Q","encoder_F","encoder_Q","F","route_F","route_Q","max_decoder_Q","joint_F")


def available(own):
    check(os.environ.get("CUDA_VISIBLE_DEVICES")==A1,"Only physical A1")
    lines=subprocess.check_output(["nvidia-smi","-i","1","--query-compute-apps=pid,gpu_uuid",
                                  "--format=csv,noheader,nounits"],text=True).splitlines()
    check(all(own and line.split(", ")==[str(os.getpid()),A1] for line in lines),"A1 occupied")


def inspect(r):
    counts=Counter()
    for e in r["trace"]:
        p,=e["parts"]
        warm=e["head_initialized_before"]==256 and bool(p["sampled"])
        check(warm==e["warm_sample"] and (not warm or e["route"]=="sparse"),"Bound outside warm sample")
        phase=("warm_sample" if warm else "cold_sample" if p["sampled"] else p["phase"])
        counts[phase+"_F"]+=1
        counts[phase+"_Q"]+=e["q"]
        if warm:
            counts["warm_admitted"]+=len(p["admitted"])
            counts["warm_revoked"]+=len(p["revoked"])
    check(r["projection_modules"]==85 and r["max_decoder_Q"]<=128 and not r["joint_F"]
          and sum(n for k,n in counts.items() if k.endswith("_F"))==r["decoder_F"]
          and sum(n for k,n in counts.items() if k.endswith("_Q"))==r["decoder_Q"],
          "Projection bill mismatch")
    return dict(counts)


def first_difference(off,on):
    chain=sha256()
    for x,y in zip(off["trace"],on["trace"]):
        if x==y:
            chain.update(digest(x).encode())
            continue
        keys=("tick","ordinal","head_mask_before","head_initialized_before","warm_sample","positions","route",
              "sampled_absolute","dirty_before_absolute","encoder_F","offer")
        check(x["warm_sample"] and all(x[k]==y[k] for k in keys),"First divergence outside same warm input")
        p,t=x["parts"][0],y["parts"][0]
        check(all(p[k]==t[k] for k in ("rows","sampled","prefix","age_before","age_after","dirty_before"))
              and set(p["admitted"])<=set(t["admitted"]) and set(t["revoked"])<=set(p["revoked"]),
              "First changed proposal is not an expansion of native entropy selection")
        extra=sorted((set(t["admitted"])-set(p["admitted"]))|(set(p["revoked"])-set(t["revoked"])))
        check(bool(extra),"No changed acceptance at first differing tick")
        return dict(tick=x["tick"],ordinal=x["ordinal"],age=p["age_before"],extra=extra,
                    measured=len(x["positions"]),predecessor_sha256=chain.hexdigest())
    check(off["trace"]==on["trace"] and off["output_ids"]==on["output_ids"],"Unexplained trajectory change")
    return None


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ("out","samples","checkpoint","history"):
        ap.add_argument("--"+n,type=Path,required=True)
    ap.add_argument("--deadline",type=int,required=True)
    a=ap.parse_args()
    available(False)
    import torch,transformers
    from transformers import AutoProcessor,DiffusionGemmaForBlockDiffusion
    from transformers.models.diffusion_gemma import generation_diffusion_gemma as native
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.evaluation.frozen import sha
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.methods.t2t.budget_family import Law
    from .gemma_warm_entropy import auditor
    from .mask_pair_gemma import SAMPLES_SHA,input_digest
    check(time.time()<a.deadline and torch.cuda.device_count()==1 and sha(a.samples)==SAMPLES_SHA and
          sha(Path(native.__file__))=="cded57b7e365f0265b1c088e993c358b5f70acdf48055fce396540b4d9d691bc"
          and transformers.__version__=="5.12.0" and a.checkpoint.name=="diffusiongemma-cd40195","Wrong runtime")
    tasks=load_tasks(a.samples)
    source={n:sha(Path(__file__).with_name(n+".py")) for n in ("gemma_warm_entropy","gemma_warm_entropy_eval",
             "gemma_fixed_dual","gemma_fixed_dual_eval","gemma_strict_sampler","gemma_unified_audit")}
    audit=auditor()
    proc=AutoProcessor.from_pretrained(a.checkpoint,local_files_only=True)
    available(False)
    model=DiffusionGemmaForBlockDiffusion.from_pretrained(a.checkpoint,local_files_only=True,dtype=torch.bfloat16,
           attn_implementation="sdpa",experts_implementation="grouped_mm",device_map={"":"cuda:0"}).eval()
    special=set(proc.tokenizer.all_special_ids)|set(model.generation_config.eos_token_id)
    scorer=sha(Path(scoring.__file__))
    a.out.mkdir(parents=True,exist_ok=True)
    positive=0
    with torch.inference_mode():
        for j,i in enumerate(ORDER):
            available(True)
            check(time.time()<a.deadline,"Deadline")
            dst=a.out/f"C128-pair-{i:02}.json"
            src=a.history/f"C128-profile-fuse1-{i:02}.json"
            if dst.exists():
                row=json.loads(dst.read_bytes())
                check(row["source"]==source and row["index"]==i and row["ref_sha256"]==sha(src),"Cannot resume mixed run")
            else:
                task=tasks[i]
                h=json.loads(src.read_bytes())
                data=proc.apply_chat_template([{"role":"user","content":task["prompt"]}],tokenize=True,
                            add_generation_prompt=True,return_dict=True,return_tensors="pt").to(model.device)
                check(h["budget"]==128 and h["fuse"] is True and h["gate"]=="off" and h["prefix"]==[1,4]
                      and h["input_sha256"]==input_digest(data) and h["samples_sha256"]==SAMPLES_SHA
                      and h["scorer_sha256"]==scorer and h["torch"]==str(torch.__version__)
                      and all(h[k]==task[k] for k in ("index","dataset","seed","task_id")),"Wrong control")
                runs={}
                for enabled in ((False,True) if j%2==0 else (True,False)):
                    r=audit(model,data,q=128,seed=task["seed"],ablation="00",law=Law(1,4),
                            policy={"warm_entropy":enabled})
                    r["phases"]=inspect(r)
                    ids=r["output_ids"]
                    text=proc.tokenizer.decode(ids,skip_special_tokens=True)
                    r["N"]=sum(x not in special for x in ids)
                    r["correct"]=bool(scoring.score(text,task)["correct"])
                    if not enabled:
                        check(text==h["output_text"] and all(r[k]==h[k] for k in (*BILL,"output_ids","correct","N")),
                              "Exact fixed90_fused replay failed")
                    runs[enabled]=r
                row={k:task[k] for k in ("index","dataset","task_id","seed")}
                row.update(schema="warm-entropy-pair-v1",q=128,gpu=A1,source=source,samples_sha256=SAMPLES_SHA,
                           ref_sha256=sha(src),first=first_difference(runs[False],runs[True]),
                           input_sha256=input_digest(data),scorer_sha256=scorer)
                for enabled,k in ((False,"off"),(True,"on")):
                    row[k]={v:runs[enabled][v] for v in (*BILL,"N","correct","output_ids","phases")}
                write(dst,row)
            if j<3:
                positive+=int(row["first"] is not None)
            if j==2:
                check(ORDER[:3]==list(QUAL) and positive,"No actionable warm admission in anchors")
                print("QUALIFIED",positive,"/ 3",flush=True)
            if j<3 or (j+1)%10==0:
                print("PAIRS",j+1,i,row["off"]["F"],row["on"]["F"],flush=True)


if __name__=="__main__":
    main()

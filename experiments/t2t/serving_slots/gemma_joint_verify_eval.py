"""Frozen fixed-Q512 matched terminal study. Bind explicitly to A800 physical 7."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from .layout_sweep import write

GPU="GPU-f7fcacba-741b-a1ea-b370-4780cd948f6d"
QUAL=(0,10,20)
ORDER=[j+k for j in range(10) for k in QUAL]+list(range(30,90))
BILL=("F","decoder_F","decoder_Q","encoder_F","encoder_Q","max_decoder_Q","joint_F","route_Q","route_F")
KEYS=("index","dataset","task_id","seed","input_sha256","samples_sha256","scorer_sha256","torch","N","correct")


def check(ok,msg):
    if not ok:
        raise ValueError(msg)


def digest(x):
    return hashlib.sha256(json.dumps(x,separators=(",",":")).encode()).hexdigest()


def path(root,i,on):
    return root/f"C512-joint{int(on)}-{i:02}.json"


def pair(root,history,i):
    a,b=[json.loads(path(root,i,v).read_text()) for v in (False,True)]
    h=json.loads((history/f"C512-promotion1-{i:02}.json").read_text())
    check(h["schema"]=="full-promotion-sample-v1" and h["promotion"] and h["gate"]=="half" and h["fuse"]
          and h["budget"]==512 and h["prefix"]==[1,8],"Wrong promotion reference")
    bad=[k for k in (*KEYS,*BILL) if a[k]!=h[k]]
    check(not bad and a["ids"]==digest(h["output_ids"]),f"#{i} off differs from promotion1: {bad} or IDs")
    for v,r in ((False,a),(True,b)):
        check(r["schema"]=="joint-verify-v1" and r["on"] is v and r["index"]==i and r["gpu"]==GPU
              and r["projections"]==85 and r["layers"]==30 and r["max_decoder_Q"]<=512
              and r["decoder_Q"]==sum(r["route_Q"].values()) and r["decoder_F"]==sum(r["route_F"].values())
              and r["encoder_Q"]==256*r["encoder_F"] and r["F"]==r["decoder_F"]+r["encoder_F"]
              and type(r["correct"]) is bool,"Wrong case or physical bill")
    check(all(a[k]==b[k] for k in (*KEYS[:8],"source","gpu","checkpoint","transformers")),"Unmatched pair")
    x,y=a["hits"],b["hits"]
    check(bool(x)==bool(y),"Exposure diverged before first intervention")
    if x:
        check(all(x[0][k]==y[0][k] for k in ("prefix","state","ordinal","tick","head_dirty","tail_dirty",
                                              "drain_rows","verify_rows","joint_rows")),"Unaligned first position/state")
    else:
        check(all(a[k]==b[k] for k in (*KEYS,*BILL,"ids")),"Nonidentical unexposed pair")
    check(b["route_F"].get("verify_joint_paid",0)==len(y),"Unbilled intervention")
    return a,b


def available(own):
    check(os.environ.get("CUDA_VISIBLE_DEVICES")==GPU,"Only GPU7")
    rows=subprocess.check_output(["nvidia-smi","-i","7","--query-compute-apps=pid,gpu_uuid",
                                  "--format=csv,noheader,nounits"],text=True).splitlines()
    check(all(own and v.split(", ")==[str(os.getpid()),GPU] for v in rows),"GPU7 occupied")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("root","history","samples","checkpoint"):
        ap.add_argument("--"+k,type=Path,required=True)
    ap.add_argument("--deadline",type=int,required=True)
    args=ap.parse_args()
    available(False)
    import torch
    import transformers
    from transformers import AutoProcessor,DiffusionGemmaForBlockDiffusion
    from transformers.models.diffusion_gemma import generation_diffusion_gemma as native
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.evaluation.frozen import sha
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.methods.t2t.budget_family import Law
    from .mask_pair_gemma import SAMPLES_SHA,input_digest
    from .gemma_joint_verify import alignment,make_auditor
    check(time.time()<args.deadline and torch.cuda.device_count()==1 and transformers.__version__=="5.12.0"
          and sha(Path(native.__file__))=="cded57b7e365f0265b1c088e993c358b5f70acdf48055fce396540b4d9d691bc",
          "Wrong implementation or past deadline")
    tasks=load_tasks(args.samples)
    check(sha(args.samples)==SAMPLES_SHA,"Wrong frozen90")
    src=Path(__file__).parent
    source={n:sha(src/(n+".py")) for n in ("gemma_joint_verify_eval","gemma_joint_verify","gemma_fixed_dual",
            "gemma_fixed_dual_eval","gemma_strict_sampler","gemma_wave_kv","gemma_strict_rows","gemma_unified_audit")}
    args.root.mkdir(parents=True,exist_ok=True)
    score_sha=sha(Path(scoring.__file__))
    audit=make_auditor()
    proc=AutoProcessor.from_pretrained(args.checkpoint,local_files_only=True)
    available(False)
    model=DiffusionGemmaForBlockDiffusion.from_pretrained(args.checkpoint,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation="sdpa",experts_implementation="grouped_mm",
        device_map={"":"cuda:0"}).eval()
    eos=model.generation_config.eos_token_id
    special=set(proc.tokenizer.all_special_ids)|set([eos] if isinstance(eos,int) else eos or [])
    for j,i in enumerate(ORDER):
        t=tasks[i]
        for on in ((False,True) if j%2==0 else (True,False)):
            dst=path(args.root,i,on)
            if dst.exists():
                check(json.loads(dst.read_text())["source"]==source,"Cannot resume changed source")
                continue
            available(True)
            check(time.time()<args.deadline,"Deadline")
            print("START",dst.stem,flush=True)
            d=proc.apply_chat_template([{"role":"user","content":t["prompt"]}],tokenize=True,
                add_generation_prompt=True,return_dict=True,return_tensors="pt").to(model.device)
            r=audit(model,d,q=512,seed=t["seed"],law=Law(),ablation="00",policy={"joint_verify":on})
            hits=alignment(r["trace"],on,d["input_ids"].shape[1])
            ids=r["output_ids"]
            row={k:t[k] for k in ("index","task_id","seed","dataset")}
            row.update({k:r[k] for k in BILL})
            row.update(schema="joint-verify-v1",on=on,source=source,gpu=GPU,checkpoint=str(args.checkpoint),
                input_sha256=input_digest(d),samples_sha256=SAMPLES_SHA,scorer_sha256=score_sha,
                projections=r["projection_modules"],layers=len(model.model.decoder.layers),hits=hits,
                ids=digest(ids),correct=bool(scoring.score(proc.tokenizer.decode(ids,skip_special_tokens=True),t)["correct"]),
                N=sum(x not in special for x in ids),torch=str(torch.__version__),transformers=transformers.__version__)
            write(dst,row)
            print("DONE",dst.stem,len(hits),r["F"],r["decoder_Q"],flush=True)
        pair(args.root,args.history,i)
        if j+1==len(QUAL):
            n=sum(len(pair(args.root,args.history,k)[1]["hits"]) for k in QUAL)
            check(n>0,"No joint-close exposure; do not expand")
            write(args.root/"qualification.json",dict(indices=QUAL,hits=n,source=source))
            print("QUALIFIED",n,flush=True)


if __name__=="__main__":
    main()

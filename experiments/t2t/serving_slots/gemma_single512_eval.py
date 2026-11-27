"""A800-7 only: qualification first; never expand on zero actionable positives."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

from .gemma_joint_verify_eval import BILL, GPU, KEYS, ORDER, QUAL, available, check, digest
from .layout_sweep import write


def path(root,i,on):
    return root/f"C512-single{int(on)}-{i:02}.json"


def pair(root,history,i):
    a,b = [json.loads(path(root,i,on).read_text()) for on in (False,True)]
    h = json.loads((history/f"C512-joint1-{i:02}.json").read_text())
    check(h["schema"]=="joint-verify-v1" and h["on"] and h["index"]==i and h["gpu"]==GPU,"Wrong reference")
    for k in (*KEYS,*BILL,"ids","checkpoint","transformers","layers","projections"):
        check(a[k]==h[k],f"#{i} joint1 drift: {k}")
    check(a["joint_hits"]==h["hits"],"Joint1 close states/rows differ")
    for on,r in ((False,a),(True,b)):
        m=r["stop"]
        check(r["schema"]=="single512-v1" and r["on"] is on and r["index"]==i and r["gpu"]==GPU
              and r["projections"]==85 and r["layers"]==30 and r["max_decoder_Q"]<=512
              and r["F"]==r["decoder_F"]+r["encoder_F"] and r["encoder_Q"]==256*r["encoder_F"]
              and r["decoder_F"]==sum(r["route_F"].values())==m["decoder_ticks"]
              and r["decoder_Q"]==sum(r["route_Q"].values()) and r["route_F"].get("verify_joint_paid",0)==len(r["joint_hits"])
              and m["qualified"]+sum(m["resets"].values())==m["decoder_ticks"]
              and m["positives"]<=m["pairs"]<=m["qualified"] and type(r["correct"]) is bool,
              "Unreconciled physical/stop bill")
    check(all(a[k]==b[k] for k in (*KEYS[:8],"source","checkpoint","transformers")),"Unmatched pair")
    def first(r):
        return next((x for x in r["stop"]["hits"] if x["actionable"]),None)
    x,y=first(a),first(b)
    check(bool(x)==bool(y),"Unaligned first actionable hit")
    if x:
        check(all(x[k]==y[k] for k in x if k!="next_route"),"Different state at first actionable hit")
    else:
        check(all(a[k]==b[k] for k in ("N","ids","correct",*BILL,"joint_hits")) and
              a["stop"]["trajectory"]==b["stop"]["trajectory"],"No actionable hit but trajectory changed")
    return a,b


def seal_ledger(dst,r):
    raw=json.dumps({k:r[k] for k in ("trace","physical_trace")},separators=(",",":")).encode()
    buf=gzip.compress(raw,compresslevel=3,mtime=0)
    p=dst.with_suffix(".ledger.json.gz")
    tmp=p.with_suffix(".tmp")
    tmp.write_bytes(buf)
    tmp.replace(p)
    return hashlib.sha256(buf).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("root","history","samples","checkpoint"):
        ap.add_argument("--"+k,type=Path,required=True)
    ap.add_argument("--deadline",type=int,required=True)
    args=ap.parse_args()
    available(False)
    import torch
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from transformers.models.diffusion_gemma import generation_diffusion_gemma as native
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.evaluation.frozen import sha
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from .gemma_joint_verify import alignment
    from .gemma_single512_stop import inspect, make_auditor
    from .mask_pair_gemma import SAMPLES_SHA, input_digest
    check(time.time()<args.deadline and torch.cuda.device_count()==1 and transformers.__version__=="5.12.0"
          and sha(Path(native.__file__))=="cded57b7e365f0265b1c088e993c358b5f70acdf48055fce396540b4d9d691bc"
          and sha(args.samples)==SAMPLES_SHA and ORDER[:3]==list(QUAL),"Wrong pinned configuration")
    tasks=load_tasks(args.samples)
    source={k:sha(Path(__file__).with_name(k+".py")) for k in ("gemma_single512_stop","gemma_single512_eval",
        "gemma_fixed_dual","gemma_fixed_dual_eval","gemma_fullstop","gemma_joint_verify","gemma_joint_verify_eval",
        "gemma_strict_sampler","gemma_strict_rows","gemma_wave_kv","gemma_unified_audit","gemma_strict_request")}
    args.root.mkdir(parents=True,exist_ok=True)
    audit=make_auditor()
    proc=AutoProcessor.from_pretrained(args.checkpoint,local_files_only=True)
    available(False)
    model=DiffusionGemmaForBlockDiffusion.from_pretrained(args.checkpoint,local_files_only=True,
          dtype=torch.bfloat16,attn_implementation="sdpa",experts_implementation="grouped_mm",device_map={"":"cuda:0"}).eval()
    eos=model.generation_config.eos_token_id
    special=set(proc.tokenizer.all_special_ids)|set([eos] if isinstance(eos,int) else eos or [])
    for j,i in enumerate(ORDER):
        t=tasks[i]
        for on in ((False,True) if j%2==0 else (True,False)):
            dst=path(args.root,i,on)
            if dst.exists():
                old=json.loads(dst.read_text())
                check(old["source"]==source and sha(dst.with_suffix(".ledger.json.gz"))==old["ledger_sha256"],
                      "Cannot resume mixed or unsealed run")
                continue
            available(True)
            check(time.time()<args.deadline,"Deadline")
            print("START",dst.stem,flush=True)
            data=proc.apply_chat_template([{"role":"user","content":t["prompt"]}],tokenize=True,
                      add_generation_prompt=True,return_dict=True,return_tensors="pt").to(model.device)
            r=audit(model,data,q=512,seed=t["seed"],ablation="00",policy={"single512_stop":on})
            origin=data["input_ids"].shape[1]
            m=inspect(r["trace"],origin,on)
            ids=r["output_ids"]
            row={k:t[k] for k in ("index","dataset","task_id","seed")}
            row.update({k:r[k] for k in BILL})
            row.update(schema="single512-v1",on=on,source=source,gpu=GPU,stop=m,
                joint_hits=alignment(r["trace"],True,origin),ids=digest(ids),checkpoint=str(args.checkpoint),
                input_sha256=input_digest(data),scorer_sha256=sha(Path(scoring.__file__)),samples_sha256=SAMPLES_SHA,
                torch=str(torch.__version__),transformers=transformers.__version__,projections=r["projection_modules"],
                layers=len(model.model.decoder.layers),correct=bool(scoring.score(proc.tokenizer.decode(ids,skip_special_tokens=True),t)["correct"]),
                N=sum(x not in special for x in ids),ledger_sha256=seal_ledger(dst,r))
            write(dst,row)
            print("DONE",dst.stem,m["qualified"],m["pairs"],m["positives"],m["actionable"],r["F"],flush=True)
        pair(args.root,args.history,i)
        if j==2:
            q=[pair(args.root,args.history,k)[1]["stop"] for k in QUAL]
            n=sum(v["actionable"] for v in q)
            write(args.root/"qualification.json",dict(indices=QUAL,source=source,passed=n>0,actionable=n,
                  windows=sum(v["pairs"] for v in q),positives=sum(v["positives"] for v in q)))
            print("QUALIFIED" if n else "NO_ACTIONABLE_STOP",n,flush=True)
            if not n:
                return


if __name__=="__main__":
    main()

"""Guarded A1-only pairs against the complete historical verified-EOS Q32 row."""
import os
import time
from pathlib import Path
from importlib.metadata import version
from unittest.mock import patch

from .gemma_cold_frontier_last_eval import board, CKPT, SAMPLES, SCORER, DEADLINE, GPU, read, sha
from .gemma_joint_verify_eval import ORDER, check, digest
from .gemma_warm_fewer_fail import arm_class, align
from .layout_sweep import write

RT=Path("/gs/runtime/warm-fewer-fail-q32-a1-1128")
ROOT=Path("/gs/results/warm-fewer-fail-q32-a1-1128")
HIST=Path("/gs/results/verified-eos-1128")


def main():
    board(False)
    import torch
    from transformers import AutoProcessor,DiffusionGemmaForBlockDiffusion
    from transformers.models.diffusion_gemma import generation_diffusion_gemma as native
    from . import gemma_verified_eos as backend,gemma_verified_eos_eval as runner
    from .mask_pair_gemma import SAMPLES_SHA,input_digest
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.evaluation.frozen90 import load_tasks
    pinned=Path(runner.__file__).resolve().parent
    refs=[sha(HIST/f"C32-pair-{i:02}.json") for i in range(90)]
    base=read(HIST/"C32-pair-00.json")["source"]
    mp=pinned.parents[2]/"benchmarks/frozen90/manifest.json"
    tasks=load_tasks(SAMPLES)
    reg=dict(pid=os.getpid(),gpu=GPU,deadline=DEADLINE,checkpoint=str(CKPT),
             boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
             start_ticks=Path("/proc/self/stat").read_text().split(") ")[1].split()[19],
             policy=sha(RT/"registration.md"),launcher=sha(RT/"launch.sh"),
             sources={n:sha(Path(__file__).with_name(n+".py")) for n in
                      ("gemma_warm_fewer_fail","gemma_warm_fewer_fail_eval")},
             base=base,manifest=sha(mp),inputs=sha(SAMPLES),scorer=sha(Path(scoring.__file__)),
             refs=digest(refs),torch=str(torch.__version__),tf=version("transformers"),native=sha(Path(native.__file__)))
    check(len(tasks)==90 and sorted(ORDER)==list(range(90)) and ORDER[:3]==[0,10,20] and
          all(sha(pinned/(n+".py"))==v for n,v in base.items()) and
          reg["inputs"]==read(mp)["samples_sha256"]==SAMPLES_SHA and reg["scorer"]==SCORER and
          reg["native"]=="cded57b7e365f0265b1c088e993c358b5f70acdf48055fce396540b4d9d691bc" and
          reg["torch"]=="2.6.0+cu124" and reg["tf"]=="5.12.0" and version("math-verify")=="0.8.0"
          and torch.cuda.device_count()==1,"Unregistered environment")
    board(True)
    ROOT.mkdir()
    write(ROOT/"registration.json",reg)
    print("REGISTERED",reg["pid"],flush=True)
    proc=AutoProcessor.from_pretrained(CKPT,local_files_only=True)
    board(True)
    model=DiffusionGemmaForBlockDiffusion.from_pretrained(CKPT,local_files_only=True,dtype=torch.bfloat16,
          attn_implementation="sdpa",experts_implementation="grouped_mm",device_map={"":"cuda:0"}).eval()
    special=set(proc.tokenizer.all_special_ids)|set(model.generation_config.eos_token_id)
    elapsed=[]
    exposure=0
    for j,i in enumerate(ORDER):
        if j and DEADLINE-time.time()<max(8,1.5*max(elapsed[-6:])+4):
            write(ROOT/"partial.json",dict(pairs=j,reason="deadline guard",reg=digest(reg)))
            return
        st=time.monotonic()
        task=tasks[i]
        path=HIST/f"C32-pair-{i:02}.json"
        ref=read(path)
        check(sha(path)==refs[i] and ref["q"]==32 and ref["source"]==base and ref["scorer_sha256"]==SCORER
              and ref["samples_sha256"]==SAMPLES_SHA and all(ref[k]==task[k] for k in ("index","task_id","dataset","seed")),
              "Reference mismatch")
        data=proc.apply_chat_template([{"role":"user","content":task["prompt"]}],tokenize=True,
                add_generation_prompt=True,return_dict=True,return_tensors="pt").to(model.device)
        check(input_digest(data)==ref["on"]["input_sha256"],"Input mismatch")
        r={k:task[k] for k in ("index","task_id","dataset","seed")}
        r.update(ref=refs[i],reg=digest(reg),pid=os.getpid(),events={})
        for a in (("off","on") if not j%2 else ("on","off")):
            board(True)
            tape=[]
            cls=arm_class(backend.VerifiedEOSRequest,a=="on",tape)
            with patch.object(backend,"VerifiedEOSRequest",cls),torch.inference_mode():
                x=runner.run(model,data,task,32,True)
            x.update(N=sum(k not in special for k in x["output_ids"]),correct=bool(scoring.score(
                proc.tokenizer.decode(x["output_ids"],skip_special_tokens=True),task)["correct"]))
            if a=="off":
                check(x==ref["on"],f"Inexact off #{i}")
            r[a],r["events"][a]=x,tape
        first=align(r["off"],r["on"],r["events"]["off"],r["events"]["on"])
        r["first"]=first
        write(ROOT/f"pair-{i:02}.json",r)
        exposure+=int(first is not None)
        elapsed.append(time.monotonic()-st)
        if j in (2,29,89):
            print("SEALED",j+1,"EXPOSED",exposure,flush=True)
            write(ROOT/f"checkpoint-{j+1}.json",dict(pairs=j+1,exposed=exposure,reg=digest(reg)))
        if j==2 and not exposure:
            write(ROOT/"no-go.json",dict(pairs=3,reason="zero anchor exposure",reg=digest(reg)))
            return
    write(ROOT/"complete.json",dict(pairs=90,exposed=exposure,reg=digest(reg)))


if __name__=="__main__":
    main()

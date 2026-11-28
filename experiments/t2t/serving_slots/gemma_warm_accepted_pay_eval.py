"""One physical A1 worker; six matched anchors, conditional frozen90."""
import os
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

from .gemma_cold_frontier_last_eval import GPU, CKPT, SAMPLES, SCORER, read, sha
from .gemma_joint_verify_eval import ORDER, check, digest
from .gemma_warm_accepted_pay import ANCHORS, arm_class, align, audit, gates
from .layout_sweep import write

DEADLINE=1795849200
RT=Path("/gs/runtime/warm-accepted-pay-q32-a1-1128")
ROOT=Path("/gs/results/warm-accepted-pay-q32-a1-1128")
HIST=Path("/gs/results/verified-eos-1128")
PIN=Path("/gs/results/gemma-first-tile-kv-q32-a1-1128/registration.json")
PLAN=list(ANCHORS)+[i for i in ORDER if i not in ANCHORS]


def board(own):
    check(time.time()<DEADLINE and os.environ.get("CUDA_VISIBLE_DEVICES")==GPU,"Deadline or wrong GPU")
    def query(flag):
        return subprocess.check_output(["nvidia-smi","-i","1",flag,"--format=csv,noheader,nounits"],text=True).strip()
    check(all(own and v==f"{os.getpid()}, {GPU}" for v in
          query("--query-compute-apps=pid,gpu_uuid").splitlines()),"Another compute process on A1")
    if not own:
        check(query("--query-gpu=uuid,memory.used")==f"{GPU}, 0","A1 not empty")


def main():
    board(False)
    import torch
    from transformers import AutoProcessor,DiffusionGemmaForBlockDiffusion
    from transformers.models.diffusion_gemma import generation_diffusion_gemma as native
    from . import gemma_verified_eos as backend,gemma_verified_eos_eval as runner
    from .mask_pair_gemma import SAMPLES_SHA,input_digest
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.evaluation.frozen90 import load_tasks
    pin=read(PIN)
    pinned=Path(runner.__file__).resolve().parent
    refs=[sha(HIST/f"C32-pair-{i:02}.json") for i in range(90)]
    tasks=load_tasks(SAMPLES)
    check(pin["gpu"]==GPU and sorted(PLAN)==list(range(90)) and len(tasks)==90 and
          [tasks[i]["dataset"] for i in ANCHORS]==["gsm8k","math500","humaneval"]*2 and
          pin["refs"]==digest(refs) and pin["checkpoint"]==str(CKPT) and
          all(sha(CKPT/k)==v for k,v in pin["metadata"].items()) and
          all(sha(pinned/(k+".py"))==v for k,v in pin["base"].items()) and
          pin["manifest"]==sha(pinned.parents[2]/"benchmarks/frozen90/manifest.json") and
          pin["inputs"]==sha(SAMPLES)==SAMPLES_SHA and pin["scorer"]==sha(Path(scoring.__file__))==SCORER and
          pin["native"]==sha(Path(native.__file__)) and pin["torch"]==str(torch.__version__) and
          pin["tf"]==version("transformers") and version("math-verify")=="0.8.0" and
          torch.cuda.device_count()==1,"Bad pinned model/data/runtime")
    here=Path(__file__).parent
    reg=dict(pid=os.getpid(),gpu=GPU,deadline=DEADLINE,boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
             start_ticks=Path("/proc/self/stat").read_text().split(") ")[1].split()[19],
             policy=sha(RT/"registration.md"),launcher=sha(RT/"launch.sh"),pin=sha(PIN),refs=digest(refs),
             sources={k:sha(here/(k+".py")) for k in ("gemma_warm_accepted_pay","gemma_warm_accepted_pay_eval")})
    board(True)
    ROOT.mkdir()
    write(ROOT/"registration.json",reg)
    print("REGISTERED",reg["pid"],flush=True)
    proc=AutoProcessor.from_pretrained(CKPT,local_files_only=True)
    board(True)
    model=DiffusionGemmaForBlockDiffusion.from_pretrained(CKPT,local_files_only=True,dtype=torch.bfloat16,
          attn_implementation="sdpa",experts_implementation="grouped_mm",device_map={"":"cuda:0"}).eval()
    special=set(proc.tokenizer.all_special_ids)|set(model.generation_config.eos_token_id)
    qual=[]
    durations=[]
    for j,i in enumerate(PLAN,1):
        if durations and DEADLINE-time.time()<max(10,1.5*max(durations[-6:])+5):
            write(ROOT/"partial.json",dict(pairs=j-1,reason="deadline",reg=digest(reg)))
            return
        st=time.monotonic()
        task=tasks[i]
        path=HIST/f"C32-pair-{i:02}.json"
        ref=read(path)
        check(sha(path)==refs[i] and ref["source"]==pin["base"] and ref["q"]==32 and
              ref["samples_sha256"]==SAMPLES_SHA and ref["scorer_sha256"]==SCORER and
              all(task[k]==ref[k] for k in ("index","task_id","seed","dataset")),"Wrong historical row")
        data=proc.apply_chat_template([{"role":"user","content":task["prompt"]}],tokenize=True,
             add_generation_prompt=True,return_dict=True,return_tensors="pt").to(model.device)
        check(input_digest(data)==ref["on"]["input_sha256"],"Input mismatch")
        r={k:task[k] for k in ("index","task_id","seed","dataset")}
        r.update(pid=os.getpid(),reg=digest(reg),ref=refs[i],ticks={},audit={})
        for a in (("off","on") if j%2 else ("on","off")):
            board(True)
            tape=[]
            cls=arm_class(backend.VerifiedEOSRequest,a=="on",tape)
            with patch.object(backend,"VerifiedEOSRequest",cls),torch.inference_mode():
                x=runner.run(model,data,task,32,True)
            x.update(N=sum(k not in special for k in x["output_ids"]),correct=bool(scoring.score(
                proc.tokenizer.decode(x["output_ids"],skip_special_tokens=True),task)["correct"]))
            if a=="off":
                check(x==ref["on"],f"Inexact off #{i}")
            r[a],r["ticks"][a],r["audit"][a]=x,tape,audit(x,tape,a=="on")
        r["first"]=align(r["off"],r["on"],r["ticks"]["off"],r["ticks"]["on"])
        write(ROOT/f"pair-{i:02}.json",r)
        durations.append(time.monotonic()-st)
        if j<=6:
            qual.append(r)
            print("ANCHOR",i,"first",r["first"] and r["first"]["tick"],flush=True)
        if j==6:
            g=gates(qual)
            write(ROOT/"qualification.json",dict(gates=g,indices=ANCHORS,reg=digest(reg)))
            if not all(g.values()):
                write(ROOT/"no-go.json",dict(pairs=j,reason="qualification",reg=digest(reg)))
                return
        if j in (33,90):
            print("PAIRS",j,flush=True)
            write(ROOT/f"checkpoint-{j}.json",dict(pairs=j,reg=digest(reg)))
    write(ROOT/"complete.json",dict(pairs=90,reg=digest(reg)))


if __name__=="__main__":
    main()

"""A1 only, fixed Q32, no restart or unqualified expansion."""
import os
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

from .gemma_cold_frontier_last_eval import board, CKPT, SAMPLES, SCORER, DEADLINE, GPU, read, sha
from .gemma_joint_verify_eval import ORDER, check, digest
from .gemma_first_tile_kv import arm_class, audit
from .layout_sweep import write

RT=Path("/gs/runtime/gemma-first-tile-kv-q32-a1-1128")
ROOT=Path("/gs/results/gemma-first-tile-kv-q32-a1-1128")
HIST=Path("/gs/results/verified-eos-1128")


def matched(t,u):
    check(all(t[0][k]==u[0][k] for k in ("canvas","rng","q","first_logits")) and len(t[0]["first_logits"])==64
          and t[0]["q"]==32 and t[0]["writes"]==t[1]["debt"]>0
          and u[0]["writes"]==u[1]["debt"]==0 and t[0]["sampled"]==32 and u[0]["sampled"]==0,
          "Missing matched action and actual next-step cost difference")
    for a,b in zip((v for v in t if v["first"]),(v for v in u if v["first"])):
        check(all(a[k]==b[k] for k in ("head","canvas","rng","q")),"Per-block seed or fresh canvas drift")


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
    refs={i:sha(HIST/f"C32-pair-{i:02}.json") for i in ORDER}
    base=read(HIST/"C32-pair-00.json")["source"]
    manifest=pinned.parents[2]/"benchmarks/frozen90/manifest.json"
    src=Path(__file__).parent
    reg=dict(pid=os.getpid(),gpu=GPU,deadline=DEADLINE,checkpoint=str(CKPT),
             boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
             start_ticks=Path("/proc/self/stat").read_text().split(") ")[1].split()[19],
             policy=sha(RT/"registration.md"),launcher=sha(RT/"launch.sh"),
             sources={n:sha(src/(n+".py")) for n in ("gemma_first_tile_kv","gemma_first_tile_kv_eval")},
             base=base,manifest=sha(manifest),inputs=sha(SAMPLES),scorer=sha(Path(scoring.__file__)),
             refs=digest([refs[i] for i in range(90)]),torch=str(torch.__version__),tf=version("transformers"),
             native=sha(Path(native.__file__)),metadata={p:sha(CKPT/p) for p in
                      ("model.safetensors.index.json","config.json","generation_config.json")})
    tasks=load_tasks(SAMPLES)
    check(len(tasks)==len(refs)==90 and ORDER[:3]==[0,10,20] and set(ORDER[:30])==set(range(30)) and
          all(sha(pinned/(n+".py"))==v for n,v in base.items()) and
          reg["inputs"]==read(manifest)["samples_sha256"]==SAMPLES_SHA and reg["scorer"]==SCORER and
          reg["torch"]=="2.6.0+cu124" and reg["tf"]=="5.12.0" and version("math-verify")=="0.8.0" and
          reg["native"]=="cded57b7e365f0265b1c088e993c358b5f70acdf48055fce396540b4d9d691bc" and
          torch.cuda.device_count()==1, "Wrong frozen source/inputs/device")
    board(True)
    ROOT.mkdir()
    write(ROOT/"registration.json",reg)
    print("REGISTERED",reg["pid"],reg["policy"],flush=True)
    proc=AutoProcessor.from_pretrained(CKPT,local_files_only=True)
    board(True)
    model=DiffusionGemmaForBlockDiffusion.from_pretrained(CKPT,local_files_only=True,dtype=torch.bfloat16,
          attn_implementation="sdpa",experts_implementation="grouped_mm",device_map={"":"cuda:0"}).eval()
    special=set(proc.tokenizer.all_special_ids)|set(model.generation_config.eos_token_id)
    for j,i in enumerate(ORDER,1):
        task=tasks[i]
        path=HIST/f"C32-pair-{i:02}.json"
        h=read(path)
        check(sha(path)==refs[i] and h["source"]==base and h["q"]==32 and h["scorer_sha256"]==SCORER
              and h["samples_sha256"]==SAMPLES_SHA and all(h[k]==task[k] for k in ("index","task_id","dataset","seed")),
              "Historical control identity mismatch")
        data=proc.apply_chat_template([{"role":"user","content":task["prompt"]}],tokenize=True,
                add_generation_prompt=True,return_dict=True,return_tensors="pt").to(model.device)
        check(input_digest(data)==h["on"]["input_sha256"],"Different model input")
        r={k:task[k] for k in ("index","task_id","dataset","seed")}
        r.update(ref=refs[i],reg=digest(reg),pid=os.getpid(),ticks={},audit={})
        for enabled in ((False,True) if j%2 else (True,False)):
            board(True)
            key="on" if enabled else "off"
            t=[]
            cls=arm_class(backend.VerifiedEOSRequest,enabled,t)
            with patch.object(backend,"VerifiedEOSRequest",cls),torch.inference_mode():
                x=runner.run(model,data,task,32,True)
            x.update(N=sum(i not in special for i in x["output_ids"]),correct=bool(scoring.score(
                proc.tokenizer.decode(x["output_ids"],skip_special_tokens=True),task)["correct"]))
            if not enabled:
                check(x==h["on"],f"#{i}: off differs from verified-EOS on")
            r[key],r["ticks"][key],r["audit"][key]=x,t,audit(x,t,enabled)
        matched(r["ticks"]["off"],r["ticks"]["on"])
        write(ROOT/f"pair-{i:02}.json",r)
        if j in (3,30,90):
            stamp=dict(n=j,reg=digest(reg),pid=os.getpid())
            write(ROOT/f"checkpoint-{j}.json",stamp)
            print("CHECKPOINT",j,flush=True)
    write(ROOT/"complete.json",stamp)


if __name__=="__main__":
    main()

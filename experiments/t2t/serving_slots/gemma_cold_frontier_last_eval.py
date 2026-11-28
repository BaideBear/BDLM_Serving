"""A1-only paired worker. Inference and physical auditing come from the pinned EOS runner."""
import json
import os
import subprocess
import time
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

from .gemma_cold_frontier_last import TIERS, arm_class, audit
from .gemma_joint_verify_eval import ORDER, check, digest
from .layout_sweep import write

GPU="GPU-861c9b5d-1e73-2550-f832-b73988bc64ff"
RT=Path("/gs/runtime/cold-frontier-last-a1-1128")
ROOT=Path("/gs/results/cold-frontier-last-a1-1128")
HIST=Path("/gs/results/verified-eos-1128")
SAMPLES=Path("/gs/code/samples.json")
CKPT=Path("/gs/models/diffusiongemma-cd40195")
DEADLINE=1795843443
SCORER="a72cec90206f69616b9de436d052f6df54cf075ba9abeac4a75a11e46587d692"


def read(p):
    return json.loads(p.read_bytes())


def sha(p):
    return sha256(p.read_bytes()).hexdigest()


def board(own):
    check(time.time()<DEADLINE and os.environ.get("CUDA_VISIBLE_DEVICES")==GPU,"Deadline or wrong A1")
    def smi(q):
        return subprocess.check_output(["nvidia-smi","-i","1",q,"--format=csv,noheader,nounits"],
                                       text=True).strip()
    check(all(own and v==f"{os.getpid()}, {GPU}" for v in
              smi("--query-compute-apps=pid,gpu_uuid").splitlines()),"A1 in use")
    if not own:
        check(smi("--query-gpu=uuid,memory.used")==f"{GPU}, 0","A1 not empty")


def main():
    board(False)
    from . import gemma_verified_eos as backend, gemma_verified_eos_eval as runner
    from .mask_pair_gemma import SAMPLES_SHA,input_digest
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from transformers.models.diffusion_gemma import generation_diffusion_gemma as native
    import torch
    pinned=Path(runner.__file__).resolve().parent
    raw={(q,j):HIST/f"C{q}-pair-{j:02}.json" for q in TIERS for j in ORDER}
    refs={k:sha(p) for k,p in raw.items()}
    base=read(raw[32,0])["source"]
    manifest=pinned.parents[2]/"benchmarks/frozen90/manifest.json"
    tasks=load_tasks(SAMPLES)
    check(len(refs)==180 and set(ORDER)==set(range(90)) and ORDER[:3]==[0,10,20] and
          set(ORDER[:30])==set(range(30)) and all(sha(pinned/(n+".py"))==h for n,h in base.items()) and
          sha(SAMPLES)==read(manifest)["samples_sha256"]==SAMPLES_SHA and len(tasks)==90 and
          sha(Path(scoring.__file__))==SCORER and str(torch.__version__)=="2.6.0+cu124" and
          version("transformers")=="5.12.0" and version("math-verify")=="0.8.0" and
          sha(Path(native.__file__))=="cded57b7e365f0265b1c088e993c358b5f70acdf48055fce396540b4d9d691bc" and
          torch.cuda.device_count()==1,"Wrong pinned inputs")
    files={n:sha(Path(__file__).with_name(n+".py")) for n in ("gemma_cold_frontier_last","gemma_cold_frontier_last_eval")}
    reg=dict(schema="cold-frontier-last-a1-v1",pid=os.getpid(),gpu=GPU,
             start_ticks=Path("/proc/self/stat").read_text().split(") ",1)[1].split()[19],
             boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),deadline=DEADLINE,
             source=files,base=base,inputs=SAMPLES_SHA,manifest=sha(manifest),scorer=SCORER,
             reference_digest=digest([refs[k] for k in sorted(refs)]),policy=sha(RT/"registration.md"),
             native=sha(Path(native.__file__)),checkpoint=str(CKPT),
             checkpoint_metadata={p.name:sha(p) for p in sorted(CKPT.glob("*.json")) if p.name in
                                  {"model.safetensors.index.json","config.json","generation_config.json"}})
    board(True)
    ROOT.mkdir()
    write(ROOT/"registration.json",reg)
    print("REGISTERED",reg["pid"],reg["policy"],flush=True)
    proc=AutoProcessor.from_pretrained(CKPT,local_files_only=True)
    board(True)
    model=DiffusionGemmaForBlockDiffusion.from_pretrained(CKPT,local_files_only=True,dtype=torch.bfloat16,
          attn_implementation="sdpa",experts_implementation="grouped_mm",device_map={"":"cuda:0"}).eval()
    special=set(proc.tokenizer.all_special_ids)|set(model.generation_config.eos_token_id)
    gate=digest(reg)
    for j,i in enumerate(ORDER,1):
        task=tasks[i]
        for q in TIERS:
            board(True)
            h=read(raw[q,i])
            check(sha(raw[q,i])==refs[q,i] and h["source"]==base and h["q"]==q and
                  h["samples_sha256"]==SAMPLES_SHA and h["scorer_sha256"]==SCORER and
                  all(h[k]==task[k] for k in ("index","seed","dataset","task_id")),"Bad reference")
            data=proc.apply_chat_template([{"role":"user","content":task["prompt"]}],tokenize=True,
                                          add_generation_prompt=True,return_dict=True,return_tensors="pt").to(model.device)
            check(input_digest(data)==h["on"]["input_sha256"],"Wrong input")
            r={k:task[k] for k in ("index","seed","dataset","task_id")}
            r.update(registration=gate,q=q,pid=os.getpid(),ref=refs[q,i],phases={},ticks={})
            for enabled in ((False,True) if (j+(q==64))%2 else (True,False)):
                board(True)
                key="on" if enabled else "off"
                t=[]
                cls=arm_class(backend.VerifiedEOSRequest,enabled,t)
                with patch.object(backend,"VerifiedEOSRequest",cls), torch.inference_mode():
                    x=runner.run(model,data,task,q,True)
                text=proc.tokenizer.decode(x["output_ids"],skip_special_tokens=True)
                x.update(N=sum(a not in special for a in x["output_ids"]),correct=bool(scoring.score(text,task)["correct"]))
                if not enabled:
                    check(x==h["on"],"A1 off failed full historical replay")
                r[key],r["ticks"][key],r["phases"][key]=x,t,audit(x,t,q,enabled)
            check(r["off"]["tape"][0]!=r["on"]["tape"][0] and
                  r["ticks"]["off"][0]["cold"]==list(range(q)) and
                  r["ticks"]["on"][0]["cold"]==list(range(q,2*q)), "No physical first-tile intervention")
            write(ROOT/f"C{q}-pair-{i:02}.json",r)
        if j in (3,30,90):
            stamp=dict(pairs_per_tier=j,tiers=TIERS,registration=gate)
            write(ROOT/f"checkpoint-{j}.json",stamp)
            print("CHECKPOINT",j,flush=True)
    write(ROOT/"complete.json",dict(**stamp,outcome="frozen90"))


if __name__=="__main__":
    main()

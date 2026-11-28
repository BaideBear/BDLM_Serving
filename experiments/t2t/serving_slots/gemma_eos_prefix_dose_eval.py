"""A7-only paired driver; delegate all inference, EOS and physical audits to the pinned runner."""
import json
import os
import subprocess
import time
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

from .gemma_eos_prefix_dose import arm_class, first_change
from .gemma_joint_verify_eval import GPU, ORDER, available, check, digest
from .layout_sweep import write

RT = Path("/gs/runtime/eos-prefix-dose-q64-1128")
ROOT = Path("/gs/results/eos-prefix-dose-q64-1128")
HIST = Path("/gs/results/verified-eos-1128")
SAMPLES = Path("/gs/code/samples.json")
CKPT = Path("/gs/models/diffusiongemma-cd40195")
SCORER = "a72cec90206f69616b9de436d052f6df54cf075ba9abeac4a75a11e46587d692"


def sha(p):
    return sha256(p.read_bytes()).hexdigest()


def guard(own):
    check(time.time()<int(os.environ["DOSE_END_EPOCH"]), "Hard cutoff")
    available(own)
    if not own:
        mem=subprocess.check_output(["nvidia-smi","-i","7","--query-gpu=uuid,memory.used",
                                     "--format=csv,noheader,nounits"],text=True).strip()
        check(mem==f"{GPU}, 0", "Not an empty A7")


def main():
    guard(False)
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.methods.t2t import budget_family
    from . import gemma_verified_eos as backend, gemma_verified_eos_eval as evaluator
    from .mask_pair_gemma import SAMPLES_SHA, input_digest
    import torch, transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from transformers.models.diffusion_gemma import generation_diffusion_gemma as native
    pinned=Path(evaluator.__file__).resolve().parent
    paths={i:HIST/f"C64-pair-{i:02}.json" for i in ORDER}
    href=json.loads(paths[0].read_bytes())
    base=href["source"]
    refs={i:sha(p) for i,p in paths.items()}
    p=Path("/gs/results/paid-close-1128/C64-fuse1-00.json")
    h=json.loads(p.read_bytes())
    check(len(refs)==90 and set(ORDER)==set(range(90)) and ORDER[:3]==[0,10,20]
          and set(ORDER[:30])==set(range(30)) and sha(p)==href["reference_sha256"]
          and h["prefix"]==[1,8] and h["fuse"] is True and
          all(sha(pinned/(n+".py"))==v for n,v in base.items()), "Unpinned control")
    check(sha(SAMPLES)==SAMPLES_SHA and sha(Path(scoring.__file__))==h["scorer_sha256"]==SCORER and
          str(torch.__version__)==h["torch"]=="2.6.0+cu124" and
          transformers.__version__==h["transformers"]=="5.12.0" and version("math-verify")=="0.8.0" and
          sha(Path(native.__file__))=="cded57b7e365f0265b1c088e993c358b5f70acdf48055fce396540b4d9d691bc" and
          torch.cuda.device_count()==1, "Runtime or scoring mismatch")
    tasks=load_tasks(SAMPLES)
    check(len(tasks)==90, "Wrong sample count")
    guard(True)
    ROOT.mkdir()
    src={n:sha(Path(__file__).with_name(n+".py")) for n in ("gemma_eos_prefix_dose","gemma_eos_prefix_dose_eval")}
    reg=dict(schema="eos-prefix-dose-q64-v1",base=base,source=src,denominators=[8,4],
             bank_law_sha256=sha(Path(budget_family.__file__)),gpu=GPU,pid=os.getpid(),
             reference_digest=digest([refs[i] for i in range(90)]),policy=sha(RT/"registration.md"),
             scorer=SCORER,samples=SAMPLES_SHA,deadline=int(os.environ["DOSE_END_EPOCH"]))
    write(ROOT/"registration.json",reg)
    print("REGISTERED",os.getpid(),reg["policy"],flush=True)
    proc=AutoProcessor.from_pretrained(CKPT,local_files_only=True)
    guard(True)
    model=DiffusionGemmaForBlockDiffusion.from_pretrained(CKPT,local_files_only=True,dtype=torch.bfloat16,
          attn_implementation="sdpa",experts_implementation="grouped_mm",device_map={"":"cuda:0"}).eval()
    special=set(proc.tokenizer.all_special_ids)|set(model.generation_config.eos_token_id)
    exposed=0
    for j,i in enumerate(ORDER,1):
        guard(True)
        task=tasks[i]
        ref=json.loads(paths[i].read_bytes())
        check(sha(paths[i])==refs[i] and ref["schema"]=="verified-eos-pair-v1" and ref["source"]==base
              and ref["gpu"]==GPU and ref["q"]==64 and ref["samples_sha256"]==SAMPLES_SHA and
              ref["scorer_sha256"]==SCORER and all(ref[k]==task[k] for k in ("index","seed","task_id","dataset")),
              "Wrong reference")
        data=proc.apply_chat_template([{"role":"user","content":task["prompt"]}],tokenize=True,
                         add_generation_prompt=True,return_dict=True,return_tensors="pt").to(model.device)
        check(input_digest(data)==ref["on"]["input_sha256"], "Input mismatch")
        row={k:task[k] for k in ("index","seed","task_id","dataset")}
        row.update(registration=digest(reg),reference_sha256=refs[i],rows={})
        for den,arm in (((8,"off"),(4,"on")) if j%2 else ((4,"on"),(8,"off"))):
            guard(True)
            notes=[]
            cls=arm_class(backend.VerifiedEOSRequest,den,notes)
            with patch.object(backend,"VerifiedEOSRequest",cls), torch.inference_mode():
                x=evaluator.run(model,data,task,64,True)
            text=proc.tokenizer.decode(x["output_ids"],skip_special_tokens=True)
            x.update(N=sum(t not in special for t in x["output_ids"]),correct=bool(scoring.score(text,task)["correct"]))
            if den==8:
                check(x==ref["on"], "Off does not exactly replay the historic EOS-on receipt")
            row[arm],row["rows"][arm]=x,notes
        row["first"]=first_change(row["off"],row["on"],[row["rows"][a] for a in ("off","on")])
        exposed+=row["first"] is not None
        write(ROOT/f"C64-pair-{i:02}.json",row)
        if j in (3,30,90):
            stamp=dict(pairs=j,exposed=exposed,registration=digest(reg))
            write(ROOT/f"checkpoint-{j}.json",stamp)
            print("CHECKPOINT",j,exposed,flush=True)
            if j==3 and not exposed:
                write(ROOT/"complete.json",dict(**stamp,outcome="no-exposure"))
                return
    write(ROOT/"complete.json",dict(**stamp,outcome="frozen90"))


if __name__=="__main__":
    main()

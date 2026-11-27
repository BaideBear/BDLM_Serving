"""Run only on the released A7, against the exact ordered-EOS Q64 control."""
import json
import os
from hashlib import sha256
from pathlib import Path
import subprocess
from unittest.mock import patch

from .gemma_joint_verify_eval import check, digest, GPU, ORDER, available
from .gemma_waiting_seat import alignment, arm_class, gates
from .layout_sweep import write


ROOT=Path("/gs/results/waiting-seat-q64-1128")
PRIOR=Path("/gs/results/eos-handle-1128-v2/complete.json")
HIST=Path("/gs/results/verified-eos-1128")
SAMPLES=Path("/gs/code/samples.json")
CKPT=Path("/gs/models/diffusiongemma-cd40195")
SCORER="a72cec90206f69616b9de436d052f6df54cf075ba9abeac4a75a11e46587d692"


def sha(p):
    return sha256(p.read_bytes()).hexdigest()


def released():
    check(PRIOR.is_file() and not Path("/proc/122082").exists(), "EOS qualification not released")
    available(False)
    mem=subprocess.check_output(["nvidia-smi","-i","7","--query-gpu=uuid,memory.used",
                                 "--format=csv,noheader,nounits"],text=True).strip()
    check(mem==f"{GPU}, 0", "A7 not physically empty")


def main():
    released()
    here=Path(__file__).parent
    source={k:sha(here/(k+".py")) for k in ("gemma_waiting_seat","gemma_waiting_seat_eval")}
    replay_path=Path("/gs/results/paid-close-1128/C64-fuse1-00.json")
    replay=json.loads(replay_path.read_bytes())
    refpaths=[HIST/f"C64-pair-{i:02}.json" for i in range(90)]
    refhashes=[sha(p) for p in refpaths]
    base=json.loads(refpaths[0].read_bytes())["source"]
    check(json.loads(refpaths[0].read_bytes())["reference_sha256"]==sha(replay_path), "Wrong source runtime history")
    check(all(sha(here/(n+".py"))==h for n,h in base.items()), "Runtime is not the fixed EOS control")
    # Empty-board accounting is meaningful before THIS process imports CUDA.
    released()
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from . import gemma_verified_eos as backend, gemma_verified_eos_eval as evaluator
    from .mask_pair_gemma import SAMPLES_SHA, input_digest
    import torch, transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from importlib.metadata import version
    from transformers.models.diffusion_gemma import generation_diffusion_gemma as native
    check(sha(SAMPLES)==SAMPLES_SHA and sha(Path(scoring.__file__))==SCORER==replay["scorer_sha256"] and
          transformers.__version__==replay["transformers"]=="5.12.0" and
          str(torch.__version__)==replay["torch"]=="2.6.0+cu124" and
          version("math-verify")=="0.8.0" and sha(Path(native.__file__))==
          "cded57b7e365f0265b1c088e993c358b5f70acdf48055fce396540b4d9d691bc",
          "Wrong model, samples, scorer or native implementation")
    tasks=load_tasks(SAMPLES)
    check(len(tasks)==90 and len(set(ORDER))==90 and not ROOT.exists(), "Refuse overwrite or mixed run")
    proc=AutoProcessor.from_pretrained(CKPT,local_files_only=True)
    check(torch.cuda.device_count()==1,"Wrong physical device visibility")
    available(True)
    ROOT.mkdir()
    write(ROOT/"registration.json",dict(schema="waiting-seat-q64-v1",source=source,base=base,
          samples=SAMPLES_SHA,scorer=SCORER,reference_files_digest=digest(refhashes),
          policy_sha256=sha(Path("registration.md")),gpu=GPU,pid=os.getpid(),prior_complete_sha256=sha(PRIOR)))
    model=DiffusionGemmaForBlockDiffusion.from_pretrained(CKPT,local_files_only=True,dtype=torch.bfloat16,
          attn_implementation="sdpa",experts_implementation="grouped_mm",device_map={"":"cuda:0"}).eval()
    special=set(proc.tokenizer.all_special_ids)|set(model.generation_config.eos_token_id)
    anchors=[]
    for j,i in enumerate(ORDER):
        available(True)
        ref=json.loads(refpaths[i].read_bytes())
        task=tasks[i]
        check(sha(refpaths[i])==refhashes[i] and ref["schema"]=="verified-eos-pair-v1" and
              ref["source"]==base and ref["gpu"]==GPU and ref["q"]==64 and ref["samples_sha256"]==SAMPLES_SHA
              and ref["scorer_sha256"]==SCORER and all(ref[k]==task[k] for k in ("index","seed","task_id","dataset")),
              "Reference mismatch")
        data=proc.apply_chat_template([{"role":"user","content":task["prompt"]}],tokenize=True,
                         add_generation_prompt=True,return_dict=True,return_tensors="pt").to(model.device)
        check(input_digest(data)==ref["on"]["input_sha256"],"Input mismatch")
        runs,shadows={},[[],[]]
        for enabled in ((False,True) if j%2==0 else (True,False)):
            # The pinned auditor imports the class at call time. Scope the private
            # worker's factory override to this call; do not patch any shared file.
            cls=arm_class(backend.VerifiedEOSRequest,enabled,shadows[int(enabled)])
            with patch.object(backend,"VerifiedEOSRequest",cls), torch.inference_mode():
                x=evaluator.run(model,data,task,64,True)
            text=proc.tokenizer.decode(x["output_ids"],skip_special_tokens=True)
            result=scoring.score(text,task)
            x.update(N=sum(t not in special for t in x["output_ids"]),correct=bool(result["correct"]))
            if not enabled:
                check(x==ref["on"], "Off does not reproduce every reference field, including physical tape")
            runs[enabled]=x
        off,on=runs[False],runs[True]
        first=alignment(off,on,shadows)
        row={k:task[k] for k in ("index","seed","task_id","dataset")}
        row.update(schema="waiting-seat-pair-v1",source=source,reference_sha256=refhashes[i],first=first,
                   off=off,on=on,seat_tapes=shadows,gpu=GPU,scorer_sha256=SCORER)
        write(ROOT/f"C64-pair-{i:02}.json",row)
        if j<3:
            anchors.append(row)
        if j<3 or (j+1)%10==0:
            print("PAIR",j+1,i,"first",first and first["tick"],"F",off["F"],on["F"],
                  "correct",off["correct"],on["correct"],flush=True)
        if j==2:
            gate=gates(anchors)
            write(ROOT/"qualification.json",gate)
            if not all(gate.values()):
                write(ROOT/"complete.json",dict(outcome="no-go",pairs=3,gate=gate))
                return
    write(ROOT/"complete.json",dict(outcome="frozen90",pairs=90,gate=gates(anchors)))


if __name__=="__main__":
    main()

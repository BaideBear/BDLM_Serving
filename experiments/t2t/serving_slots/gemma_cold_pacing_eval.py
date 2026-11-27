"""One fixed GPU6 worker; only audit-passing question/width pairs can promote."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from .gemma_cold_pacing_audit import B, PHASES, QUAL, TIERS, replay, require
from .layout_sweep import write

GPU = "GPU-8c80a093-8254-7eb4-db21-6a1bbe2d5d83"
ORDER = [j+k for j in range(10) for k in QUAL] + list(range(30, 90))
CONTROL = ("index","dataset","task_id","seed","budget","input_sha256","samples_sha256",
           "scorer_sha256","N","correct","F","decoder_F","decoder_Q","encoder_F","encoder_Q",
           "max_decoder_Q","route_F","route_Q","torch","transformers")


def digest(x):
    return hashlib.sha256(json.dumps(x, separators=(",", ":")).encode()).hexdigest()


def history_path(root, q, i):
    if q == 128:
        return root/"profile-close-1128"/f"C128-profile-fuse1-{i:02}.json"
    return root/"paid-close-1128"/f"C{q}-fuse1-{i:02}.json"


def receipt(root, q, i, on):
    return root/f"C{q}-pace{int(on)}-{i:02}.json"


def checked_pair(root, history, q, i):
    a,b = [json.loads(receipt(root,q,i,x).read_text()) for x in (0,1)]
    h = json.loads(history_path(history,q,i).read_text())
    require(h["fuse"] is True and not h["slack"] and h["gate"] == "off" and
            h["prefix"] == [1,4 if q == 128 else 8] and h["joint_F"] == 0, "Wrong fixed history")
    drift = [k for k in CONTROL if h[k] != a[k]]
    require(not drift and a["output_digest"] == digest(h["output_ids"]),
            f"C{q}/{i} off-history mismatch: {drift} (also check token digest)")
    for on,r in ((False,a),(True,b)):
        require(r["schema"] == "cold-pacing-v1" and r["pace"] is on and r["index"] == i
                and r["budget"] == q and r["gpu"] == GPU and r["projection_modules"] == 85
                and type(r["correct"]) is bool and r["layers"] > 0 and r["max_decoder_Q"] <= q,
                "Wrong case or projection coverage")
        stats, blocks = r["phase"], r["blocks"]
        require(blocks and all(x["first_width"] == (q//2 if on else q) for x in blocks)
                and stats["first_Q"] == len(blocks)*B and stats["cold_Q"] == stats["first_Q"]+stats["cold_paid_Q"]
                and sum(stats[x+"_Q"] for x in PHASES) == r["decoder_Q"]
                and sum(stats[x+"_F"] for x in PHASES) == r["decoder_F"]
                and r["F"] == r["decoder_F"] + r["encoder_F"] and len(blocks) == r["encoder_F"]+1
                and stats["verify_Q"] == len(blocks)*B, "Incomplete block or phase bill")
    s = b["phase"]
    require(s["overlap_F"] > 0 and s["cap_binding_F"] >= len(b["blocks"])
            and not s["cold_drain_F"] and all(x["init_clock"] == 2*B//q for x in b["blocks"]),
            "No real overlap, pace exposure or guaranteed initialization")
    for k in (*CONTROL[:8],"source","checkpoint","layers","gpu","torch","transformers","prefix"):
        require(a[k] == b[k], "Unmatched pair: "+k)
    return a,b


def available(allow_self):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == GPU, "GPU6 UUID binding required")
    lines = subprocess.check_output(["nvidia-smi","-i","6","--query-compute-apps=pid,gpu_uuid",
                                     "--format=csv,noheader,nounits"], text=True).splitlines()
    require(all(allow_self and row.strip().split(", ") == [str(os.getpid()),GPU] for row in lines),
            "GPU6 has another compute process")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ("root","history","samples","checkpoint"):
        ap.add_argument("--"+flag,type=Path,required=True)
    ap.add_argument("--deadline",type=int,required=True)
    args = ap.parse_args()
    available(False)
    require(time.time()<args.deadline and len(ORDER)==len(set(ORDER))==90, "Deadline or manifest")
    import torch
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from transformers.models.diffusion_gemma import generation_diffusion_gemma as native
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.evaluation.frozen import sha
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.methods.t2t.budget_family import Law
    from .gemma_cold_pacing import make_auditor
    from .mask_pair_gemma import SAMPLES_SHA, input_digest

    require(torch.cuda.device_count()==1 and transformers.__version__=="5.12.0" and sha(Path(native.__file__))
            =="cded57b7e365f0265b1c088e993c358b5f70acdf48055fce396540b4d9d691bc",
            "Unregistered device or native implementation")
    tasks = load_tasks(args.samples)
    require(sha(args.samples)==SAMPLES_SHA, "Wrong frozen90")
    src = Path(__file__).parent
    source = {k:sha(src/(k+".py")) for k in ("gemma_cold_pacing","gemma_cold_pacing_audit",
              "gemma_cold_pacing_eval","gemma_fixed_dual","gemma_fixed_dual_eval","gemma_strict_rows",
              "gemma_strict_sampler","gemma_unified_audit","gemma_strict_request","gemma_wave_kv")}
    score_sha = sha(Path(scoring.__file__))
    args.root.mkdir(exist_ok=True,parents=True)
    audit = make_auditor()
    proc = AutoProcessor.from_pretrained(args.checkpoint,local_files_only=True)
    available(False)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(args.checkpoint,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation="sdpa",experts_implementation="grouped_mm",
        device_map={"":"cuda:0"}).eval()
    eos = model.generation_config.eos_token_id
    special = set(proc.tokenizer.all_special_ids)|set([eos] if isinstance(eos,int) else eos or [])
    for j,i in enumerate(ORDER):
        t = tasks[i]
        for q in TIERS:
            for on in ((False,True) if j%2==0 else (True,False)):
                dst = receipt(args.root,q,i,on)
                if dst.exists():
                    require(json.loads(dst.read_text())["source"] == source, "Cannot resume changed source")
                    continue
                available(True)
                require(time.time()<args.deadline, "Hard cutoff")
                print("START",dst.stem,flush=True)
                d = proc.apply_chat_template([{"role":"user","content":t["prompt"]}],tokenize=True,
                        add_generation_prompt=True,return_dict=True,return_tensors="pt").to(model.device)
                den = 4 if q == 128 else 8
                r = audit(model,d,q=q,seed=t["seed"],law=Law(1,den),ablation="00",policy={"pace":on})
                require(r["projection_modules"] == 85 and r["joint_F"] == r["over_offer_F"] == 0,
                        "Projection count, canvas count or offer mismatch")
                phase,blocks = replay(r["trace"],q,on)
                ids = r["output_ids"]
                row = {k:t[k] for k in ("index","dataset","task_id","seed")}
                row.update({k:r[k] for k in ("F","decoder_F","decoder_Q","encoder_F","encoder_Q","route_F",
                                             "route_Q","max_decoder_Q","projection_modules")})
                row.update(schema="cold-pacing-v1",pace=on,budget=q,prefix=[1,den],gpu=GPU,
                           layers=len(model.model.decoder.layers),source=source,torch=str(torch.__version__),
                           transformers=transformers.__version__,checkpoint=str(args.checkpoint),
                           scorer_sha256=score_sha,input_sha256=input_digest(d),samples_sha256=SAMPLES_SHA,
                           N=sum(x not in special for x in ids),output_digest=digest(ids),
                           correct=bool(scoring.score(proc.tokenizer.decode(ids,skip_special_tokens=True),t)["correct"]),
                           phase=phase,blocks=blocks)
                write(dst,row)
                print("DONE",dst.stem,row["F"],phase["overlap_F"],flush=True)
            checked_pair(args.root,args.history,q,i)
        if j+1 == len(QUAL):
            for k in QUAL:
                for q in TIERS:
                    checked_pair(args.root,args.history,q,k)
            write(args.root/"qualification.json",dict(indices=QUAL,tiers=TIERS,pairs=9,source=source))
            print("QUALIFIED 9/9; continuing with unseen indices",flush=True)


if __name__ == "__main__":
    main()

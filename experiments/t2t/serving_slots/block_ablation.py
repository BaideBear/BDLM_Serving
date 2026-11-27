"""Three-arm GPU experiment. One independent request per (sample, q, arm).

Run as a module inside the scoped GPU worker; partial files never count as cases.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[3]
TIERS = {"llada21": (64,16,32,8,4), "gemma": (512,128,256,64,32)}
SOURCE = {"llada21":["bdlm_serving/methods/t2t/llada_budget_request.py",
                      "experiments/t2t/serving_slots/llada_fixed_family_eval.py"],
          "gemma":["experiments/t2t/serving_slots/gemma_fixed_dual.py",
                   "experiments/t2t/serving_slots/gemma_fixed_dual_eval.py"]}
ORDER = (0,10,20,*(i for i in range(90) if i not in (0,10,20)))
ARMS = ("11","10","00")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def top_signature(row):
    trace = []
    for event in row["trace"]:
        pos = next((event[k] for k in ("positions","absolute_positions","rows") if k in event), None)
        if not pos or len(pos) != len(set(pos)) or len(pos) != event["q"]:
            raise RuntimeError("Missing physical row lists for 2B parity")
        trace.append((pos,event.get("phase"),event.get("encoded"),
                      event.get("written_absolute"),event.get("position_updates")))
    if not trace:
        raise RuntimeError("Missing 2B trace")
    return ([row[k] for k in ("output_ids","N","F","Q","correct","input_sha256")],
            row.get("failure"),trace)


def put(path, obj):
    data = json.dumps(obj, ensure_ascii=False, separators=(",",":"), allow_nan=False).encode()
    if path.suffix == ".gz":
        data = gzip.compress(data, compresslevel=1, mtime=0)
    tmp = path.with_suffix(path.suffix+".partial")
    with tmp.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.link(tmp, path)
    tmp.unlink()


def build(kind, samples):
    import torch
    if kind == "gemma":
        from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
        from . import fixed_family_eval as ev, gemma_fixed_dual_eval as adapter
        ev.check_snapshot(ev.CKPT)
        proc = AutoProcessor.from_pretrained(ev.CKPT, local_files_only=True)
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(ev.CKPT,local_files_only=True,
            dtype=torch.bfloat16,device_map={"":"cuda"},attn_implementation="sdpa",
            experts_implementation="grouped_mm").eval()
        def case(task,q,arm):
            return adapter.fixed(model,proc,task,q,ablation=arm)
    else:
        from types import SimpleNamespace
        from .spawn_frozen import load_gpu
        from .llada_mask_pilot import CKPT
        from . import llada_fixed_family_eval as ev
        from bdlm_serving.methods.t2t.budget_family import Law
        model,tok,_,ref = load_gpu(SimpleNamespace(samples=samples,checkpoint=CKPT),lambda _:None)
        def case(task,q,arm):
            prompt=json.loads((ref/f"{task['index']:02}-native32.json").read_bytes())["prompt_ids"]
            return ev.fixed(model,tok,task,prompt,q,Law(),ablation=arm)
    if any(p.device.type!="cuda" for p in model.parameters()):
        raise RuntimeError("CPU/offload is not a GPU study")
    return case


def run(kind, out, samples):
    import torch, transformers
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.evaluation import scoring
    if torch.cuda.device_count()!=1 or not torch.cuda.is_available():
        raise RuntimeError("Expected a protected single-GPU worker")
    expected={"llada21":"4.57.1","gemma":"5.12.0"}[kind]
    if transformers.__version__!=expected:
        raise RuntimeError(f"Expected transformers {expected}, got {transformers.__version__}")
    tasks=load_tasks(samples)
    hashes={p:sha(ROOT/p) for p in (*SOURCE[kind],"bdlm_serving/methods/t2t/budget_family.py")}
    if sha(samples)!="bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed" or \
       sha(Path(scoring.__file__))!="a72cec90206f69616b9de436d052f6df54cf075ba9abeac4a75a11e46587d692":
        raise RuntimeError("Frozen samples or scorer changed")
    out.mkdir(parents=True,exist_ok=False)
    meta=out/"run.json"
    cap = max(TIERS[kind])
    put(meta,dict(protocol="block-ablation-v2",model=kind,device="cuda",torch=str(torch.__version__),
        transformers=expected,arms=ARMS,tiers=TIERS[kind],order=ORDER,maximum_output=2048,
        samples=sha(samples),sources=hashes,scorer=sha(Path(scoring.__file__)),runner=sha(Path(__file__)),
        dense_top_all=True,gpu=os.environ["CUDA_VISIBLE_DEVICES"],physical_request_batch=1))
    stamp=sha(meta)
    print(f"Loading {kind} for 1350 independent CUDA requests",flush=True)
    case=build(kind,samples)
    summaries={}
    with torch.inference_mode():
        for i in ORDER:
            for q in TIERS[kind]:
                same=None
                for arm in ARMS:
                    if any(sha(ROOT/p)!=v for p,v in hashes.items()):
                        raise RuntimeError("Pinned study sources changed")
                    start=time.monotonic()
                    row=case(tasks[i],q,arm)
                    torch.cuda.synchronize()
                    required=("index","dataset","task_id","seed")
                    if (any(row[k]!=tasks[i][k] for k in required) or row["q"]!=q
                        or row["F"]<=0 or row["Q"]<row["F"] or row["TPF"]!=row["N"]/row["F"]
                        or row["efficiency"]!=row["N"]/row["Q"] or
                        type(row["correct"])!=bool or row["correct"]!=bool(row["score"]["correct"])):
                        raise RuntimeError("Unpaired or unaccounted completion")
                    row.update(arm=arm,registration=stamp,elapsed=time.monotonic()-start)
                    name=f"{i:02}-q{q}-{arm}.json.gz"
                    if q==cap:
                        signature=top_signature(row)
                        if same is not None and signature!=same:
                            raise RuntimeError(f"Fixed 2B arms disagree on sample {i}: {arm}")
                        same=signature
                    put(out/name,row)
                    summaries[name]={k:row[k] for k in ("N","F","Q","correct","TPF","efficiency")}
                    print(f"case={i:02} q={q} arm={arm} N/F/Q={row['N']}/{row['F']}/{row['Q']} "
                          f"score={int(row['correct'])} second={row['elapsed']:.2f}",flush=True)
    put(out/"complete.json",dict(registration=stamp,rows=len(summaries),cases=summaries))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model",choices=TIERS)
    ap.add_argument("out",type=Path)
    ap.add_argument("samples",type=Path)
    a=ap.parse_args()
    run(a.model,a.out,a.samples)

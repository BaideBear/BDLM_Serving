"""Protected, fixed-offer qualification on a previously sealed B1 sample subset.

Usage: CUDA_VISIBLE_DEVICES='' python packed_pilot.py MODEL CARD NEW_OUTPUT
No mutation of old cases or running interpreters. No production designation
without output, work and selected-position parity on these cases.
"""
import gc
import gzip
import json
import os
from pathlib import Path
import sys


def run(kind, out, samples):
    import torch
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from experiments.t2t.serving_slots import block_ablation as old
    from experiments.t2t.serving_slots.packed_decode import run_fixed
    tasks = load_tasks(samples)
    indices = [1,9,12,4,5,7,10,20] if kind == "llada21" else [0,10,20,1]
    q = 64 if kind == "llada21" else 128
    arm = "00"
    refdir = samples.parent/f"block-ablation-{kind}-v1"
    refs = [json.loads(gzip.decompress((refdir/f"{i:02}-q{q}-{arm}.json.gz").read_bytes())) for i in indices]
    if any(r["seed"] != tasks[i]["seed"] or r["index"] != i or r["arm"] != arm or
           r["q"] != q for r,i in zip(refs,indices)):
        raise ValueError("wrong reference")
    out.mkdir(exist_ok=False)
    if kind == "gemma":
        from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
        from experiments.t2t.serving_slots import fixed_family_eval as ev
        ev.check_snapshot(ev.CKPT)
        proc = AutoProcessor.from_pretrained(ev.CKPT,local_files_only=True)
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(ev.CKPT,local_files_only=True,
            dtype=torch.bfloat16,device_map={"":"cuda"},attn_implementation="sdpa",
            experts_implementation="grouped_mm").eval()
        items = [dict(seed=tasks[i]["seed"],data=proc.apply_chat_template(
            [{"role":"user","content":tasks[i]["prompt"]}],tokenize=True,add_generation_prompt=True,
            return_dict=True,return_tensors="pt").to(model.device)) for i in indices]
    else:
        from types import SimpleNamespace
        from experiments.t2t.serving_slots.spawn_frozen import load_gpu
        from experiments.t2t.serving_slots.llada_mask_pilot import CKPT
        model,tok,_,native = load_gpu(SimpleNamespace(samples=samples,checkpoint=CKPT),lambda _:None)
        items = [dict(seed=tasks[i]["seed"],input_ids=torch.tensor(
            [json.loads((native/f"{i:02}-native32.json").read_bytes())["prompt_ids"]],device=model.device),
            mask_id=tok.convert_tokens_to_ids("<|mask|>"),eos_ids=[tok.eos_token_id]) for i in indices]
    # Warm the same path without including model load/kernel startup in timings.
    run_fixed(model,kind,items[:1],q=q,arm=arm,max_active=1)
    attempts = []
    for active in (1,2,4,8):
        if active > len(items):
            break
        gc.collect()
        torch.cuda.empty_cache()
        try:
            trial = run_fixed(model,kind,items,q=q,arm=arm,max_active=active)
        except torch.cuda.OutOfMemoryError:
            record = dict(active=active,oom=True,qualified=False)
            attempts.append(record)
            old.put(out/f"trial-{active}.json",record)
            print(record,flush=True)
            break  # Never retry a partially mutated KV bank.
        pairs = []
        for i,(row,ref) in enumerate(zip(trial["results"],refs)):
            trace = row["trace"]
            key = "rows" if kind=="llada21" else "positions"
            paid = [c for c in trial["calls"] if i in c["lanes"] and c["phase"]=="decoder"]
            enc = [c for c in trial["calls"] if i in c["lanes"] and c["phase"]=="encoder"][1:]
            pairs.append(dict(index=indices[i],ids=row["ids"]==ref["output_ids"],
                positions=[e[key] for e in trace]==[e[key] for e in ref["trace"]],
                F=len(paid)+len(enc)==ref["F"],
                useful_Q=sum(c["widths"][c["lanes"].index(i)] for c in paid+enc)==ref["Q"]))
        dec=[c for c in trial["calls"] if c["phase"]=="decoder"]
        total,free=torch.cuda.mem_get_info(model.device)[::-1]
        # Estimate external device use from whole-device free and process reserve.
        # Leave at least 8% or 4 GiB for subsequent longer contexts.
        peak_headroom=total-trial["peak_reserved"]-max(0,total-free-torch.cuda.memory_reserved(model.device))
        safe=peak_headroom>=max(4<<30,int(.08*total))
        qualified=all(all(p[k] for k in ("ids","positions","F","useful_Q")) for p in pairs) and safe
        record=dict(model=kind,arm=arm,q=q,active=active,target_rows=512,seconds=trial["seconds"],
                    requests=len(items),requests_s=len(items)/trial["seconds"],pairs=pairs,
                    peak_gib=trial["peak_reserved"]/(1<<30),peak_headroom_gib=peak_headroom/(1<<30),
                    max_rows=max(c["paid_rows"] for c in dec),mean_rows=sum(c["paid_rows"] for c in dec)/len(dec),
                    padding_rows=sum(c["paid_rows"]-c["used_rows"] for c in dec),safe=safe,
                    qualified=qualified,oom=False)
        old.put(out/f"trial-{active}.json",record)
        print(json.dumps(record),flush=True)
        attempts.append(record)
        del trial
        if not safe or (active==1 and not qualified):
            break
    choices=[r for r in attempts if r["qualified"]]
    best=max(choices,key=lambda r:r["requests_s"]) if choices else None
    old.put(out/"qualification.json",dict(model=kind,q=q,indices=indices,attempts=attempts,
        fastest_qualified_active=best["active"] if best else None))


if __name__=="__main__":
    here=Path(__file__).resolve()
    sys.path.insert(0,str(here.parents[3]))
    from experiments.t2t.serving_slots import block_ablation as entry, block_ablation_gpu as protected
    if len(sys.argv)!=4 or (sys.argv[3]!="worker" and
        not Path(sys.argv[3]).name.startswith("block-ablation-pack-pilot-")):
        raise ValueError("expected a new and distinct pack pilot directory")
    entry.run=run  # Process-private entry; the stock holder transfer remains in force.
    protected.SELF=here
    protected.main()

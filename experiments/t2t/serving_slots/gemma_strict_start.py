"""Frozen-input CPU test of B256 bidirectional context under incremental startup."""
import json
import os
from pathlib import Path

import torch

from bdlm_serving.evaluation.frozen import ROOT, sha
from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.runtime.diffusiongemma import STUDY, check_snapshot, observe
from .gemma_fixed_caps import DATA, SAMPLES_SHA
from .gemma_strict_project import CanvasProjector
from .gemma_unified import _with_rng
from .gemma_unified_audit import projectors
from .mask_pair_gemma import input_digest


def chunk_rows(q):
    if type(q) is not int or q not in (32, 64, 128, 256):
        raise ValueError("Unsupported startup offer")
    return [list(range(i,i+q)) for i in range(0,256,q)]


def compare(ref, measured):
    if measured.shape != ref.shape or len(ref.shape) != 3 or ref.shape[0] != 1:
        raise ValueError("Expected matched row logits")
    diff = (ref.float()-measured.float()).abs()
    if not diff.isfinite().all():
        raise ValueError("Nonfinite logit difference")
    return dict(rows=ref.shape[1], top1_same=int((ref.argmax(-1)==measured.argmax(-1)).sum()),
                max_abs=float(diff.max()), mean_abs=float(diff.mean()), exact=torch.equal(ref,measured))


@torch.inference_mode()
def probe(model, data, seed):
    origin = data["input_ids"].shape[1]
    cfg, kw = model._prepare_generation_config(None,max_new_tokens=2048)
    if kw or not (data["attention_mask"]==1).all() or any(
            k not in {"input_ids","attention_mask","mm_token_type_ids"} for k in data) or (
            "mm_token_type_ids" in data and data["mm_token_type_ids"].count_nonzero()):
        raise ValueError("Expected canonical text-only Gemma input")
    device = data["input_ids"].device
    cache = model._prepare_cache_for_generation(cfg,1,origin+2048-256)
    pos = torch.arange(origin,dtype=torch.int32,device=device)[None]
    ids, mapping = model._prepare_encoder_inputs(data["input_ids"],data["attention_mask"].bool(),
                                                pos,cache,True,256,1)
    cache=model.model.encoder(input_ids=ids,attention_mask=mapping,past_key_values=cache,
                              position_ids=pos).past_key_values
    sampler=model._prepare_sampler(cfg)
    canvas,_=_with_rng(torch.Generator(device=device).manual_seed(seed).get_state(),device,
                       lambda:sampler.initialize_canvas(1,device))
    masks=dict.fromkeys(model.config.text_config.layer_types)
    native=model(decoder_input_ids=canvas,decoder_attention_mask=masks,past_key_values=cache,
                 decoder_position_ids=torch.arange(origin,origin+256,dtype=torch.int32,device=device)[None],
                 self_conditioning_logits=None).logits
    with CanvasProjector(model,cache,origin=origin) as full:
        ref=full.project(canvas,range(256),256)
    reference = compare(native,ref)
    del native
    result={}
    for q in (32,64,128,256):
        passes=[]
        with CanvasProjector(model,cache,origin=origin) as lane:
            for sweep in range(2 if q!=256 else 1):
                chunks=[]
                for rows in chunk_rows(q):
                    logits=lane.project(canvas,rows,q)
                    if lane.bank.live != list(range(origin,origin+(256 if sweep else rows[-1]+1))):
                        raise ValueError("Unexpected set of visible decoder KV")
                    m=compare(ref[:,rows],logits)
                    chunks.append(dict(first=rows[0],last=rows[-1],visible=len(lane.bank.live),**m))
                passes.append(dict(sweep=sweep+1,cumulative_Q=(sweep+1)*256,
                    cumulative_F=(sweep+1)*len(chunks),top1_same=sum(r["top1_same"] for r in chunks),
                    rows=256,chunks=chunks))
        result[str(q)]=passes
    if not result["256"][0]["chunks"][0]["exact"] or cache.get_seq_length()!=origin:
        raise ValueError("Repeated full-bank anchor or encoder cache changed")
    return reference,result


def main():
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion, __version__
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_available() or (
            os.sched_getaffinity(0)!=set(range(32,36))):
        raise ValueError("Registered CPU lane only")
    cfg=json.loads((STUDY/"config.json").read_bytes())
    samples=DATA/"samples.json"
    if __version__!=cfg["transformers"] or sha(samples)!=SAMPLES_SHA:
        raise ValueError("Wrong runtime or inputs")
    out=DATA/"gemma-strict-start-cpu3-v1"
    out.mkdir(mode=0o700,exist_ok=False)
    checkpoint=ROOT.parent.parent/"models/diffusiongemma-cd40195"
    check_snapshot(checkpoint)
    proc=AutoProcessor.from_pretrained(checkpoint,local_files_only=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model=DiffusionGemmaForBlockDiffusion.from_pretrained(checkpoint,local_files_only=True,
            dtype=torch.bfloat16,device_map={"":"cpu"},attn_implementation=cfg["attention"],
            experts_implementation=cfg["experts"]).eval()
    records=[]
    for task in (load_tasks(samples)[i] for i in (0,10,20)):
        data=proc.apply_chat_template([{"role":"user","content":task["prompt"]}],tokenize=True,
                           add_generation_prompt=True,return_tensors="pt",return_dict=True)
        with observe(model) as events, projectors(model) as (modules,positions):
            anchor,results=probe(model,data,task["seed"])
        origin=data["input_ids"].shape[1]
        expected=[list(range(origin,origin+256)) for _ in range(2)]
        for q in (32,64,128,256):
            expected.extend([[origin+i for i in rs] for rs in chunk_rows(q)]*(2 if q!=256 else 1))
        widths=list(map(len,expected))
        if positions!=expected or events != [("encoder",origin)]+[("decoder",v) for v in widths] or (
                not modules or any(seq!=widths for seq in modules.values())):
            raise ValueError("Physical input, decoder or per-layer projection mismatch")
        r=dict(index=task["index"],seed=task["seed"],input_digest=input_digest(data),
               projected_modules=len(modules),native_vs_dense_bank=anchor,results=results)
        (out/f"{task['index']:02d}.json").write_text(json.dumps(r))
        print(r["index"],{q:[d["top1_same"] for d in a] for q,a in results.items()},flush=True)
        records.append(r)
    report=dict(protocol="frozen canvas; no sampling; cold ascending then optional warm ascending sweep",
                device="cpu",quality_evaluated=False,manifest_sha256=SAMPLES_SHA,
                reference="Q256 dense bank on same input",sources={n:sha(Path(__file__).with_name(n))
                for n in ("gemma_strict_start.py","gemma_strict_project.py","gemma_wave_kv.py")},
                records=records)
    (out/"complete.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()

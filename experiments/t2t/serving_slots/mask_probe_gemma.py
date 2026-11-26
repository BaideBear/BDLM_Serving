"""Check the direction of information flow in a real two-canvas decoder forward."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

from bdlm_serving.evaluation.frozen import ROOT
from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.runtime.diffusiongemma_wave import packing_mask


def compare(left, right):
    if left.shape != right.shape:
        raise ValueError("Different readout shapes")
    # Bound peak CPU memory for the full-vocabulary comparison.
    diffs = [float((x-y).abs().max()) for x, y in zip(left.split(16), right.split(16), strict=True)]
    if not all(v >= 0 and v < float("inf") for v in diffs):
        raise ValueError("Non-finite readout")
    return {"max_abs": max(diffs), "argmax_changed": int((left.argmax(-1) != right.argmax(-1)).sum()),
            "rows": len(left)}


@torch.inference_mode()
def probe(checkpoint, samples, index, experts, expert_layout, parked=False):
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    native = json.loads((ROOT/"archive/diffusiongemma-native/config.json").read_text())
    task = load_tasks(samples)[index]
    proc = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    data = proc.apply_chat_template([{"role":"user","content":task["prompt"]}], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt")
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(checkpoint, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation=native["attention"], experts_implementation=experts,
        device_map={"":"cpu"}).eval()
    if parked:
        from bdlm_serving.evaluation.frozen import sha
        from experiments.t2t.serving_slots.gemma_fixed_caps import SAMPLES_SHA
        if (sha(samples) != SAMPLES_SHA or index != 0 or experts != native["experts"]
                or expert_layout != "packed"):
            raise ValueError("Use the registered index 0 parked-cache probe and native experts")
        return parked_probe(model,data,task)
    return measure(model,data,task,experts,expert_layout)


@torch.inference_mode()
def parked_probe(model,data,task):
    """Read final head logits on the identical first head-only step with a parked tail."""
    from bdlm_serving.runtime.diffusiongemma_wave import resolve_plan
    from experiments.t2t.serving_slots import gemma_unified as engine
    from bdlm_serving.evaluation.frozen import sha
    b = model.config.canvas_length
    plan = dict(resolve_plan(ROOT/"archive/diffusiongemma-wave/config.json"),cap=512,flowblock_spawn_threshold=0.0)
    offers = (256,512,32,64,128,256,512,32)
    begin = engine.WaveKV.begin
    origin = int(data["input_ids"].shape[1])
    if b != 256 or model.training or model.device.type != "cpu":
        raise ValueError("Expected B256 CPU evaluation model")

    class StopAfterLogits(Exception):
        pass

    def replay(blocked, perturb):
        tick=-1
        witness=[]
        out=None
        target=False
        def before(bank,positions,canvas_origin,device,causal=True):
            nonlocal tick,target
            tick+=1
            boundary=origin+b
            future=[p for p in bank.live if p>=boundary]
            target=bool(future) and max(positions)<boundary
            begin(bank,positions,canvas_origin,device,causal=blocked if target else causal)
            h=hashlib.sha256()
            if target:
                if len(future)!=b or len(bank.buffers)!=len(model.model.decoder.layers):
                    raise ValueError("Tail is not fully initialized")
                idx=torch.tensor([p-bank.base for p in future],dtype=torch.long,device=device)
                for layer in sorted(bank.buffers):
                    k,v=bank.buffers[layer]
                    for src in (k,v):
                        frozen=src.index_select(-2,idx).contiguous()
                        h.update(frozen.view(torch.uint8).cpu().numpy().tobytes())
                    if perturb:
                        # Only the cached value vectors change; keys and projected rows are untouched.
                        v.index_copy_(-2,idx,v.index_select(-2,idx)+1)
            witness.append((tick,list(positions),list(bank.live),h.hexdigest() if target else None))
            return bank
        def finish(_,args,result):
            nonlocal out
            if target:
                out=result.logits[0].detach().cpu().clone()
                if not torch.isfinite(out).all() or out.shape[0]!=len(witness[-1][1]):
                    raise ValueError("Wrong logits readout")
                raise StopAfterLogits
            logits=result.logits.contiguous().view(torch.uint8)
            witness[-1]+= (hashlib.sha256(logits.cpu().numpy().tobytes()).hexdigest(),)
        with patch.object(engine.WaveKV,"begin",before), model.register_forward_hook(finish):
            try:
                engine.generate(model,data,plan,next_budget=lambda state: offers[state.tick%len(offers)],
                                seed=task["seed"],causal=True)
            except StopAfterLogits:
                pass
        if out is None:
            raise ValueError("No resident tail on a head-only decoder step")
        return out,witness

    ref=None
    rows=[]
    for blocked in (True,False):
        a,w=replay(blocked,False)
        twin,tw=replay(blocked,False)
        changed,cw=replay(blocked,True)
        if ref is None: ref=w
        if w!=tw or w!=cw or w!=ref:
            raise ValueError("Changed pre-intervention logits, geometry or cached tail")
        same=compare(a,twin)
        delta=compare(a,changed)
        if same["max_abs"]!=0 or (delta["max_abs"]==0)!=blocked:
            raise ValueError(f"Failed parked-cache directional control: {blocked}, {same}, {delta}")
        rows.append(dict(causal=blocked,twin=same,perturbed=delta))
    tick,q,live,tail_sha=ref[-1]
    return dict(index=task["index"],task_id=task["task_id"],seed=task["seed"],device="cpu",dtype="bfloat16",
                tick=tick,offer=offers[tick%len(offers)],head_q=len(q),tail_q=0,cached_tail_rows=b,
                visible_rows=len(live),cached_tail_sha256=tail_sha,results=rows,quality_evaluated=False,
                test_only=dict(spawn_threshold=0,cap=512,offers=offers,perturbation="add 1 to parked V at every layer"),
                sources={p.name:sha(p) for p in (Path(__file__),Path(engine.__file__),Path(engine.__file__).with_name("gemma_wave_kv.py"))})


@torch.inference_mode()
def measure(model, data, task, experts="grouped_mm", expert_layout="packed"):
    native = json.loads((ROOT/"archive/diffusiongemma-native/config.json").read_text())
    device, b = data["input_ids"].device, model.config.canvas_length
    if device != model.device or model.training or expert_layout not in ("packed","per_canvas"):
        raise ValueError("Expected colocated evaluation model and a supported expert layout")
    if b != 256:
        raise ValueError("Expected B256 checkpoint")
    cfg, kw = model._prepare_generation_config(None, max_new_tokens=2048)
    prefix, mask = data["input_ids"], data["attention_mask"].bool()
    kw.update({k:v for k,v in data.items() if k not in ("input_ids","attention_mask")})
    plen = prefix.shape[1]
    cache = model._prepare_cache_for_generation(cfg, 1, plen+2048-b)
    pos = torch.arange(plen, dtype=torch.int32,device=device)[None]
    ids, mapping = model._prepare_encoder_inputs(prefix, mask, pos, cache, True, b, 1, **kw)
    cache = model.model.encoder(input_ids=ids, attention_mask=mapping, past_key_values=cache,
                                position_ids=pos, **kw).past_key_values
    kw.pop("mm_token_type_ids", None)
    sampler = model._prepare_sampler(cfg)
    with torch.random.fork_rng(devices=[] if device.type=="cpu" else [device]):
        if device.type=="cpu":
            torch.default_generator.manual_seed(task["seed"])
        else:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(task["seed"])
        base = torch.cat([sampler.initialize_canvas(1, device) for _ in range(2)], dim=1)
    pos = torch.arange(plen,plen+2*b,dtype=torch.int32,device=device)[None]
    blocked = packing_mask(cache, model.config.text_config.layer_types, b, 2, device)
    opened = {k:v.clone() for k,v in blocked.items()}
    for k,v in opened.items():
        v[:,:,:b,-b:] = True
        if int((v != blocked[k]).sum()) != b*b or not v.all():
            raise ValueError("The intervention must open exactly the forbidden quadrant")
    vs = model.config.text_config.vocab_size

    def forward(mapping, changed, fingerprint=False):
        x = base.clone()
        if changed is not None:
            x[0,changed] = (x[0,changed]+1) % vs
        # Same resident cache, logits shape, positions and weights. Do not sample.
        history = {}
        def capture(tag):
            def hook(_, args, output):
                value = output[0] if isinstance(output,tuple) else output
                h = value.reshape(2*b,-1)[:b].contiguous().view(torch.uint8)
                history[tag] = hashlib.sha256(h.cpu().numpy()).hexdigest()
            return hook
        def separate(fn):
            def apply(hidden, indices, weights):
                return torch.cat([fn(*args) for args in zip(hidden.split(b),indices.split(b),
                                                             weights.split(b),strict=True)])
            return apply
        with ExitStack() as stack:
            for i,layer in enumerate(model.model.decoder.layers):
                if expert_layout=="per_canvas":
                    stack.enter_context(patch.object(layer.experts,"forward",separate(layer.experts.forward)))
                if fingerprint:
                    for tag,module in (("attn",layer.self_attn),("experts",layer.experts),("output",layer)):
                        stack.callback(module.register_forward_hook(capture(f"{i}/{tag}")).remove)
            result = model(decoder_input_ids=x, self_conditioning_logits=None, past_key_values=cache,
                           decoder_attention_mask=mapping, decoder_position_ids=pos, **kw).logits[0]
        if result.shape != (2*b,vs):
            raise ValueError("Unexpected logit shape")
        # Keep paired baselines out of VRAM between calls in the GPU diagnostic.
        return result.cpu(),history

    rows, diagnostics = [], {}
    for label, mapping in (("block_causal",blocked),("bidirectional",opened)):
        ref, reference_history = forward(mapping,None,True)
        twin, history = forward(mapping,None,True)
        diagnostics[label] = dict(same_input=compare(ref,twin),
            first_same_input_layer_diff=next((k for k,v in history.items() if reference_history[k]!=v),None))
        del twin
        for source, index_changed, target, sl in (
            ("tail_first",b,"head",slice(0,b)),
            ("head_last",b-1,"tail",slice(b,2*b)),
            ("head_last",b-1,"earlier_head",slice(0,b-1)),
        ):
            if source == "tail_first" or target == "tail":
                other, history = forward(mapping,index_changed,target=="head")
                if target=="head":
                    diagnostics[label]["first_head_layer_diff"]=next(
                        (k for k,v in history.items() if reference_history[k]!=v), None)
            rows.append(dict(mask=label,changed=source,read=target,**compare(ref[sl],other[sl])))
        del ref, other
    return dict(model=native["checkpoint"],revision=native["revision"],seed=task["seed"],index=task["index"],
                device=str(device),dtype="bfloat16",attention=native["attention"],experts=experts,
                expert_layout=expert_layout,diagnostics=diagnostics,
                prompt_rows=plen,canvas=b,q=2*b,torch=str(torch.__version__),results=rows)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--samples",type=Path,required=True)
    p.add_argument("--index",type=int,default=0)
    p.add_argument("--experts",choices=("grouped_mm","eager"),default="grouped_mm")
    p.add_argument("--expert-layout",choices=("packed","per_canvas"),default="packed",
                   help="Diagnostic control: split only expert calls, keep two-canvas attention unchanged")
    p.add_argument("--parked",action="store_true",help="Probe cached-only tail influence on real sparse head logits")
    a=p.parse_args()
    report=probe(**vars(a))
    print(json.dumps(report,indent=2),flush=True)
    if a.parked:
        raise SystemExit(0)
    signals=[r["max_abs"] for r in report["results"]]
    if (any(v["same_input"]["max_abs"]!=0 for v in report["diagnostics"].values())
        or signals[0]!=0 or not all(s>0 for s in signals[1:])):
        raise SystemExit("Directional controls failed; inspect raw values before interpreting")

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
def probe(checkpoint, samples, index, experts, expert_layout):
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    native = json.loads((ROOT/"archive/diffusiongemma-native/config.json").read_text())
    task = load_tasks(samples)[index]
    proc = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    data = proc.apply_chat_template([{"role":"user","content":task["prompt"]}], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt")
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(checkpoint, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation=native["attention"], experts_implementation=experts,
        device_map={"":"cpu"}).eval()
    return measure(model,data,task,experts,expert_layout)


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
    a=p.parse_args()
    report=probe(**vars(a))
    print(json.dumps(report,indent=2),flush=True)
    signals=[r["max_abs"] for r in report["results"]]
    if (any(v["same_input"]["max_abs"]!=0 for v in report["diagnostics"].values())
        or signals[0]!=0 or not all(s>0 for s in signals[1:])):
        raise SystemExit("Directional controls failed; inspect raw values before interpreting")

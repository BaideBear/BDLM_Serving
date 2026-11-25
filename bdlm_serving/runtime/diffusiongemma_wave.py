"""FlowBlock and B3 request scheduling on the native DiffusionGemma sampler."""
from contextlib import nullcontext
from types import SimpleNamespace
import json
from pathlib import Path

import torch

from ..evaluation.frozen import ROOT, load_tasks
from ..methods.b3.gate import B3Gate, logits_entropy
from .diffusiongemma import check_snapshot, observe


def resolve_plan(path, size=None, windows=None):
    plan = json.loads(Path(path).read_text())
    b = plan["block_size"] if size is None else size
    ws = plan["flowblock_windows"] if windows is None else windows
    if type(b) is not int or b < 1 or type(plan["cap"]) is not int or plan["cap"] < b or plan["cap"] % b:
        raise ValueError("block_size must be a positive divisor of the generation cap")
    if not isinstance(ws, list) or not ws or any(type(w) is not int or not 1 <= w <= plan["cap"]//b for w in ws):
        raise ValueError("each FlowBlock window must fit in the configured output cap")
    if len(set(ws)) != len(ws) or plan["batch_size"] != 1 or not 0 <= plan["flowblock_spawn_threshold"] <= 1:
        raise ValueError("invalid window, batch size or spawn threshold")
    if type(plan["include_b3"]) is not bool:
        raise ValueError("include_b3 must be boolean")
    return dict(plan, block_size=b, flowblock_windows=ws,
                policies=[f"flow_w{w}" for w in ws] + (["b3"] if plan["include_b3"] else []))


def packing_mask(cache, layer_types, size, width, device):
    if width == 1:
        return dict.fromkeys(layer_types)
    slots = torch.arange(width*size, device=device) // size
    tail = slots[:, None] >= slots[None, :]
    return {kind: torch.nn.functional.pad(tail, (cache.layers[layer_types.index(kind)].keys.shape[-2], 0),
                                          value=True)[None, None] for kind in set(layer_types)}


def generate(model, data, policy, plan):
    b, cap = plan["block_size"], plan["cap"]
    if b != model.config.canvas_length:
        raise ValueError(f"block_size={b} disagrees with checkpoint canvas_length={model.config.canvas_length}")
    cfg, kw = model._prepare_generation_config(None, max_new_tokens=cap)
    prefix = data["input_ids"]
    device, plen = prefix.device, prefix.shape[1]
    mask = data["attention_mask"].bool()
    kw.update({k: v for k, v in data.items() if k not in ("input_ids", "attention_mask")})
    if prefix.shape[0] != 1 or not mask.all():
        raise ValueError("audited runner requires one unpadded text request")
    cache = model._prepare_cache_for_generation(cfg, 1, plen + cap - b)
    if cache.is_compileable:
        raise ValueError("requires native dynamic cache")
    def positions(start, length):
        return torch.arange(start, start+length, device=device, dtype=torch.int32)[None]
    def encode(first):
        nonlocal cache
        pos = positions(0, plen) if first else positions(prefix.shape[1]-b, b)
        ids, mapping = model._prepare_encoder_inputs(prefix, mask, pos, cache, first, b, 1, **kw)
        cache = model.model.encoder(input_ids=ids, attention_mask=mapping, past_key_values=cache,
                                    position_ids=pos, **kw).past_key_values
    encode(True)
    kw.pop("mm_token_type_ids", None)
    processor = model._prepare_logits_processor(cfg)
    eos = torch.tensor(cfg.eos_token_id, device=device)
    b3 = policy == "b3"
    w = 2 if b3 else int(policy.removeprefix("flow_w"))
    gate = B3Gate(block_size=b) if b3 else None
    active, trace = [], []
    def spawn():
        sampler = model._prepare_sampler(cfg)
        ids = sampler.initialize_canvas(1, device)
        active.append(SimpleNamespace(ids=ids, arg=ids, sampler=sampler,
            stop=model._prepare_diffusion_stopping_criteria(cfg), sc=None, previous=None,
            age=0, done=False, ordinal=(prefix.shape[1]-plen)//b + len(active)))
    spawn()
    while True:
        # Packed FlowBlock charges all W slots; B3 activates its second slot on demand.
        width = len(active) if b3 else w
        pad_slots = width-len(active)
        ids = torch.cat([s.ids for s in active] + [torch.zeros_like(active[0].ids)]*pad_slots, dim=1)
        masks = packing_mask(cache, model.config.text_config.layer_types, b, width, device)
        existing = next((s.sc for s in active if s.sc is not None), None)
        sc = None
        hook = nullcontext()
        if existing is not None:
            slots = [s.sc for s in active] + [None]*pad_slots
            sc = torch.cat([v if v is not None else torch.zeros_like(existing) for v in slots], dim=1)
            if any(v is None for v in slots):
                valid = torch.tensor([v is not None for v in slots], device=device).repeat_interleave(b)[None,:,None]
                # Native fresh slots use a zero soft embedding, not the embedding of uniform logits.
                def fresh(_, args):
                    return (args[0], args[1]*valid)
                hook = model.model.decoder.self_conditioning.register_forward_pre_hook(fresh)
        with hook:
            logits = model(decoder_input_ids=ids, self_conditioning_logits=sc, past_key_values=cache,
                decoder_attention_mask=masks, decoder_position_ids=positions(prefix.shape[1], width*b), **kw).logits
        accepted_abs, entropies = [], []
        for i, s in enumerate(active):
            raw = logits[:, i*b:(i+1)*b]
            if b3:
                entropies.append(logits_entropy(raw)[0])
            if s.done:
                continue
            prior = s.previous
            result = model._denoising_step(
                decoder_forward=lambda **ignored: SimpleNamespace(logits=raw), current_canvas=s.ids,
                argmax_canvas=s.arg, input_ids=prefix, decoder_position_ids=None,
                self_conditioning_logits=s.sc, mask_mapping=masks, past_key_values=cache,
                finished_denoising=torch.zeros(1, dtype=torch.bool, device=device),
                cur_step=cfg.max_denoising_steps-s.age, sampler=s.sampler, logits_processor=processor,
                diffusion_stopping_criteria=s.stop)
            s.ids, s.arg, s.sc, finished = result
            now = s.sampler.accepted_token_mask[0]
            eligible = torch.ones_like(now) if prior is None else ~prior
            new = now & eligible
            s.ratio = int(new.sum())/max(int(eligible.sum()), 1)
            if b3:
                accepted_abs.extend((new.nonzero().flatten()+s.ordinal*b).tolist())
            s.previous = now.clone()
            s.age += 1
            s.done = bool(finished) or s.age == cfg.max_denoising_steps
            if s.done:
                s.ids = s.arg
        if b3:
            open_next = gate.observe(active[0].ordinal, entropies[0], accepted_abs, len(active)>1,
                bool(torch.isin(active[0].arg, eos).any()),
                entropies[1] if len(active)>1 else None)["open_on_next_pass"]
        else:
            open_next = active[-1].ratio >= plan["flowblock_spawn_threshold"]
        trace.append({"width":width,"active":len(active),"q":width*b,"head":active[0].ordinal})
        del logits, raw, sc
        while active and active[0].done:
            s = active.pop(0)
            prefix = torch.cat([prefix, s.arg], dim=1)
            suffix = prefix[0, plen:]
            stop = torch.isin(s.arg[0], eos).nonzero()
            if len(stop) or len(suffix) == cap:
                end = len(suffix)-b+int(stop[0,0])+1 if len(stop) else len(suffix)
                return suffix[:end].tolist(), trace
            mask = torch.nn.functional.pad(mask, (0,b), value=True)
            encode(False)
        if not active or (open_next and len(active)<w and (prefix.shape[1]-plen)//b+len(active)<cap//b):
            spawn()


def run_wave(args, before_cuda=None):
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    plan = resolve_plan(args.config, args.block_size, args.flowblock_windows)
    native = json.loads((ROOT/"experiments/diffusiongemma-native/config.json").read_text())
    if transformers.__version__ != native["transformers"]:
        raise ValueError("Transformers version differs from native baseline")
    tasks = load_tasks(args.samples)
    check_snapshot(args.checkpoint)
    proc = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(args.checkpoint, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation=native["attention"], experts_implementation=native["experts"],
        device_map={"":"cuda" if before_cuda is None else "cpu"}).eval()
    if model.config.canvas_length != plan["block_size"]:
        raise ValueError("configured block_size must match checkpoint canvas_length")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"run.json").write_text(json.dumps(dict(plan, model=native["model"], revision=native["revision"],
        transformers=transformers.__version__, torch=str(torch.__version__)), indent=2))
    if before_cuda:
        before_cuda(model)
    model.to("cuda")
    eos = set(model.generation_config.eos_token_id)
    special = set(proc.tokenizer.all_special_ids) | eos | {model.generation_config.pad_token_id}
    with torch.inference_mode():
        for policy in plan["policies"]:
            for task in tasks:
                data = proc.apply_chat_template([{"role":"user","content":task["prompt"]}], tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
                torch.manual_seed(task["seed"])
                with observe(model) as events:
                    seq, trace = generate(model, data, policy, plan)
                b = plan["block_size"]
                if (events[0] != ("encoder", data["input_ids"].shape[1]) or
                    [q for tag,q in events[1:] if tag=="decoder"] != [t["q"] for t in trace] or
                    any(q != b for tag,q in events[1:] if tag=="encoder")):
                    raise RuntimeError("observed model work disagrees with scheduler")
                f = len(events)-1
                q = sum(q for _,q in events[1:])
                n = sum(i not in special for i in seq)
                if args.native_reference and policy == "flow_w1":
                    ref = json.loads((args.native_reference/f"{task['index']:02d}-native256.json").read_text())
                    if (any(ref[k]!=task[k] for k in ("task_id","seed","index","dataset")) or
                            seq != ref["output_ids"] or [list(e) for e in events] != ref["events"]):
                        raise RuntimeError(f"W1/native token or forward mismatch: {task['task_id']}")
                row = {k:task[k] for k in ("index","task_id","seed","dataset")}
                row.update(model="diffusiongemma", policy=policy, status="success", output_ids=seq,
                    output_text=proc.tokenizer.decode(seq,skip_special_tokens=True), output_tokens=n,
                    generation_forwards=f, generation_query_rows=q, generation_physical_padded_rows=q,
                    max_q=max(k for _,k in events[1:]), tpf=n/f, token_efficiency=n/q, events=events,
                    trace=trace, eos=seq[-1] in eos, truncated=seq[-1] not in eos)
                path = args.output/f"{task['index']:02d}-{policy}.json"
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(row, ensure_ascii=False))
                tmp.replace(path)
                print(policy, task["task_id"], f"N={n} F={f} Q={q}", flush=True)

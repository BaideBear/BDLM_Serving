"""FlowBlock and B3 request scheduling on the native DiffusionGemma sampler."""
from contextlib import nullcontext
from types import SimpleNamespace
import json
from pathlib import Path

import torch

from ..evaluation.frozen import ROOT, load_tasks, sha
from ..evaluation.original30 import check_reproduction, pin_rows, save_json
from ..methods.b3.gate import B3Gate, logits_entropy
from ..methods.t2t.wave_offer import TRACES, WaveTick, audit as audit_offers, periodic, widths
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


def generate(model, data, policy, plan, *, next_budget=None, verify_promotion=True):
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
    external = next_budget is not None
    if external and policy!="external_w2":
        raise ValueError("External offers require the whole-canvas request executor")
    compact = policy == "active_w2" or external
    w = 2 if b3 or compact else int(policy.removeprefix("flow_w"))
    gate = B3Gate(block_size=b) if b3 else None
    active, trace = [], []
    def spawn():
        sampler = model._prepare_sampler(cfg)
        ids = sampler.initialize_canvas(1, device)
        active.append(SimpleNamespace(ids=ids, arg=ids, sampler=sampler,
            stop=model._prepare_diffusion_stopping_criteria(cfg), sc=None, previous=None,
            age=0, done=False, tentative=False, debt=[], ratio=0.,
            ordinal=(prefix.shape[1]-plen)//b + len(active)))
    spawn()
    may_open=False
    while True:
        # Packed FlowBlock charges all W slots; B3 activates its second slot on demand.
        width = len(active) if b3 or compact else w
        sample_width=len(active)
        if external:
            state=WaveTick(len(trace),active[0].ordinal,len(active),tuple(len(s.debt) for s in active),may_open)
            offer=next_budget(state)
            widths(b,offer,[s.debt for s in active])  # validate before admission
            if may_open and offer==2*b:
                spawn()
            before=[s.debt for s in active]
            width,sample_width=widths(b,offer,before)
            ages=[s.age for s in active]
            admissions=[[] for s in active]
            edit_rows=[[] for s in active]
        projected=active[:width]
        pad_slots = width-len(projected)
        ids = torch.cat([s.ids for s in projected] + [torch.zeros_like(active[0].ids)]*pad_slots, dim=1)
        masks = packing_mask(cache, model.config.text_config.layer_types, b, width, device)
        existing = next((s.sc for s in projected if s.sc is not None), None)
        sc = None
        hook = nullcontext()
        if existing is not None:
            slots = [s.sc for s in projected] + [None]*pad_slots
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
        for i, s in enumerate(projected):
            raw = logits[:, i*b:(i+1)*b]
            if external:
                s.debt=[]
                if i>=sample_width:
                    continue
            if b3:
                entropies.append(logits_entropy(raw)[0])
            if s.done:
                continue
            if external and verify_promotion and i and (
                    s.tentative or s.age>=cfg.max_denoising_steps-1):
                # Leave at least one native step for the real committed prefix.
                continue
            prior = s.previous
            if external:
                old_ids=s.ids.clone()
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
            if external:
                strict=(now & prior & (old_ids[0]!=s.ids[0])) if prior is not None else torch.zeros_like(now)
                s.debt=(new | strict).nonzero().flatten().tolist()
                admissions[i]=new.nonzero().flatten().tolist()
                edit_rows[i]=strict.nonzero().flatten().tolist()
            s.ratio = int(new.sum())/max(int(eligible.sum()), 1)
            if b3:
                accepted_abs.extend((new.nonzero().flatten()+s.ordinal*b).tolist())
            s.previous = now.clone()
            s.age += 1
            s.done = bool(finished) or s.age == cfg.max_denoising_steps
            if external and verify_promotion and i:
                s.tentative,s.done=s.done,False
            elif s.done:
                s.ids = s.arg
        if b3:
            open_next = gate.observe(active[0].ordinal, entropies[0], accepted_abs, len(active)>1,
                bool(torch.isin(active[0].arg, eos).any()),
                entropies[1] if len(active)>1 else None)["open_on_next_pass"]
        else:
            open_next = active[-1].ratio >= plan["flowblock_spawn_threshold"]
        trace.append({"width":width,"active":len(active),"q":width*b,"head":active[0].ordinal})
        if external:
            trace[-1].update(tick=state.tick,offer=offer,open_eligible=may_open,
                dirty_before=before,dirty_after=[s.debt for s in active],ages_before=ages,
                ages_after=[s.age for s in active],sample_width=sample_width,
                refresh_only=width-sample_width,encoded=[],promoted=[],admitted=admissions,strict_edits=edit_rows)
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
            if external:
                trace[-1]["encoded"].append(s.ordinal)
                if active and verify_promotion:
                    active[0].tentative=False
                    # The old soft draft was produced against the speculative
                    # head, not the newly encoded prefix. Keep canvas and age.
                    active[0].sc=None
                    active[0].stop=model._prepare_diffusion_stopping_criteria(cfg)
                    trace[-1]["promoted"].append(active[0].ordinal)
                    if active[0].done or active[0].age>=cfg.max_denoising_steps:
                        raise ValueError("Speculative tail exhausted its authoritative verification step")
        may_open=bool(active and open_next and len(active)<w and (prefix.shape[1]-plen)//b+len(active)<cap//b)
        if not active or (may_open and not external):
            spawn()


def reference90(args, settings):
    root = getattr(args, "reference30_root", None)
    if root is None or args.native_reference is None:
        raise ValueError("frozen90 requires original30 wave and native90 references")
    old = json.loads((root/"run.json").read_text())
    # Changing which independent arms run is allowed; changing the algorithm,
    # canvas, sampler version or weights isn't.
    if (not set(settings["policies"]) <= set(old["policies"]) or
            any(old.get(k) != v for k, v in settings.items()
                if k not in ("policies", "flowblock_windows", "include_b3"))):
        raise ValueError("Original30 wave settings mismatch")
    return {p: pin_rows(root, p) for p in settings["policies"]}


def run_wave(args, before_cuda=None):
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    plan = resolve_plan(args.config, args.block_size, args.flowblock_windows)
    if getattr(args, "flow_only", False):
        plan.update(include_b3=False, policies=[p for p in plan["policies"] if p != "b3"])
    tier512 = getattr(args, "tier512_reference", None)
    switch = getattr(args, "wave_switch_reference", None)
    if switch is not None and tier512 is None:
        raise ValueError("Switch suite requires the archived wave controls")
    if tier512 is not None:
        if args.suite != "frozen90" or plan["block_size"] != 256 or plan["flowblock_windows"] != [1,2]:
            raise ValueError("Active-width W2 requires the registered 512-tier reference")
        plan.update(include_b3=False, policies=["flow_w2"])
    native = json.loads((ROOT/"archive/diffusiongemma-native/config.json").read_text())
    if transformers.__version__ != native["transformers"]:
        raise ValueError("Transformers version differs from native baseline")
    suite = getattr(args, "suite", "frozen30")
    if suite not in ("frozen30", "frozen90"):
        raise ValueError("Unknown test suite")
    settings = dict(plan, model=native["model"], revision=native["revision"],
                    transformers=transformers.__version__, torch=str(torch.__version__))
    refs = {}
    if suite == "frozen90":
        from ..evaluation.frozen90 import load_tasks as load_tasks90
        tasks = load_tasks90(args.samples)
        refs = reference90(args, settings)
        settings.update(suite=suite, seed_offset=0, samples_sha256=sha(args.samples))
        if tier512 is not None:
            archived = json.loads((tier512/"run.json").read_text())
            if (archived != dict(settings, policies=["flow_w1","flow_w2"]) or
                json.loads((tier512/"complete-flow_w2.json").read_text()) != dict(policy="flow_w2", requests=90)):
                raise ValueError("Wrong W2 reference or missing complete 90")
            settings.update(policies=["flow_w2","active_w2"], nominal_q=512,
                            qualification="replay packed W2 on all 90 before active_w2")
            if switch is not None:
                if (json.loads((switch/"run.json").read_text())!=settings or
                    json.loads((switch/"complete-active_w2.json").read_text())!=dict(policy="active_w2",requests=90)):
                    raise ValueError("Fixed active W2 was not qualified")
                settings.update(policies=list(TRACES),offer_sequences=TRACES,nominal_q=None,
                                qualification="flat W1 and flat active W2 output, events and original wave trace")
    else:
        tasks = load_tasks(args.samples)
    check_snapshot(args.checkpoint)
    proc = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(args.checkpoint, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation=native["attention"], experts_implementation=native["experts"],
        device_map={"":"cuda" if before_cuda is None else "cpu"}).eval()
    if model.config.canvas_length != plan["block_size"]:
        raise ValueError("configured block_size must match checkpoint canvas_length")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"run.json").write_text(json.dumps(settings, indent=2))
    if before_cuda:
        before_cuda(model)
    model.to("cuda")
    eos = set(model.generation_config.eos_token_id)
    special = set(proc.tokenizer.all_special_ids) | eos | {model.generation_config.pad_token_id}
    with torch.inference_mode():
        for policy in settings["policies"]:
            for task in tasks:
                data = proc.apply_chat_template([{"role":"user","content":task["prompt"]}], tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
                torch.manual_seed(task["seed"])
                with observe(model) as events:
                    if switch is not None:
                        seq,trace=generate(model,data,"external_w2",plan,next_budget=periodic(TRACES[policy]),
                                           verify_promotion=not policy.startswith("wave_bridge_"))
                        audit_offers(trace,events,plan["block_size"],TRACES[policy])
                    else:
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
                if switch is not None and policy.startswith("wave_bridge_"):
                    old,folder=("flow_w1",tier512) if TRACES[policy]==(256,) else ("active_w2",switch)
                    peer=json.loads((folder/f"{task['index']:02d}-{old}.json").read_text())
                    core=[{k:e[k] for k in ("width","active","q","head")} for e in trace]
                    if dict(row,policy=old,trace=core,events=[list(e) for e in events])!=peer:
                        raise ValueError(f"Flat offer did not replay {old}: {task['task_id']}")
                if tier512 is not None and policy == "flow_w2":
                    peer = json.loads((tier512/path.name).read_text())
                    if dict(row,events=[list(e) for e in events]) != peer:
                        raise ValueError(f"Packed reference differs: {task['task_id']}")
                if policy in refs and task["index"] < 30:
                    check_reproduction(refs[policy][task["index"]], row)
                save_json(path, row)
                if policy in refs and task["index"] == 29:
                    save_json(args.output/f"reproduced30-{policy}.json", dict(policy=policy,
                              matched=30, reference_sha256=[h for _, h in refs[policy]]))
                print(policy, task["task_id"], f"N={n} F={f} Q={q}", flush=True)
            if refs:
                save_json(args.output/f"complete-{policy}.json", dict(policy=policy, requests=len(tasks)))

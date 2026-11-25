"""Fixed Optimus IBS/OBS on DiffusionGemma: sparse decoder rows, native encoder."""
import argparse
from contextlib import ExitStack, contextmanager
from copy import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ..evaluation.frozen import ROOT, load_tasks
from ..methods.optimus.upstream import COMMIT, load_controls
from .diffusiongemma import check_snapshot, observe

CHUNKS = (4, 8, 16, 32, 64, 128, 256)
B = 256
CAP = 2048


def controller(controls, mode, c, start, previous):
    if mode not in ("ibs", "obs") or c not in CHUNKS:
        raise ValueError((mode, c))
    width = min(B + (c if mode == "obs" else 0), CAP-start)
    ctl = controls.DelayedCacheState.new(B, window_length=width, chunked_decode_enabled=True,
        chunk_size=c, extended_window_enabled=mode == "obs")
    if previous is not None:
        prior = np.zeros(width, dtype=np.uint8)
        prior[:min(width, len(previous))] = previous[:width]
        ctl.prime_from_shifted_window(prior)
    return ctl, width


class Resident:
    def __init__(self):
        self.initialized = np.zeros(CAP, dtype=bool)
        self.buffers = {}

    def select(self, start, absolute, device):
        self.start = start
        self.end = int(absolute[-1])+1
        self.index = torch.as_tensor(absolute, device=device)
        self.initialized[absolute] = True
        if not self.initialized[start:self.end].all():
            raise RuntimeError("uninitialized resident decoder KV")
        self.visited = set()

    def extend(self, layer, q, k, v):
        count = self.index.numel()
        if q.shape[-2] != count or layer in self.visited:
            raise RuntimeError("decoder Q disagrees with planner")
        self.visited.add(layer)
        if layer not in self.buffers:
            self.buffers[layer] = tuple(t.new_empty((*t.shape[:-2], CAP, t.shape[-1])) for t in (k,v))
        result = []
        for storage, t in zip(self.buffers[layer], (k,v), strict=True):
            storage.index_copy_(-2, self.index, t[..., -count:, :])
            result.append(torch.cat((t[..., :-count, :], storage[..., self.start:self.end, :]), dim=-2))
        return result


@contextmanager
def resident_attention(model, bank):
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import ALL_ATTENTION_FUNCTIONS as impl
    alias = "bdlm_optimus_resident_sdpa"
    if alias in impl:
        raise RuntimeError("resident attention already active")
    original = impl["sdpa"]

    def sparse(module, q, k, v, mask, **kwargs):
        if bank.bypass:
            return original(module, q, k, v, mask, **kwargs)
        if mask is not None or kwargs.get("is_causal"):
            raise RuntimeError("expected unpadded bidirectional decoder")
        k, v = bank.extend(module.layer_idx, q, k, v)
        return original(module, q, k, v, mask, **kwargs)

    with ExitStack() as stack:
        impl[alias] = sparse
        stack.callback(impl.__delitem__, alias)
        for layer in model.model.decoder.layers:
            attn = layer.self_attn
            if attn.config._attn_implementation != "sdpa":
                raise ValueError("expected pinned SDPA backend")
            stack.callback(setattr, attn, "config", attn.config)
            attn.config = copy(attn.config)
            attn.config._attn_implementation = alias
        yield


def generate(model, data, controls, unmask, mode, c, qualify=False):
    bank = Resident()
    bank.bypass = False
    with resident_attention(model,bank):
        return _generate(bank,model,data,controls,unmask,mode,c,qualify)


def _generate(bank, model, data, controls, unmask, mode, c, qualify):
    cfg, kw = model._prepare_generation_config(None, max_new_tokens=CAP)
    prefix, mask = data["input_ids"], data["attention_mask"].bool()
    plen, device = prefix.shape[1], prefix.device
    if prefix.shape[0] != 1 or not mask.all():
        raise ValueError("requires one unpadded request")
    kw.update({k:v for k,v in data.items() if k not in ("input_ids","attention_mask")})
    cache = model._prepare_cache_for_generation(cfg, 1, plen+CAP-B)
    if cache.is_compileable:
        raise ValueError("requires native dynamic cache")

    def encode(first):
        nonlocal cache
        off = 0 if first else prefix.shape[1]-B
        pos = torch.arange(off, prefix.shape[1], dtype=torch.int32, device=device)[None]
        ids, mapping = model._prepare_encoder_inputs(prefix, mask, pos, cache, first, B, 1, **kw)
        cache = model.model.encoder(input_ids=ids, attention_mask=mapping, past_key_values=cache,
                                    position_ids=pos, **kw).past_key_values

    encode(True)
    kw.pop("mm_token_type_ids", None)
    sampler = model._prepare_sampler(cfg)
    # Initialize identical random canvases for every arm before policy-specific re-noising.
    ids = torch.cat([sampler.initialize_canvas(1,device) for _ in range(CAP//B)],dim=1)
    resolved = np.zeros(CAP,dtype=np.uint8)
    masks = dict.fromkeys(model.config.text_config.layer_types)
    trace, previous = [], None
    vocab = model.model.decoder.embed_tokens.weight.shape[0]
    dtype = model.model.decoder.embed_tokens.weight.dtype
    sc_bank = torch.zeros(2*B, vocab, device=device,dtype=dtype)
    eos = set(cfg.eos_token_id)
    for start in range(0,CAP,B):
        ctl, width = controller(controls, mode, c, start, previous)
        buf = np.empty(width, dtype=np.int64)
        for step in range(2*width+4):
            prior = resolved[start:start+width].copy()
            # A whole OBS tail can already have been finalized before the slide.
            if ctl.cached_prefix_len >= B:
                previous = prior[B:].copy()
                break
            n, _ = ctl.fill_processing_buffers(buf, 0, prior)
            if n <= 0:
                raise RuntimeError("empty Optimus selection")
            local = buf[:n].copy()
            absolute = start+local
            seen = bank.initialized[absolute].copy()
            bank.select(start, absolute, device)
            index = bank.index
            pos = (index+plen).to(torch.int32)[None]
            sc = sc_bank[index % (2*B)][None] if seen.any() else None
            args = dict(decoder_input_ids=ids[:,index], self_conditioning_logits=sc,
                        decoder_attention_mask=masks, decoder_position_ids=pos, past_key_values=cache, **kw)
            if qualify and not trace:
                bank.bypass = True
                ref = model(**args).logits
                bank.bypass = False
            with ExitStack() as stack:
                if sc is not None and not seen.all():
                    valid = torch.as_tensor(seen,device=device)[None,:,None]
                    handle = model.model.decoder.self_conditioning.register_forward_pre_hook(
                        lambda _,args: (args[0], args[1]*valid))
                    stack.callback(handle.remove)
                logits = model(**args).logits[0]
            if len(bank.visited) != len(model.model.decoder.layers):
                raise RuntimeError("not all decoder layers projected selected Q")
            if qualify and not trace:
                torch.testing.assert_close(logits, ref[0], rtol=0,atol=0)
                del ref
            sc_bank.index_copy_(0,index % (2*B),logits.to(dtype))
            unresolved = prior[local] == 0
            accepted = 0
            if unresolved.any():
                sel = torch.as_tensor(local[unresolved],device=device)
                raw = logits[torch.as_tensor(unresolved,device=device)]
                proposal = raw.argmax(-1)
                updated = unmask.low_confidence_dynamic_sparse(raw, proposal,
                    torch.as_tensor(prior.copy(),device=device),sel,row_width=width)
                now = updated.cpu().numpy()
                take = updated[sel].bool()
                accepted = int(now.sum())-int(prior.sum())
                if accepted < 1 or np.any((now != prior) & ~np.isin(np.arange(width),local)):
                    raise RuntimeError("unmasker made no progress or touched an unselected position")
                # Re-noise only unresolved rows that were actually evaluated in this pass.
                noise = torch.randint(sampler.vocab_size,proposal.shape,device=device)
                ids[0,start+sel] = torch.where(take,proposal,noise)
                resolved[start:start+width] = now
            ctl.update_from_mask(prior, mark_ready_if_all_left_non_mask=True,
                                 dllm_mask_cur=resolved[start:start+width])
            trace.append(dict(block=start//B,step=step,q=n,accepted=accepted,ready=ctl.cached_prefix_len,
                              cross=int((local>=B).sum()),visible_end=bank.end))
            if qualify and len(trace)==2:
                return [], trace
            if ctl.cached_prefix_len >= B:
                previous = prior[B:].copy() if mode=="obs" else None
                break
        else:
            raise RuntimeError("Optimus progress bound exceeded")
        if not resolved[start:start+B].all():
            raise RuntimeError("retired unfinished canvas")
        head = ids[0,start:start+B].tolist()
        stop = next((i for i,t in enumerate(head) if t in eos),None)
        if stop is not None or start+B == CAP:
            end = start+(stop+1 if stop is not None else B)
            return ids[0,:end].tolist(),trace
        prefix = torch.cat((prefix,ids[:,start:start+B]),dim=1)
        mask = torch.nn.functional.pad(mask,(0,B),value=True)
        encode(False)
    raise RuntimeError("generation ended without EOS or cap")


def run(args, before_cuda=None):
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    native = json.loads((ROOT/"archive/diffusiongemma-native/config.json").read_text())
    if transformers.__version__ != native["transformers"]:
        raise ValueError("wrong Transformers version")
    controls, strategy, hashes = load_controls()
    unmask = controls.UnmaskingProcessor(SimpleNamespace(confidence_threshold=.9, block_length=B,
        denoising_steps=B, unmasking_strategy=strategy.LOW_CONFIDENCE_DYNAMIC))
    tasks = load_tasks(args.samples)
    check_snapshot(args.checkpoint)
    proc = AutoProcessor.from_pretrained(args.checkpoint,local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(args.checkpoint,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation=native["attention"],experts_implementation=native["experts"],
        device_map={"":"cpu" if before_cuda else "cuda"}).eval()
    if model.config.canvas_length != B:
        raise ValueError("checkpoint is not B256")
    policies = [(m,c) for c in args.chunks for m in args.modes]
    if len(set(policies)) != len(policies):
        raise ValueError("duplicate configurations")
    args.output.mkdir(exist_ok=False,parents=True)
    (args.output/"run.json").write_text(json.dumps(dict(native, policies=[f"{m}_{c}" for m,c in policies],
        upstream=COMMIT,hashes=hashes,acceptance="optimus_dynamic_sparse_argmax_0.9",
        hard_q_cap=False, torch=str(torch.__version__)),indent=2))
    if before_cuda:
        before_cuda(model)
    model.to("cuda")
    eos = set(model.generation_config.eos_token_id)
    special = set(proc.tokenizer.all_special_ids)|eos|{model.generation_config.pad_token_id}
    inputs = [proc.apply_chat_template([{"role":"user","content":t["prompt"]}],tokenize=True,
        add_generation_prompt=True, return_dict=True,return_tensors="pt").to(model.device) for t in tasks]
    with torch.inference_mode():
        for m in args.modes:
            torch.manual_seed(tasks[0]["seed"])
            generate(model,inputs[0],controls,unmask,m,B,qualify=True)
        print("Smoke passed: native/selected first-step logits bit-identical; sparse Q observed.",flush=True)
        for mode,c in policies:
            policy = f"{mode}_{c}"
            for t,data in zip(tasks,inputs,strict=True):
                torch.manual_seed(t["seed"])
                with observe(model) as events:
                    seq, trace = generate(model,data,controls,unmask,mode,c)
                if (events[0] != ("encoder",data["input_ids"].shape[1]) or
                    [q for tag,q in events[1:] if tag=="decoder"] != [x["q"] for x in trace] or
                    any(q!=B for tag,q in events[1:] if tag=="encoder") or
                    (mode=="ibs" and any(x["cross"] for x in trace))):
                    raise RuntimeError("observed generation work differs from trace")
                n = sum(i not in special for i in seq)
                f, q = len(events)-1, sum(q for _,q in events[1:])
                row = {k:t[k] for k in ("index","task_id","seed","dataset")}
                row.update(model="gemma",policy=policy,status="success",output_ids=seq,
                    output_text=proc.tokenizer.decode(seq,skip_special_tokens=True),events=events,trace=trace,
                    output_tokens=n,generation_forwards=f,generation_query_rows=q,generation_physical_padded_rows=q,
                    max_q=max(q for _,q in events[1:]),max_decoder_q=max(x["q"] for x in trace),
                    tpf=n/f,token_efficiency=n/q,eos=seq[-1] in eos,truncated=seq[-1] not in eos)
                path = args.output/f"{t['index']:02d}-{policy}.json"
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(row,ensure_ascii=False))
                tmp.replace(path)
                print(policy,t["task_id"],f"N={n} F={f} Q={q}",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint","samples","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--modes",choices=["ibs","obs"],nargs="+",default=["ibs","obs"])
    parser.add_argument("--chunks",choices=CHUNKS,type=int,nargs="+",default=CHUNKS)
    run(parser.parse_args())

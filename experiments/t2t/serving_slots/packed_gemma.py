"""Request-private WaveKV banks with shared batched decoder projections."""
from contextlib import ExitStack
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from .gemma_fixed_dual import AblationRequest, B
from .gemma_wave_kv import WaveKV, wave_attention
from .packed_decode import projection_audit


def gemma_fn(model, item, offer, arm):
    def evaluate(issue):
        class Request(AblationRequest):
            def _encode(self, length, *, first):
                def encode():
                    return super(Request, self)._encode(length, first=first)
                issue("encoder", length, encode)

            def _project(self, states, positions):
                if self.cache.get_seq_length() != self.origin+B*self.ordinal:
                    raise RuntimeError("wrong encoder epoch")
                ids, hist = [], []
                for c in states:
                    base = self.origin+B*c.ordinal
                    local = [p-base for p in positions if base <= p < base+B]
                    if local:
                        ids.append(c.sampler.ids[:,local])
                        hist.extend(c.sampler.sc.get(i) for i in local)
                self.bank.begin(positions, self.origin, self.ids.device, causal=True)
                logits, receipt = issue("decoder", len(positions),
                    (self, tuple(positions), torch.cat(ids, dim=1), hist))
                self.physical.append(receipt)
                return logits

        with torch.inference_mode():
            req = Request(model, item["data"], q=offer, ablation=arm, seed=item["seed"],
                          max_new_tokens=item.get("max_new_tokens", 2048))
            req.physical = []
            trace = []
            while req.status == "ready":
                e = req.step()
                e["batch_receipt"] = req.physical[-1]
                trace.append(e)
            return dict(ids=req.tokens, trace=trace)
    return evaluate


class Banks(WaveKV):
    """Combine lane-local K/V only after applying the original physical-block mask."""
    def __init__(self, group):
        super().__init__(B)
        self.group = group

    def extend(self, layer, q, k, v, mask=None):
        if mask is not None or q.shape[-2] != k.shape[-2]:
            raise RuntimeError("expected only selected decoder K/V, with no shared encoder prefix")
        all_k, all_v, all_m = [], [], []
        for i, call in enumerate(self.group):
            req, w = call.data[0], call.width
            enc = req.cache.layers[layer]
            kl = torch.cat((enc.keys, k[i:i+1,:,:w]), dim=2)
            vl = torch.cat((enc.values, v[i:i+1,:,:w]), dim=2)
            lk, lv, lm = req.bank.extend(layer, q[i:i+1,:,:w], kl, vl, None)
            all_k.append(lk)
            all_v.append(lv)
            all_m.append(lm)
        end = max(ki.shape[-2] for ki in all_k)
        def padded(x):
            return torch.cat([F.pad(a,(0,0,0,end-a.shape[-2])) for a in x])
        masks = []
        for m in all_m:
            w = m.shape[-2]
            m = F.pad(m,(0,end-m.shape[-1],0,q.shape[-2]-w))
            m[:,:,w:,0] = True
            masks.append(m)
        return padded(all_k), padded(all_v), torch.cat(masks)


def gemma_forward(model, group, width):
    ids, positions, histories, validity = [], [], [], []
    for c in group:
        req, pos, selected, hist = c.data
        if len(pos) != c.width or len(hist) != c.width:
            raise RuntimeError("missing selected input")
        ids.append(F.pad(selected,(0,width-c.width)))
        positions.append(F.pad(torch.as_tensor(pos,device=selected.device,dtype=torch.int32)[None],
                               (0,width-c.width)))
        histories.extend([*hist, *([None]*(width-c.width))])
        validity.extend([x is not None for x in hist]+[False]*(width-c.width))
    n = len(group)
    exemplar = next((x for x in histories if x is not None), None)
    sc = None
    if exemplar is not None:
        zero = torch.zeros_like(exemplar)
        sc = torch.stack([zero if x is None else x for x in histories]).reshape(n, width, -1)
    # Native attention requires a cache object. Pass an empty prefix view;
    # Banks.extend inserts the *correct lane's* real encoder prefix per layer.
    layers = group[0].data[0].cache.layers
    def empty(x):
        return x[:,:,:0].expand(n,-1,-1,-1)
    empty_prefix = SimpleNamespace(layers=[SimpleNamespace(keys=empty(l.keys),values=empty(l.values))
                                          for l in layers])
    modules = [m for l in model.model.decoder.layers for name in ("q_proj","k_proj","v_proj")
               if (m := getattr(l.self_attn, name)) is not None]
    if len(modules) != 85:
        raise RuntimeError("unexpected Gemma checkpoint layout")
    with ExitStack() as stack:
        visits = projection_audit(stack, modules, n, width)
        stack.enter_context(wave_attention(model, Banks(group)))
        if sc is not None and not all(validity):
            valid = torch.tensor(validity,device=ids[0].device).reshape(n,width,1)
            stack.enter_context(model.model.decoder.self_conditioning.register_forward_pre_hook(
                lambda _, a: (a[0],a[1].masked_fill(~valid,0))))
        logits = model(decoder_input_ids=torch.cat(ids), decoder_position_ids=torch.cat(positions),
                       self_conditioning_logits=sc, past_key_values=empty_prefix,
                       decoder_attention_mask=dict.fromkeys(model.config.text_config.layer_types)).logits
    if visits != [1]*85 or any(len(c.data[0].bank.visited) != len(model.model.decoder.layers)
                               for c in group):
        raise RuntimeError("incorrect physical decoder or KV work")
    return logits

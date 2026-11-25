"""Dependency-preserving layer tiling control for the native LLaDA backend.

All open positions are refreshed at every layer. Smaller tiles are NOT free
work reduction and are not token-level permanent retirement.
"""
import importlib
import types

import torch
from transformers.cache_utils import DynamicCache


class AbsoluteKV(DynamicCache):
    def __init__(self, capacity, layers):
        super().__init__()
        self.capacity = capacity
        self.buffers = [None] * layers
        self.length = 0

    @classmethod
    def from_pairs(cls, pairs, capacity):
        result = cls(capacity, len(pairs))
        for layer, (k, v) in enumerate(pairs):
            result.write(layer, 0, k, v)
        result.length = pairs[0][0].shape[-2] if pairs else 0
        return result

    def write(self, layer, start, k, v):
        end = start + k.shape[-2]
        if end > self.capacity or k.shape != v.shape:
            raise ValueError("Invalid absolute KV range")
        if self.buffers[layer] is None:
            shape = (2, k.shape[0], k.shape[1], self.capacity, k.shape[-1])
            self.buffers[layer] = torch.empty(shape, dtype=k.dtype, device=k.device)
        storage = self.buffers[layer]
        storage[0, :, :, start:end].copy_(k)
        storage[1, :, :, start:end].copy_(v)

    def get_seq_length(self, layer_idx=0):
        return self.length

    def crop(self, maximum_length):
        if maximum_length < 0 or maximum_length > self.length:
            raise ValueError("Only valid prefix crop is supported")
        self.length = maximum_length

    def to_legacy_cache(self):
        return tuple((x[0, :, :, :self.length], x[1, :, :, :self.length])
                     for x in self.buffers if x is not None)

    def update(self, *args, **kwargs):
        raise RuntimeError("Use explicit absolute-position write(), not append()")


class LayerTiledLLaDA:
    def __init__(self, native_model, q_budget, validate_inputs=True):
        if q_budget not in (8, 16, 32, 64):
            raise ValueError("Unsupported Q budget")
        self.native = native_model
        self.q_budget = q_budget
        self.module = importlib.import_module(native_model.__class__.__module__)
        self.last_work = None
        self.validate_inputs = validate_inputs

    @property
    def device(self):
        return self.native.device

    @property
    def dtype(self):
        return self.native.dtype

    def _sample_with_temperature_topk_topp(self, *args, **kwargs):
        return self.native._sample_with_temperature_topk_topp(*args, **kwargs)

    @torch.inference_mode()
    def __call__(self, input_ids, *, attention_mask, position_ids, past_key_values,
                 use_cache=True, output_attentions=True):
        if not isinstance(past_key_values, AbsoluteKV) or not use_cache:
            raise ValueError("Explicit absolute cache required")
        cache = past_key_values
        if input_ids.shape[0] != 1:
            raise ValueError("Single-request qualification only")
        start = cache.length
        width = input_ids.shape[1]
        end = start + width
        if self.validate_inputs:
            expected = torch.arange(start, end, device=input_ids.device)[None]
            if not torch.equal(position_ids, expected):
                raise ValueError("This control requires a contiguous open dependency region")
        if attention_mask.shape != (1, 1, width, end):
            raise ValueError("Wrong absolute visibility geometry")
        if self.validate_inputs and not ((attention_mask == 0) | (attention_mask == 1)).all():
            raise ValueError("Binary visibility required")
        mask = torch.zeros_like(attention_mask, dtype=self.dtype).masked_fill(
            ~attention_mask.bool(), torch.finfo(self.dtype).min)
        core = self.native.model
        hidden = core.word_embeddings(input_ids)
        positions = core.rotary_emb(hidden, position_ids)
        ranges = [(lo, min(lo+self.q_budget, width)) for lo in range(0, width, self.q_budget)]
        layer_shapes = []
        for layer_index, layer in enumerate(core.layers):
            attn = layer.attention
            queries = []
            for lo, hi in ranges:
                h = layer.input_layernorm(hidden[:, lo:hi])
                qkv = attn.query_key_value(h).view(
                    1, hi-lo, attn.num_heads+2*attn.num_key_value_heads, attn.head_dim)
                q, k, v = qkv.split(
                    [attn.num_heads, attn.num_key_value_heads, attn.num_key_value_heads], dim=-2)
                q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                if attn.config.use_qk_norm:
                    q, k = attn.query_layernorm(q), attn.key_layernorm(k)
                q, k = self.module.apply_rotary_pos_emb(
                    q, k, positions[0][:, lo:hi], positions[1][:, lo:hi])
                cache.write(layer_index, start+lo, k, v)
                queries.append(q)
            # The dependency barrier is essential: every K/V row for this layer
            # now corresponds to the same frozen layer input, including rows
            # that will be queried in a later tile.
            storage = cache.buffers[layer_index]
            keys, values = storage[0, :, :, :end], storage[1, :, :, :end]
            outputs = []
            observed = []
            for (lo, hi), query in zip(ranges, queries):
                interface = (self.module.eager_attention_forward
                             if attn.config._attn_implementation == "eager"
                             else self.module.ALL_ATTENTION_FUNCTIONS[attn.config._attn_implementation])
                attended, _ = interface(
                    attn, query, keys, values, mask[:, :, lo:hi],
                    dropout=0., scaling=attn.scaling, sliding_window=attn.sliding_window)
                attended = attn.dense(attended.reshape(1, hi-lo, -1).contiguous())
                post = hidden[:, lo:hi] + attended
                update = layer.mlp(layer.post_attention_layernorm(post))
                if isinstance(update, tuple):
                    update = update[0]
                outputs.append(post + update.to(post.device))
                observed.append({"query": query.shape[-2], "kv": keys.shape[-2],
                                 "mlp_rows": post.shape[1],
                                 "visible_qk_pairs": None})
            hidden = torch.cat(outputs, dim=1)
            layer_shapes.append(observed)
        logits = torch.cat([self.native.lm_head(core.norm(hidden[:, lo:hi])).float()
                            for lo, hi in ranges], dim=1)
        cache.length = end
        self.last_work = {
            "q_budget": self.q_budget, "window_rows": width, "prefix_rows": start,
            "layers": len(layer_shapes), "layer_shapes": layer_shapes,
            "physical_attention_calls": sum(map(len, layer_shapes)),
            "physical_query_rows_over_layers": sum(x["query"] for l in layer_shapes for x in l),
            "max_query_rows": max(x["query"] for l in layer_shapes for x in l),
            "equivalent_full_stack_sweeps": sum(map(len, layer_shapes))/len(layer_shapes),
            "fresh_kv_all_open_positions": True, "prefix_moved": False,
        }
        return types.SimpleNamespace(logits=logits, past_key_values=cache)

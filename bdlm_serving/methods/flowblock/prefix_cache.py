"""FlowBlock packed controller plus delayed, per-block prefix retirement only.

The original author loop is retained. Retired rows are gathered out before
the transformer; their resident KV is put in the original packed key layout.
"""
import types
import torch
from transformers.cache_utils import DynamicCache
from bdlm_serving.methods.flowblock.control import FlowBlockControl, NativeBackend


class SparseTailCache(DynamicCache):
    def __init__(self, data, positions, selected, full_q):
        super().__init__()
        self.data, self.positions, self.selected = data, positions, selected
        self.tail_start = data.shape[4] - full_q
        self.present = [None] * (2 * data.shape[0])
        # The rectangular-mask adapter recognizes this selected-cache marker.
        self.visible_end = data.shape[4]
        self.layer_rows = []

    def get_seq_length(self, layer_idx=0):
        return self.tail_start

    def update(self, k, v, layer, cache_kwargs=None):
        self.layer_rows.append([layer, int(k.shape[-2])])
        result = []
        for which, current in enumerate((k, v)):
            # Preserve original absolute committed prefix and synthetic tail.
            old = self.data[layer, which]
            tail = old.index_select(-2, self.positions).clone()
            tail.index_copy_(-2, self.selected, current)
            full = torch.cat((old[..., :self.tail_start, :], tail), dim=-2)
            self.present[2*layer+which] = full
            result.append(full)
        return tuple(result)


class PrefixBackend(NativeBackend):
    def __init__(self, native, controller, enabled):
        super().__init__(native)
        self.controller, self.enabled = controller, enabled
        self.prefixes, self.last_inputs = {}, {}
        self.slot = 0

    def __call__(self, unused, *, inputs_embeds, position_ids, use_cache,
                 past_key_values, attention_mask, replace_position):
        self.slot = 0
        width = inputs_embeds.shape[1]
        self.skipped = torch.zeros(width, dtype=torch.bool, device=self.device)
        if not self.enabled:
            return super().__call__(unused, inputs_embeds=inputs_embeds,
                position_ids=position_ids, use_cache=use_cache,
                past_key_values=past_key_values, attention_mask=attention_mask,
                replace_position=replace_position)
        assert inputs_embeds.shape[0] == 1, "Requests are batched by ForwardBatcher"
        data = past_key_values._data
        start = int(position_ids[0, 0])
        absolute = torch.arange(start, start+width, device=self.device)
        raw = self.controller.raw_tokens.data
        tokens = raw[0].index_select(0, position_ids[0])
        for slot in range(width//32):
            block = (start//32)+slot
            lo, hi = slot*32, (slot+1)*32
            # Preserve original extra/inactive window work; no second optimization.
            if start+hi > raw.shape[1]:
                continue
            now = tokens[lo:hi]
            prefix = self.prefixes.get(block, 0)
            old = self.last_inputs.get(block)
            if old is not None:
                assert torch.equal(now[:prefix], old[:prefix]), "Retired token edited"
                eligible = (now != self.controller.owner.decoder.mask_id) & (now == old)
                bad = (~eligible).nonzero().flatten()
                leading = int(bad[0]) if bad.numel() else 32
                prefix = max(prefix, leading)
            self.prefixes[block] = prefix
            self.skipped[lo:lo+prefix] = True
            self.last_inputs[block] = now.clone()
        # The author loop clamps out-of-range window slots to the final real
        # token. Once that token retires, remove ALL its query replicas too.
        # Their old KV remains at the original virtual tail addresses.
        retired_positions = position_ids[0, self.skipped]
        removed_before_aliases = int(self.skipped.sum())
        if retired_positions.numel():
            self.skipped |= torch.isin(position_ids[0], retired_positions)
        boundary_aliases = int(self.skipped.sum()) - removed_before_aliases
        selected = (~self.skipped).nonzero().flatten()
        query = selected.numel()
        cache = SparseTailCache(data, absolute.clamp(max=data.shape[4]-1), selected, width)
        if query:
            out = self.native(
                inputs_embeds=inputs_embeds.index_select(1, selected),
                position_ids=position_ids.index_select(1, selected),
                attention_mask=attention_mask.index_select(1, selected)[:, None],
                use_cache=True, past_key_values=cache, output_attentions=True)
            full_logits = out.logits.new_zeros((1, width, out.logits.shape[-1]))
            full_logits.index_copy_(1, selected, out.logits)
            assert len(cache.layer_rows) == len(self.native.model.layers)
            assert all(n == query for _, n in cache.layer_rows)
        else:
            full_logits = inputs_embeds.new_zeros(
                (1, width, self.native.config.vocab_size))
            for layer in range(data.shape[0]):
                for which in (0, 1):
                    old = data[layer, which]
                    tail = old.index_select(-2, cache.positions)
                    cache.present[2*layer+which] = torch.cat(
                        (old[..., :cache.tail_start, :], tail), dim=-2)
        self.events.append({
            "phase": "flowblock_prefix", "q_tokens": query,
            "physical_forwards": int(query > 0), "max_physical_q": query,
            "dense_q": width, "skipped_q": width-query,
            "retired_boundary_aliases": boundary_aliases,
            "kv_length": data.shape[4],
            "query_rows_over_layers": query*len(self.native.model.layers),
            "cache_layer_rows": cache.layer_rows,
            "absolute_positions": position_ids[0, selected].tolist(),
            "removed_positions": position_ids[0, self.skipped].tolist(),
            "block_prefixes": [self.prefixes.get(start//32+s, 0)
                               for s in range(width//32)],
            "original_key_mask_verified": True,
        })
        return types.SimpleNamespace(logits=full_logits, past_key_values=cache.present)


class FlowBlockPrefix(FlowBlockControl):
    def __init__(self, *args, enabled=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = PrefixBackend(self.backend.native, self, enabled)
        self.owner.model = self.backend
        original_joint = self.owner._joint_decode_slot_packed
        original_sample = self.owner.decoder._top1_and_confidence

        def joint(owner, logits, tokens, prompt_mask, post_edit, finished, active):
            slot = self.backend.slot
            self.backend.slot += 1
            frozen = self.backend.skipped[slot*32:(slot+1)*32][None]

            def sample(values):
                # Full block shape preserves SDAR's per-block RNG consumption.
                ids, confidence = original_sample(values)
                return (torch.where(frozen, tokens, ids),
                        torch.where(frozen, torch.zeros_like(confidence), confidence))

            owner.decoder._top1_and_confidence = sample
            try:
                result = original_joint(logits, tokens, prompt_mask,
                                        post_edit, finished, active)
                if self.enabled:
                    event = self.backend.events[-1]
                    changed = result[0] != tokens
                    edit = changed & (tokens != owner.decoder.mask_id)
                    event["edited"] = event.get("edited", 0) + int(edit.sum())
                    event["newly_unmasked"] = event.get("newly_unmasked", 0) + int(
                        (changed & (tokens == owner.decoder.mask_id)).sum())
                return result
            finally:
                owner.decoder._top1_and_confidence = original_sample

        self.owner._joint_decode_slot_packed = types.MethodType(joint, self.owner)
        self.enabled = enabled
        self.provenance.update({
            "prefix_cache_enabled": enabled,
            "prefix_rule": "Leading non-MASK inputs unchanged since their last forward",
            "retired_rows": "Excluded before transformer; resident KV frozen; no later T2T",
            "sampling": "Original full-block sampler shape; retired candidates suppressed",
            "remaining_control": "Original author AST, original block-causal key layout",
        })

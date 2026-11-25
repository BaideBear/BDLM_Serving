"""B32-aligned B3 control with actual native full-model calls.

Gate and sampler are inherited unchanged. Only the old layered executor and
physical-call metadata are replaced. This is not an official B3 engine.
"""
import torch
from transformers.cache_utils import DynamicCache
from bdlm_serving.methods.b3.aligned import B3AlignedKernel


class NativeB3Kernel(B3AlignedKernel):
    protocol = "llada_native_full_forward_b3_b32_v1"

    def _record(self, state, phase, start, end, prefix, *, work=None):
        super()._record(state, phase, start, end, prefix, work=work)
        state.events[-1].update(
            execution_protocol=self.protocol, nfe_is_layer_proxy=False,
            native_model_calls=0 if work is None else 1)

    def _forward(self, state, q_start, q_end, store, phase):
        result = super()._forward(state, q_start, q_end, store, phase)
        if phase == "prefill":
            state.events[-1].update(native_model_calls=1, generation_budget_applies=False)
        return result

    def _execute(self, state, start, end, phase):
        assert state.cache.length == start and 0 < end-start <= 64
        cache = DynamicCache()
        for layer, (k, v) in enumerate(state.cache.to_legacy_cache()):
            cache.update(k, v, layer)
        pos = torch.arange(start, end, device=self.model.device)
        keys = torch.arange(end, device=self.model.device)
        mask = (pos[:, None]//32 >= keys[None, :]//32)[None, None]
        out = self.model(state.x[:, start:end], attention_mask=mask,
                         position_ids=pos[None], past_key_values=cache,
                         use_cache=True, output_attentions=True)
        written = 0
        for layer, (k, v) in enumerate(cache):
            state.cache.write(layer, start, k[:, :, start:end], v[:, :, start:end])
            written += (k[:, :, start:end].numel()+v[:, :, start:end].numel())*k.element_size()
        state.cache.length = end
        layers = len(self.model.model.layers)
        self._record(state, phase, start, end, start, work={
            "equivalent_full_stack_sweeps": 1, "physical_attention_calls": layers,
            "max_query_rows": end-start, "physical_query_rows_over_layers": layers*(end-start),
            "layer_shapes": [[{"query": end-start, "kv": end, "mlp_rows": end-start,
                              "visible_qk_pairs": None}] for _ in range(layers)]})
        state.events[-1]["active_kv_write_bytes"] = written
        for block in range(start//32, end//32):
            state.kv_status[block] = {"status": "PROVISIONAL",
                "token_revision": state.block_revisions.get(block, 0),
                "context_revision": state.context_version}
        return out.logits

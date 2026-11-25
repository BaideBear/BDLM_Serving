"""Conservative ERQW control: exact block sealing and bounded layer tiles.

This is a static-Q prototype. It does not claim safe token/group retirement.
The native sampler and stopping rule live in the archived SerialKernel.
"""
from dataclasses import dataclass, field

import torch
from transformers.cache_utils import DynamicCache

from bdlm_serving.runtime.tiled import AbsoluteKV, LayerTiledLLaDA
from bdlm_serving.runtime.serial import RequestState, SerialKernel


@dataclass
class RetirementState(RequestState):
    block_revisions: dict = field(default_factory=dict)
    kv_status: dict = field(default_factory=dict)
    seal_ready: object = None
    pending_refill: object = None


class ExactRollingKernel(SerialKernel):
    def __init__(self, model, q_budget, seal=True, fuse_refresh=False):
        super().__init__(model, "llada21")
        self.q_budget = q_budget
        self.seal = seal
        self.fuse_refresh = fuse_refresh
        if fuse_refresh and q_budget != 64:
            raise ValueError("A fused two-block call requires the Q64 tier")
        self.tiled = LayerTiledLLaDA(model, q_budget, validate_inputs=False)

    def create(self, *args, **kwargs):
        base = super().create(*args, **kwargs)
        if not base.cached:
            raise ValueError("Committed-prefix caching is required")
        state = RetirementState(**vars(base))
        state.cache = AbsoluteKV(state.x.shape[1], len(self.model.model.layers))
        return state

    def _record(self, state, phase, start, end, prefix, *, work=None):
        physical = 0 if work is None else work["equivalent_full_stack_sweeps"]
        state.events.append({
            "phase": phase, "block": state.block, "q_start": start, "q_end": end,
            "q_tokens": 0 if work is None else end-start,
            "prefix_kv_length": prefix, "context_version": state.context_version,
            "q_budget": self.q_budget, "physical_forwards": physical,
            "physical_layer_attention_calls": 0 if work is None else work["physical_attention_calls"],
            "max_physical_q": 0 if work is None else work["max_query_rows"],
            "query_rows_over_layers": 0 if work is None else work["physical_query_rows_over_layers"],
            "layer_shapes": [] if work is None else work["layer_shapes"],
            "denoise_rows": (end-start if phase == "denoise" else
                             32 if phase == "commit_refill" else 0),
            "refresh_rows": 32 if phase in ("commit", "commit_refill") else 0,
            "padding_rows": 0, "prefix_moved": False,
        })

    def _execute(self, state, start, end, phase):
        assert state.cache.length == start
        positions = torch.arange(start, end, device=self.model.device)
        keys = torch.arange(end, device=self.model.device)
        visibility = (positions[:, None]//32 >= keys[None, :]//32)[None, None]
        out = self.tiled(
            state.x[:, start:end], attention_mask=visibility,
            position_ids=positions[None], past_key_values=state.cache,
            use_cache=True, output_attentions=True)
        self._record(state, phase, start, end, start, work=self.tiled.last_work)
        for block in range(start//32, end//32):
            state.kv_status[block] = {
                "status": "PROVISIONAL", "token_revision": state.block_revisions.get(block, 0),
                "context_revision": state.context_version,
            }
        return out.logits

    def _forward(self, state, q_start, q_end, store, phase):
        if phase == "prefill":
            # Prompt construction is reported separately and is outside the
            # generation-Q cap. No candidate generation work uses this route.
            cache = DynamicCache()
            pos = torch.arange(q_end, device=self.model.device)
            mask = (pos[:, None]//32 >= pos[None, :]//32)[None, None]
            out = self.model(state.x[:, :q_end], attention_mask=mask,
                             position_ids=pos[None], past_key_values=cache,
                             use_cache=True, output_attentions=True)
            state.cache = AbsoluteKV.from_pairs(cache.to_legacy_cache(), state.x.shape[1])
            self._record(state, phase, 0, q_end, 0)
            state.events[-1].update(q_tokens=q_end, physical_forwards=1,
                                    max_physical_q=q_end, prompt_only=True)
            return out.logits
        if phase == "commit" and state.seal_ready is not None:
            block, revision, context = state.seal_ready
            assert (block, revision, context) == (
                state.block, state.block_revisions.get(block, 0), state.context_version)
            assert state.cache.length == q_end
            state.kv_status[block]["status"] = "EXACT"
            state.seal_ready = None
            self._record(state, "seal_reuse", q_start, q_end, q_start)
            return None
        if phase == "denoise" and state.pending_refill is not None:
            pending = state.pending_refill
            assert pending["start"] == q_start and pending["end"] == q_end
            assert pending["context_version"] == state.context_version
            assert torch.equal(pending["input"], state.x[:, q_start:q_end])
            assert state.cache.length == q_start
            # The fused call already wrote these exact input-state KV rows.
            state.cache.length = q_end
            state.pending_refill = None
            self._record(state, "refill_logits_reuse", q_start, q_end, q_start)
            return pending["logits"]
        fuse = (phase == "commit" and self.fuse_refresh and
                q_end+32 <= state.x.shape[1] and
                not any(bool((state.x[:, state.prompt_length:q_end] == e).any())
                        for e in state.eos_ids))
        if fuse:
            logits = self._execute(state, q_start, q_end+32, "commit_refill")
            state.pending_refill = {
                "start": q_end, "end": q_end+32,
                "context_version": state.context_version+1,
                "input": state.x[:, q_end:q_end+32].clone(),
                "logits": logits[:, 32:].clone(),
            }
            state.cache.crop(q_end)
            state.kv_status[state.block]["status"] = "EXACT"
            return logits[:, :32]
        logits = self._execute(state, q_start, q_end, phase)
        if store:
            state.kv_status[state.block]["status"] = "EXACT"
        return logits

    @torch.inference_mode()
    def step(self, state):
        action = super().step(state)
        if action == "denoise":
            event = state.events[-1]
            assert event["phase"] in ("denoise", "refill_logits_reuse")
            changed = event["accepted"] + event["edited"]
            if changed:
                state.block_revisions[state.block] = state.block_revisions.get(state.block, 0)+1
                state.kv_status[state.block]["status"] = "DIRTY"
            if self.seal and state.phase == "commit" and not changed:
                state.seal_ready = (
                    state.block, state.block_revisions.get(state.block, 0), state.context_version)
                event["seal_eligible"] = True
            else:
                state.cache.crop(state.block*32)
        return action

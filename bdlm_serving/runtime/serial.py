"""Request-state reference, not an optimized serving engine.

One step performs at most one forward. Only final blocks enter immutable KV.
No speculative future work or intra-block cache freezing is implemented here.
"""
import copy
import inspect
import textwrap
import types
from dataclasses import dataclass, field

import torch
from transformers.cache_utils import DynamicCache
from bdlm_serving.runtime.rng import REFERENCE_RNG_LOCK


def install_llada_rectangular_mask(model, correct_binary_mask=True, allow_selected_cache=False):
    """Private copy accepting rectangular masks and explicit block causality.

The installed Transformers 4.57.1 helper passes 4D floats through as additive
bias. This checkpoint supplies binary 0/1 floats. Convert explicitly instead
of treating the checkpoint/environment combination as a valid BDLM reference.
"""
    fn = model.model.forward.__func__
    source = textwrap.dedent(inspect.getsource(fn))
    # The documentation decorator uses inspect.getsource while executing.
    # Reuse its already-expanded docstring instead of applying it to exec code.
    source = source[source.index("def forward("):]
    old = "if attention_mask.size() == (batch_size, 1, seq_length, seq_length):"
    new = ("if attention_mask.size() == "
           "(batch_size, 1, seq_length, seq_length + past_seen_tokens):")
    if allow_selected_cache:
        new = ("if attention_mask.size() == "
               "(batch_size, 1, seq_length, "
               "(past_key_values.visible_end if hasattr(past_key_values, 'selected') "
               "else seq_length + past_seen_tokens)):")
    if source.count(old) != 1:
        raise RuntimeError("Unrecognized checkpoint source; refuse to patch")
    source = source.replace(old, new)
    mask_old = """attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_seen_tokens,
        )"""
    mask_new = """attention_mask = torch.zeros_like(
            attention_mask, dtype=inputs_embeds.dtype
        ).masked_fill(~attention_mask.bool(), torch.finfo(inputs_embeds.dtype).min)"""
    if correct_binary_mask:
        if source.count(mask_old) != 1:
            raise RuntimeError("Unrecognized mask conversion source")
        source = source.replace(mask_old, mask_new)
    namespace = dict(fn.__globals__)
    exec(compile(source, "<isolated_llada_rectangular_mask>", "exec"), namespace)
    namespace["forward"].__doc__ = fn.__doc__
    model.model.forward = types.MethodType(namespace["forward"], model.model)
    return {"old": old, "new": new, "scope": "private model.model.forward only",
            "allow_selected_cache": allow_selected_cache,
            "correct_binary_mask": correct_binary_mask,
            "mask_old": mask_old if correct_binary_mask else None,
            "mask_new": mask_new if correct_binary_mask else None}


@dataclass
class RequestState:
    model_kind: str
    config: dict
    x: torch.Tensor
    prompt_length: int
    gen_length: int
    block: int
    block_length: int
    mask_id: int
    eos_ids: list
    cached: bool
    cache: object
    rng_cpu: torch.Tensor
    rng_cuda: torch.Tensor
    phase: str = "prefill"
    step_in_block: int = 0
    post_steps: int = 0
    committed_end: int = 0
    context_version: int = 0
    events: list = field(default_factory=list)


class SerialKernel:
    def __init__(self, model, model_kind, sdar_module=None):
        self.model = model
        self.kind = model_kind
        self.sdar = sdar_module
        self.last_logits = None
        self.capture_logits = False

    def _rng_devices(self):
        return [self.model.device.index or 0] if self.model.device.type == "cuda" else []

    def create(self, input_ids, config, gen_length, seed, mask_id, eos_ids, cached=True):
        if self.kind == "sdar" and not cached:
            raise ValueError("SDAR reference uses native committed-prefix cache")
        b = 32
        p = input_ids.shape[1]
        total = ((p + gen_length + b - 1) // b) * b
        x = torch.full((1, total), mask_id, device=self.model.device, dtype=torch.long)
        x[:, :p] = input_ids
        # Initial RNG matches torch.manual_seed in native_screen.py, but each
        # request owns its state; stepping it preserves the caller's global RNG.
        with REFERENCE_RNG_LOCK, torch.random.fork_rng(devices=self._rng_devices()):
            torch.manual_seed(seed)
            rng_cpu = torch.get_rng_state().clone()
            rng_cuda = (torch.cuda.get_rng_state(self.model.device).clone()
                        if self._rng_devices() else torch.empty(0, dtype=torch.uint8))
        return RequestState(
            model_kind=self.kind, config=copy.deepcopy(config), x=x, prompt_length=p,
            gen_length=gen_length, block=p // b, block_length=b, mask_id=mask_id,
            eos_ids=list(eos_ids), cached=cached,
            cache=DynamicCache() if cached else None,
            rng_cpu=rng_cpu, rng_cuda=rng_cuda)

    @staticmethod
    def snapshot(state):
        # Reference implementation intentionally clones the cache. This is a
        # correctness boundary, not a claim of low-overhead checkpointing.
        return copy.deepcopy(state)

    def _forward(self, state, q_start, q_end, store, phase):
        prefix = state.cache.get_seq_length() if state.cached else 0
        assert prefix == q_start if state.cached else q_start == 0
        q_positions = torch.arange(q_start, q_end, device=self.model.device)
        k_positions = torch.arange(q_end, device=self.model.device)
        mask = (q_positions[:, None] // 32 >= k_positions[None, :] // 32)
        mask = mask.unsqueeze(0)
        if self.kind == "llada21":
            mask = mask.unsqueeze(0).to(torch.bfloat16)
        else:
            mask = mask.to(torch.float32)
        kw = {
            "attention_mask": mask, "position_ids": q_positions.unsqueeze(0),
            "use_cache": state.cached,
        }
        if state.cached:
            kw["past_key_values"] = state.cache
        if self.kind == "sdar":
            kw["store_kv"] = store
        else:
            kw["output_attentions"] = True
        out = self.model(state.x[:, q_start:q_end], **kw)
        if self.kind == "llada21" and state.cached and not store:
            # Native attention updates DynamicCache on every forward. The
            # tentative block must never become committed cache.
            state.cache.crop(prefix)
        if state.cached:
            assert state.cache.get_seq_length() == (q_end if store else prefix)
        state.events.append({
            "phase": phase, "block": state.block,
            "q_start": q_start, "q_end": q_end, "q_tokens": q_end - q_start,
            "prefix_kv_length": prefix, "store_kv": store,
            "context_version": state.context_version,
        })
        return out.logits

    @torch.inference_mode()
    def step(self, state):
        if state.model_kind != self.kind:
            raise ValueError("Request/model mismatch")
        self.last_logits = None
        if state.phase == "done":
            return "done"
        b = state.block_length
        start, end = state.block * b, (state.block + 1) * b
        if state.phase == "prefill":
            state.phase = "denoise"
            state.committed_end = start
            if state.cached and start:
                self._forward(state, 0, start, True, "prefill")
            return "prefill"
        cur = state.x[:, start:end]
        if state.phase == "commit":
            assert not (cur == state.mask_id).any()
            if state.cached:
                # Cache is computed from FINAL tokens, not the input to the
                # preceding sampling step.
                self._forward(state, start, end, True, "commit")
            state.committed_end = end
            state.context_version += 1
            generated = state.x[:, state.prompt_length:end]
            eos = any(bool((generated == e).any()) for e in state.eos_ids)
            state.block += 1
            state.step_in_block = state.post_steps = 0
            state.phase = "done" if eos or end == state.x.shape[1] else "denoise"
            return "commit"
        assert state.phase == "denoise"
        old = cur.clone()
        masked = cur == state.mask_id
        count = int(masked.sum())
        if self.kind == "sdar" and not count:
            state.phase = "commit"
            return self.step(state)
        if self.kind == "llada21" and not count:
            state.post_steps += 1
            if state.post_steps > state.config.get("max_post_steps", 16):
                state.phase = "commit"
                return self.step(state)
        logits = self._forward(
            state, start if state.cached else 0, end, False, "denoise")[:, -b:, :]
        if self.capture_logits:
            self.last_logits = logits.detach().clone()
        with REFERENCE_RNG_LOCK, torch.random.fork_rng(devices=self._rng_devices()):
            torch.set_rng_state(state.rng_cpu)
            if self._rng_devices():
                torch.cuda.set_rng_state(state.rng_cuda, self.model.device)
            if self.kind == "sdar":
                proposed, confidence = self.sdar.sample_with_temperature_topk_topp(
                    logits, temperature=state.config["temperature"],
                    top_k=state.config["top_k"], top_p=state.config["top_p"])
            else:
                proposed, confidence = self.model._sample_with_temperature_topk_topp(
                    logits, temperature=0.0, top_k=None, top_p=None)
            state.rng_cpu = torch.get_rng_state().clone()
            if self._rng_devices():
                state.rng_cuda = torch.cuda.get_rng_state(self.model.device).clone()
        accept = torch.zeros_like(masked)
        threshold = state.config.get("confidence_threshold", state.config.get("threshold"))
        if count:
            masked_confidence = torch.where(masked, confidence, -torch.inf)
            accept = masked & (masked_confidence > threshold)
            if not accept.any():
                accept[0, torch.topk(masked_confidence[0], k=1).indices] = True
        edit = torch.zeros_like(masked)
        if self.kind == "llada21":
            positions = torch.arange(start, end, device=cur.device).unsqueeze(0)
            eligible = ~masked & (positions >= state.prompt_length)
            edit = eligible & (proposed != old) & (confidence > state.config["editing_threshold"])
        selected = accept | edit
        cur[selected] = proposed[selected]
        state.step_in_block += 1
        state.events[-1].update(
            masked_before=count, accepted=int(accept.sum()), edited=int(edit.sum()),
            masked_after=int((cur == state.mask_id).sum()),
            step_in_block=state.step_in_block, post_steps=state.post_steps)
        if self.kind == "llada21" and not count and not edit.any():
            state.phase = "commit"
        if self.kind == "sdar":
            if not (cur == state.mask_id).any():
                state.phase = "commit"
            elif state.step_in_block >= b:
                raise RuntimeError("Native progress bound failed")
        return "denoise"

    def run(self, state, max_actions=4096):
        for _ in range(max_actions):
            if state.phase == "done":
                return state
            self.step(state)
        raise RuntimeError("Reference action limit exceeded")

    @staticmethod
    def output(state):
        ids = state.x[:, state.prompt_length:state.prompt_length + state.gen_length]
        eos = torch.zeros_like(ids, dtype=torch.bool)
        for e in state.eos_ids:
            eos |= ids == e
        positions = eos[0].nonzero(as_tuple=True)[0]
        return ids[:, :int(positions[0]) + 1] if len(positions) else ids


def cache_tensors(cache):
    if cache is None:
        return []
    return [t for pair in cache.to_legacy_cache() for t in pair]


def compare_states(a, b):
    pairs = list(zip(cache_tensors(a.cache), cache_tensors(b.cache)))
    matching_shapes = all(x.shape == y.shape for x, y in pairs)
    return {
        "tokens_equal": torch.equal(a.x, b.x),
        "cache_tensors_equal": all(torch.equal(x, y) for x, y in pairs),
        "cache_tensor_count_equal": len(cache_tensors(a.cache)) == len(cache_tensors(b.cache)),
        "cache_shapes_equal": matching_shapes,
        "cache_max_abs": (max((float((x - y).abs().max()) for x, y in pairs if x.numel()),
                             default=0.0) if matching_shapes else None),
        "rng_cpu_equal": torch.equal(a.rng_cpu, b.rng_cpu),
        "rng_cuda_equal": torch.equal(a.rng_cuda, b.rng_cuda),
        "control_equal": all(getattr(a, k) == getattr(b, k) for k in
                             ("phase", "block", "step_in_block", "post_steps",
                              "committed_end", "context_version", "events")),
    }

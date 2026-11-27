"""Batched LLaDA2.1 driver for the one-block / selective-history study.

The native contract this module targets is deliberately narrow:

* one physical B32 block per request, all 32 rows are projected and receive
  block-local attention;
* only the selected rows are packed into the history partition
  ``block_history`` call, and every unselected row reuses the per-layer
  ``(m, l, u)`` state it last obtained from an actual history read;
* prompt/committed KV is never re-read as "dense then masked": the adapter's
  history bounds decide which KV columns are legal.

The companion adapter is :class:`bdlm_serving.methods.t2t.history_llada.LLadaHistory`.
It expects the model's attention cache to already contain the current block
after ``past_key_value.update(...)``; ``GroupCache`` below copies the committed
history once per group/epoch and exposes the block at a uniform trailing offset
so the adapter can split ``k[..., :h]`` / ``k[..., h:]``.

No model is executed at import time.  This module is pure request bookkeeping
plus cache assembly so it can be unit-checked on CPU.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable, Optional

import torch
from transformers.cache_utils import DynamicCache

from bdlm_serving.runtime.rng import REFERENCE_RNG_LOCK
from .budget_family import Law, choose


BLOCK = 32
NATIVE_CONFIG = {
    "confidence_threshold": 0.5,
    "editing_threshold": 0.5,
    "max_post_steps": 16,
}


def _rng_devices(model):
    return [model.device.index or 0] if model.device.type == "cuda" else []


def _initial_rng(model, seed):
    with REFERENCE_RNG_LOCK, torch.random.fork_rng(devices=_rng_devices(model)):
        torch.manual_seed(seed)
        cpu = torch.get_rng_state().clone()
        cuda = (torch.cuda.get_rng_state(model.device).clone()
                if _rng_devices(model) else torch.empty(0, dtype=torch.uint8))
    return cpu, cuda


def _percentiles(values):
    values = sorted(float(v) for v in values)
    if not values:
        return dict(count=0, mean=None, p50=None, p95=None, max=None)

    def rank(p):
        return values[max(0, min(len(values) - 1, int(p * len(values) + 0.999999) - 1))]

    return dict(count=len(values), mean=mean(values), p50=rank(0.50),
                p95=rank(0.95), max=values[-1])


@dataclass
class HistoryStore:
    """Committed prompt + completed-block KV for one request.

    Per-layer tensors are ``(1, G, H, D)``.  The store is only appended to by a
    native store forward (prefill or commit); denoise reads never mutate it.
    """

    keys: list = field(default_factory=list)
    values: list = field(default_factory=list)
    length: int = 0
    version: int = 0

    def append(self, key_layers, value_layers):
        if not key_layers:
            raise ValueError("empty layer list")
        if self.length == 0:
            self.keys = [x.clone() for x in key_layers]
            self.values = [x.clone() for x in value_layers]
        else:
            if len(self.keys) != len(key_layers):
                raise ValueError("layer count changed")
            self.keys = [torch.cat((old, new), dim=-2)
                         for old, new in zip(self.keys, key_layers, strict=True)]
            self.values = [torch.cat((old, new), dim=-2)
                           for old, new in zip(self.values, value_layers, strict=True)]
        q = key_layers[0].shape[-2]
        if any(x.shape[-2] != q for x in key_layers + value_layers):
            raise ValueError("ragged layer write")
        self.length += q
        self.version += 1

    @classmethod
    def empty(cls):
        return cls()


class GroupCache(DynamicCache):
    """History KV for one batch, with the current block at ``h``.

    ``h = max(lengths)``.  Each row keeps its own committed history in
    ``[0, H_i)``; the current block is written at the same trailing offset
    ``h`` for every row.  Rows whose ``H_i < h`` therefore have masked padding
    between their history and the block, which is exactly what the adapter's
    ``_bounds`` expects.  With equal lengths (the default exact-H bucketing)
    there is no padding and every key is visible.
    """

    def __init__(self, stores, lengths, h):
        super().__init__()
        self.stores = list(stores)
        self.lengths = [int(x) for x in lengths]
        self.h = int(h)
        self.q = None
        self.key_cache = []
        self.value_cache = []
        self.layer_written = []

    @property
    def padded(self):
        return any(x != self.h for x in self.lengths)

    def get_seq_length(self, layer_idx=0):
        return self.h

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if key_states.ndim != 4 or value_states.shape != key_states.shape:
            raise ValueError("expected (batch, G, q, D) KV")
        n, g, q, d = key_states.shape
        if n != len(self.stores) or n != len(self.lengths):
            raise ValueError("batch/store mismatch")
        if self.q is None:
            self.q = q
        elif q != self.q:
            raise ValueError("ragged q within one forward")
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
            self.layer_written.append(False)
        if self.key_cache[layer_idx] is None:
            # Allocate once per layer per group epoch; later calls overwrite
            # only the block columns.
            shape = (n, g, self.h + q, d)
            key_buf = key_states.new_zeros(shape)
            value_buf = value_states.new_zeros(shape)
            for i, store in enumerate(self.stores):
                length = self.lengths[i]
                if length:
                    if len(store.keys) <= layer_idx:
                        raise ValueError("store missing layer")
                    if store.keys[layer_idx].shape[-2] != length:
                        raise ValueError("store length disagrees")
                    key_buf[i, :, :length, :].copy_(store.keys[layer_idx][0])
                    value_buf[i, :, :length, :].copy_(store.values[layer_idx][0])
            self.key_cache[layer_idx] = key_buf
            self.value_cache[layer_idx] = value_buf
        buf_k, buf_v = self.key_cache[layer_idx], self.value_cache[layer_idx]
        if buf_k.shape[-2] != self.h + q:
            raise ValueError("cache shape changed mid-epoch")
        buf_k[:, :, self.h:self.h + q, :].copy_(key_states)
        buf_v[:, :, self.h:self.h + q, :].copy_(value_states)
        self.layer_written[layer_idx] = True
        return buf_k[:, :, :self.h + q, :], buf_v[:, :, :self.h + q, :]

    def block(self, layer_idx):
        if self.q is None or layer_idx >= len(self.key_cache):
            raise RuntimeError("cache layer was never written")
        if not self.layer_written[layer_idx]:
            raise RuntimeError("cache layer missing block write")
        k = self.key_cache[layer_idx]
        v = self.value_cache[layer_idx]
        return (k[:, :, self.h:self.h + self.q, :],
                v[:, :, self.h:self.h + self.q, :])

    def assert_complete(self, layers):
        if len(self.key_cache) < layers or not all(self.layer_written[:layers]):
            raise RuntimeError("not every decoder layer wrote the cache")
        if len(self.key_cache) != layers:
            raise RuntimeError("extra cache layers")
        if self.q is None:
            raise RuntimeError("cache had no query rows")


@dataclass
class HistoryRequest:
    index: int
    task_id: str
    dataset: str
    seed: int
    x: torch.Tensor
    prompt_length: int
    gen_length: int
    config: dict
    mask_id: int
    eos_ids: list
    block: int
    committed_end: int
    rng_cpu: torch.Tensor
    rng_cuda: torch.Tensor
    phase: str = "prefill"
    step_in_block: int = 0
    post_steps: int = 0
    context_version: int = 0
    store: HistoryStore = field(default_factory=HistoryStore.empty)
    states: Optional[dict] = None
    dirty: set = field(default_factory=set)
    dirty_since: dict = field(default_factory=dict)
    cold: set = field(default_factory=set)
    cold_since: dict = field(default_factory=dict)
    established: set = field(default_factory=set)
    last_read: dict = field(default_factory=dict)
    clock: int = 0
    done: bool = False
    events: list = field(default_factory=list)
    forward_phases: Counter = field(default_factory=Counter)
    q_by_phase: Counter = field(default_factory=Counter)
    generation_forwards: int = 0
    generation_query_rows: int = 0
    prefill_rows: int = 0
    denoise_history_rows: int = 0
    commit_history_rows: int = 0
    history_edges: int = 0
    block_edges: int = 0
    stale_reused: int = 0
    accepted: int = 0
    edited: int = 0
    revoked: int = 0
    forced_accepts: int = 0
    cold_blocked_rounds: int = 0
    commit_with_cold: int = 0
    dirty_wait: list = field(default_factory=list)
    cold_wait: list = field(default_factory=list)
    stale_age: list = field(default_factory=list)
    history_q_per_forward: list = field(default_factory=list)
    decoder_q_per_forward: list = field(default_factory=list)
    batch_sizes: list = field(default_factory=list)
    last_write_clock: list = field(default_factory=lambda: [-1] * BLOCK)

    @property
    def device(self):
        return self.x.device

    @property
    def total(self):
        return self.x.shape[1]

    @property
    def history_length(self):
        return self.committed_end

    def block_start(self):
        return self.block * BLOCK

    def current_block(self):
        start = self.block_start()
        return self.x[:, start:start + BLOCK]

    def output_ids(self):
        ids = self.x[0, self.prompt_length:self.prompt_length + self.gen_length].tolist()
        end = next((i + 1 for i, token in enumerate(ids) if token in self.eos_ids), None)
        return ids[:end] if end is not None else ids


def make_request(model, index, task, prompt_ids, *, generation_cap, config=None):
    """Create one request from frozen prompt IDs and the native protocol."""
    if not prompt_ids or any(type(x) is not int for x in prompt_ids):
        raise ValueError("prompt_ids must be a non-empty int list")
    p = len(prompt_ids)
    total = ((p + generation_cap + BLOCK - 1) // BLOCK) * BLOCK
    mask_id = task["_mask_id"]
    x = torch.full((1, total), mask_id, device=model.device, dtype=torch.long)
    x[:, :p] = torch.tensor([prompt_ids], device=model.device, dtype=torch.long)
    block = p // BLOCK
    cpu, cuda = _initial_rng(model, task["seed"])
    return HistoryRequest(
        index=index, task_id=task["task_id"], dataset=task["dataset"], seed=task["seed"],
        x=x, prompt_length=p, gen_length=generation_cap,
        config=dict(NATIVE_CONFIG if config is None else config),
        mask_id=mask_id, eos_ids=list(task["_eos_ids"]),
        block=block, committed_end=block * BLOCK, rng_cpu=cpu, rng_cuda=cuda,
    )


@dataclass
class Selection:
    selected: list
    masks: set
    dirty_before: set
    cold_before: set
    prefix: tuple
    full_endpoint: bool
    endpoint_same_as_b: bool


def _prefix_candidates(request, masks, established, dirty, cold):
    if not masks:
        return ()
    first_mask = min(masks)
    start = request.block_start()
    return tuple(i for i in range(first_mask - 1, -1, -1)
                 if start + i >= request.prompt_length
                 and i in established and i not in dirty and i not in cold)


def select_rows(request, budget, *, full_endpoint=False, law=Law(), cold_share=0):
    """Return the in-block rows that actually re-read history this forward.

    ``budget`` is the external fixed C.  A full endpoint (C >= B, or the
    explicit ``full`` gate) selects every row and does not use the heuristic.
    With no legal history (H == 0) no history row is selected; the empty
    partition is already the exact model result.
    """
    if type(budget) is not int or budget < 0:
        raise ValueError("C must be a non-negative integer")
    start = request.block_start()
    masks = {i for i, token in enumerate(request.current_block()[0].tolist())
             if token == request.mask_id}
    if request.committed_end == 0:
        return Selection([], masks, set(request.dirty), set(), (),
                         full_endpoint=bool(full_endpoint or budget >= BLOCK),
                         endpoint_same_as_b=bool(budget >= BLOCK))
    dirty = set(request.dirty)
    cold = set(request.cold)
    prefix = _prefix_candidates(request, masks, request.established, dirty, cold)
    if full_endpoint or budget >= BLOCK:
        selected = list(range(BLOCK))
    else:
        if budget == 0:
            selected = []
        else:
            reserve = min(len(cold), (budget * cold_share + 99) // 100)
            due = (set(sorted(dirty, key=lambda j: (request.dirty_since.get(j, 0), j))[:budget-reserve])
                   if cold_share else dirty)
            plan = choose(budget, BLOCK, due, masks, prefix, law=law, cold=cold)
            selected = sorted(plan.rows)
    if (selected != sorted(set(selected)) or any(type(i) is not int or not 0 <= i < BLOCK
                                                 for i in selected)):
        raise RuntimeError("invalid history row selection")
    return Selection(selected, masks, dirty, cold, prefix,
                     full_endpoint=bool(full_endpoint or budget >= BLOCK),
                     endpoint_same_as_b=bool(budget >= BLOCK))


def _empty_state(model, device):
    attention = model.model.layers[0].attention
    heads = int(getattr(attention, "num_heads", model.config.num_attention_heads))
    d = int(getattr(attention, "head_dim", 0)
            or getattr(model.config, "head_dim", 0)
            or (model.config.hidden_size // heads))
    m = torch.full((1, heads, BLOCK), -float("inf"), device=device, dtype=torch.float32)
    l = torch.zeros((1, heads, BLOCK), device=device, dtype=torch.float32)
    u = torch.zeros((1, heads, BLOCK, d), device=device, dtype=torch.float32)
    return m, l, u


def pack_states(model, requests):
    """Per-request state slices -> the adapter's one dict for this batch."""
    layers = len(model.model.layers)
    empty = None
    result = {}
    for layer in range(layers):
        ms, ls, us = [], [], []
        for request in requests:
            state = None if request.states is None else request.states.get(layer)
            if state is None:
                if empty is None:
                    empty = _empty_state(model, request.x.device)
                state = empty
            ms.append(state[0])
            ls.append(state[1])
            us.append(state[2])
        result[layer] = (torch.cat(ms, dim=0), torch.cat(ls, dim=0), torch.cat(us, dim=0))
    return result


def unpack_states(requests, states):
    if sorted(states) != list(range(len(states))):
        raise RuntimeError("adapter returned a non-contiguous layer map")
    for i, request in enumerate(requests):
        request.states = {
            layer: tuple(part[i:i + 1] for part in value)
            for layer, value in states.items()
        }


def build_mask(requests, h, q, *, prefill=False):
    """Return the rectangular binary mask consumed by the patched model forward."""
    n = len(requests)
    if prefill:
        # Prompt prefill: native block-causal visibility over the committed prefix.
        pos = torch.arange(q, device=requests[0].device)
        allowed = (pos[:, None] // BLOCK >= pos[None, :] // BLOCK)
        return allowed[None, None].expand(n, 1, q, q)
    if q != BLOCK:
        raise ValueError("only B32 block forwards use the history mask")
    mask = torch.zeros((n, 1, q, h + q), device=requests[0].device, dtype=torch.bool)
    for i, request in enumerate(requests):
        length = request.committed_end
        if length:
            mask[i, 0, :, :length] = True
        mask[i, 0, :, h:h + q] = True
    return mask


class BatchedHistoryRunner:
    """Drive fixed-C selective history over one batch group at a time."""

    def __init__(self, model, adapter, requests, *, budget, batch_size,
                 bucket="exact", full_endpoint=False, max_rounds=20000, cold_share=0, law=Law()):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if bucket not in ("exact", "pooled"):
            raise ValueError("bucket must be exact or pooled")
        if len(requests) < 1:
            raise ValueError("no requests")
        self.model = model
        self.adapter = adapter
        self.requests = list(requests)
        self.budget = budget
        self.cold_share, self.law = cold_share, law
        self.batch_size = batch_size
        self.bucket = bucket
        self.full_endpoint = full_endpoint
        self.max_rounds = max_rounds
        self.events = []
        self._cache_pool = {}
        self._used_keys = set()
        self._layers = len(model.model.layers)
        self._projections = tuple(layer.attention.query_key_value
                                  for layer in model.model.layers)
        if not self._projections:
            raise ValueError("no native LLaDA QKV projections")

    # -- grouping ---------------------------------------------------------
    def _groups(self, requests, key):
        requests = sorted(requests, key=lambda r: (key(r), r.index))
        if self.bucket == "exact":
            buckets = defaultdict(list)
            for request in requests:
                buckets[key(request)].append(request)
            groups = []
            for value in sorted(buckets):
                bucket = buckets[value]
                groups.extend(bucket[i:i + self.batch_size]
                              for i in range(0, len(bucket), self.batch_size))
            return groups
        return [requests[i:i + self.batch_size]
                for i in range(0, len(requests), self.batch_size)]

    def _cache(self, requests, stage):
        h = max(request.committed_end for request in requests)
        key = (stage, tuple(request.index for request in requests),
               tuple(request.committed_end for request in requests),
               tuple(request.store.version for request in requests), h)
        cache = self._cache_pool.get(key)
        if cache is None:
            cache = GroupCache([request.store for request in requests],
                               [request.committed_end for request in requests], h)
            self._cache_pool[key] = cache
        self._used_keys.add(key)
        return cache, h, key

    # -- forward ----------------------------------------------------------
    def _forward(self, requests, cache, h, q, *, stage, selected=None,
                 cache_states=None):
        if q != BLOCK and stage != "prefill":
            raise ValueError("only prompt prefill may use a non-B query")
        if q == 0:
            raise ValueError("zero-width forward")
        # committed_end is the target prefix length during prefill; that KV
        # does not exist yet. Only later block calls start at committed_end.
        starts = [0 if stage == "prefill" else r.committed_end for r in requests]
        ids = torch.stack([r.x[0, start:start+q] for r,start in zip(requests,starts,strict=True)])
        positions = torch.stack([torch.arange(start,start+q,device=r.device)
                                 for r,start in zip(requests,starts,strict=True)])
        if stage == "prefill" and any(start or q != r.committed_end or q > r.prompt_length
                                      for r,start in zip(requests,starts,strict=True)):
            raise ValueError("prefill must cover exactly the complete prompt blocks")
        embeds = self.model.get_input_embeddings()(ids)
        mask = build_mask(requests, h, q, prefill=stage == "prefill")
        seen = [0] * self._layers
        widths = [None] * self._layers
        hooks = []
        for i, projection in enumerate(self._projections):
            def hook(_module, inputs, layer=i):
                shape = inputs[0].shape
                if len(shape) != 3 or seen[layer]:
                    raise RuntimeError(f"layer {layer} projection shape {tuple(shape)}")
                seen[layer] += 1
                widths[layer] = int(shape[0] * shape[1])
            hooks.append(projection.register_forward_pre_hook(hook))
        try:
            if stage == "denoise":
                if selected is None:
                    raise ValueError("denoise needs selected rows")
                with self.adapter.activate(selected, cache_states=cache_states) as scope:
                    out = self.model(
                        inputs_embeds=embeds, position_ids=positions,
                        attention_mask=mask, past_key_values=cache,
                        use_cache=True, output_attentions=True)
                states = scope.states
            else:
                out = self.model(
                    inputs_embeds=embeds, position_ids=positions,
                    attention_mask=mask, past_key_values=cache,
                    use_cache=True, output_attentions=True)
                states = None
        finally:
            for hook in hooks:
                hook.remove()
        if any(x != 1 for x in seen) or any(w != len(requests) * q for w in widths):
            raise RuntimeError(f"Q audit mismatch: seen={seen}, widths={widths}")
        if out.logits.ndim != 3 or out.logits.shape[:2] != (len(requests), q):
            raise RuntimeError("model did not return one logit row per physical query")
        cache.assert_complete(self._layers)
        return out.logits, [len(requests) * q] * self._layers, states

    # -- native sampling --------------------------------------------------
    def _sample(self, request, logits):
        with REFERENCE_RNG_LOCK, torch.random.fork_rng(devices=_rng_devices(self.model)):
            torch.set_rng_state(request.rng_cpu)
            if _rng_devices(self.model):
                torch.cuda.set_rng_state(request.rng_cuda, self.model.device)
            proposed, confidence = self.model._sample_with_temperature_topk_topp(
                logits, temperature=0.0, top_k=None, top_p=None)
            request.rng_cpu = torch.get_rng_state().clone()
            if _rng_devices(self.model):
                request.rng_cuda = torch.cuda.get_rng_state(self.model.device).clone()
        if proposed.shape != logits.shape[:2] or confidence.shape != proposed.shape:
            raise RuntimeError("native sampler returned an invalid shape")
        if not bool(torch.isfinite(confidence).all()):
            raise RuntimeError("nonfinite native confidence")
        return proposed, confidence

    def _apply_sample(self, request, logits, usable):
        proposed, confidence = self._sample(request, logits)
        start = request.block_start()
        cur = request.x[:, start:start + BLOCK]
        before = cur[0].tolist()
        old = cur.clone()
        masked = old == request.mask_id
        positions = torch.arange(start, start + BLOCK, device=request.device)
        usable_t = torch.tensor([i in usable for i in range(BLOCK)],
                                device=request.device, dtype=torch.bool)[None]
        threshold = request.config.get("confidence_threshold",
                                       request.config.get("threshold"))
        edit_threshold = request.config["editing_threshold"]
        accept = masked & usable_t & (confidence > threshold)
        forced = False
        if bool((masked & usable_t).any()) and not bool(accept.any()):
            order = confidence.masked_fill(~(masked & usable_t), -torch.inf)
            accept[0, torch.topk(order[0], k=1).indices] = True
            forced = True
        editable = ~masked & usable_t & (positions[None] >= request.prompt_length)
        edit = editable & (proposed != old) & (confidence > edit_threshold)
        write = accept | edit
        cur[write] = proposed[write]
        after = cur[0].tolist()
        written = [i for i in range(BLOCK) if before[i] != after[i]]
        admitted = int((write & masked).sum())
        edited = int((write & ~masked).sum())
        revoked = int((write & (proposed == request.mask_id)).sum())
        blocked = sorted(i for i in range(BLOCK) if i not in usable and int(masked[0, i]))
        request.accepted += admitted
        request.edited += edited
        request.revoked += revoked
        request.forced_accepts += int(forced)
        return dict(
            masked_before=int(masked.sum()),
            masked_after=int((cur == request.mask_id).sum()),
            selected_masked_before=int((masked & usable_t).sum()),
            accepted=admitted, edited=edited, revoked=revoked,
            forced_accept=forced, blocked_cold_masks=blocked,
            written=written, before=before, after=after,
            proposed=proposed[0].tolist(), confidence=confidence[0].tolist(),
        )

    # -- one action -------------------------------------------------------
    def _next_action(self, request):
        if request.phase == "prefill":
            return "prefill"
        if request.phase == "commit":
            return "commit"
        if request.phase != "denoise":
            raise RuntimeError(f"unknown phase {request.phase}")
        count = int((request.current_block() == request.mask_id).sum())
        if count == 0:
            if request.post_steps + 1 > request.config.get("max_post_steps", 16):
                request.phase = "commit"
                return "commit"
            request.post_steps += 1
        return "denoise"

    def _record_batch(self, event):
        self.events.append(event)
        return len(self.events) - 1

    def _prefill(self, group):
        h = group[0].committed_end
        if any(request.committed_end != h for request in group):
            raise RuntimeError("prefill group must have one history length")
        if h == 0:
            for request in group:
                request.phase = "denoise"
                request.established = set(range(BLOCK))
                request.cold = set()
            return
        cache = GroupCache([HistoryStore.empty() for _ in group], [0] * len(group), 0)
        logits, layer_q, _ = self._forward(group, cache, 0, h, stage="prefill")
        for i, request in enumerate(group):
            keys = [cache.key_cache[layer][i:i + 1, :, :h, :].clone()
                    for layer in range(self._layers)]
            values = [cache.value_cache[layer][i:i + 1, :, :h, :].clone()
                      for layer in range(self._layers)]
            request.store = HistoryStore()
            request.store.append(keys, values)
            request.prefill_rows += h
            request.forward_phases["prefill"] += 1
            request.q_by_phase["prefill"] += h
            request.phase = "denoise"
            request.established = set()
            request.cold = set(range(BLOCK))
            request.cold_since = {i: request.clock for i in request.cold}
        trace = self._record_batch(dict(
            phase="prefill", stage="prefill", batch=[r.index for r in group],
            batch_size=len(group), q=layer_q[0] // len(group), h=0,
            decoder_q=layer_q[0], physical_q=layer_q[0], store_kv=True,
            padded=False, history_rows=0, history_edges=0, block_edges=0,
        ))
        for request in group:
            request.events.append(dict(
                phase="prefill", stage="prefill", block=request.block,
                q=h, decoder_q=h, history_q=0, store_kv=True,
                batch=[r.index for r in group], batch_size=len(group),
                trace=trace, prompt_prefill=True,
            ))

    def _denoise(self, group):
        selected, usable, selections = [], [], []
        for request in group:
            selection = select_rows(request, self.budget,
                                    full_endpoint=self.full_endpoint, cold_share=self.cold_share,
                                    law=self.law)
            selections.append(selection)
            selected.append(selection.selected)
            cold_after = set(request.cold) - set(selection.selected)
            usable.append({i for i in range(BLOCK) if i not in cold_after})
        states = pack_states(self.model, group)
        cache, h, key = self._cache(group, "denoise")
        logits, layer_q, states_out = self._forward(
            group, cache, h, BLOCK, stage="denoise",
            selected=selected, cache_states=states)
        unpack_states(group, states_out)
        sampling = []
        for i, request in enumerate(group):
            selection = selections[i]
            sel = set(selection.selected)
            best_before = set(request.established)
            # History-visible rows may sample this round.  Cold rows with H>0
            # that were not read are barred from both MASK admission and T2T.
            usable_i = usable[i]
            result = self._apply_sample(request, logits[i:i + 1], usable_i)
            sampling.append(result)
            clock = request.clock
            if request.committed_end > 0:
                selected_dirty = (selection.dirty_before & sel)
                selected_cold = (selection.cold_before & sel)
                for row in selected_dirty:
                    request.dirty_wait.append(clock - request.dirty_since.get(row, clock))
                for row in selected_cold:
                    request.cold_wait.append(clock - request.cold_since.get(row, clock))
                stale = [row for row in best_before if row not in sel]
                request.stale_reused += len(stale)
                for row in stale:
                    if row in request.last_read:
                        request.stale_age.append(clock - request.last_read[row])
            request.dirty.difference_update(sel)
            for row in sel:
                request.dirty_since.pop(row, None)
            for row in selection.selected:
                request.established.add(row)
                request.cold.discard(row)
                request.cold_since.pop(row, None)
                request.last_read[row] = clock
            for row in result["written"]:
                request.dirty.add(row)
                request.dirty_since[row] = clock + 1
                request.last_write_clock[row] = clock
            if usable_i == set(range(BLOCK)):
                pass
            else:
                request.cold_blocked_rounds += 1
            request.clock += 1
            request.step_in_block += 1
            count_before = result["masked_before"]
            if count_before == 0 and result["edited"] == 0:
                request.phase = "commit"
            request.generation_forwards += 1
            request.generation_query_rows += BLOCK
            request.forward_phases["denoise"] += 1
            request.q_by_phase["denoise"] += BLOCK
            request.history_q_per_forward.append(len(selection.selected))
            request.decoder_q_per_forward.append(BLOCK)
            request.batch_sizes.append(len(group))
            rows_sel = len(selection.selected)
            request.denoise_history_rows += rows_sel
            edges = sum(request.committed_end for _ in range(rows_sel))
            request.history_edges += edges
            request.block_edges += BLOCK * BLOCK
        masks_before = sum(sampling[i]["masked_before"] for i in range(len(group)))
        stage = "sample" if masks_before else (
            "drain" if any(selections[i].selected and not selections[i].masks
                           for i in range(len(group))) else "verify")
        trace = self._record_batch(dict(
            phase="denoise", stage=stage, batch=[r.index for r in group],
            batch_size=len(group), q=BLOCK, h=h, padded=any(
                r.committed_end != h for r in group),
            nominal_C=self.budget, gate_full_endpoint=self.full_endpoint,
            endpoint_same_as_b=bool(self.budget >= BLOCK),
            decoder_q=sum(layer_q) // self._layers,
            layer_q=list(layer_q),
            history_lengths={r.index: r.committed_end for r in group},
            selected={r.index: selections[i].selected
                      for i, r in enumerate(group)},
            cold_before={r.index: sorted(selections[i].cold_before)
                         for i, r in enumerate(group)},
            dirty_before={r.index: sorted(selections[i].dirty_before)
                          for i, r in enumerate(group)},
            prefix={r.index: list(selections[i].prefix)
                    for i, r in enumerate(group)},
            masks={r.index: sorted(selections[i].masks)
                   for i, r in enumerate(group)},
            history_rows=sum(len(x) for x in selected),
            history_edges=sum(r.committed_end * len(selections[i].selected)
                              for i, r in enumerate(group)),
            block_edges=len(group) * BLOCK * BLOCK,
            full_endpoint_by_request={r.index: selections[i].full_endpoint
                                      for i, r in enumerate(group)},
            sampling={r.index: {k: sampling[i][k] for k in (
                "masked_before", "masked_after", "accepted", "edited",
                "revoked", "forced_accept", "blocked_cold_masks")}
                for i, r in enumerate(group)},
        ))
        for i, request in enumerate(group):
            selection = selections[i]
            request.events.append(dict(
                phase="denoise", stage=stage, block=request.block,
                context_version=request.context_version, q=BLOCK,
                decoder_q=BLOCK, history_q=len(selection.selected),
                history_rows=len(selection.selected),
                history_edges=sum(request.committed_end
                                  for _ in range(len(selection.selected))),
                block_edges=BLOCK * BLOCK,
                selected=selection.selected,
                cold_before=sorted(selection.cold_before),
                dirty_before=sorted(selection.dirty_before),
                prefix=list(selection.prefix), masks=sorted(selection.masks),
                cold_after=sorted(request.cold),
                dirty_after=sorted(request.dirty),
                established=sorted(request.established),
                batch=[r.index for r in group], batch_size=len(group),
                padded=any(r.committed_end != h for r in group), h=h,
                trace=trace, full_endpoint=selection.full_endpoint,
                endpoint_same_as_b=selection.endpoint_same_as_b,
                nominal_C=self.budget,
                **{k: sampling[i][k] for k in (
                    "masked_before", "masked_after", "accepted", "edited",
                    "revoked", "forced_accept", "blocked_cold_masks")},
            ))

    def _commit(self, group):
        h = max(request.committed_end for request in group)
        if any(int((request.current_block() == request.mask_id).sum()) for request in group):
            raise RuntimeError("refusing to commit a block with MASK")
        had_cold = {request.index: bool(request.cold) for request in group}
        cache, h, key = self._cache(group, "commit")
        logits, layer_q, _ = self._forward(group, cache, h, BLOCK, stage="commit")
        for i, request in enumerate(group):
            if request.cold:
                request.commit_with_cold += 1
            keys, values = [], []
            for layer in range(self._layers):
                block_k, block_v = cache.block(layer)
                keys.append(block_k[i:i + 1].clone())
                values.append(block_v[i:i + 1].clone())
            old_length = request.committed_end
            if request.store.length != old_length:
                raise RuntimeError("store length disagrees with committed prefix")
            request.store.append(keys, values)
            request.generation_forwards += 1
            request.generation_query_rows += BLOCK
            request.commit_history_rows += BLOCK
            request.history_edges += BLOCK * old_length
            request.block_edges += BLOCK * BLOCK
            request.history_q_per_forward.append(BLOCK)
            request.decoder_q_per_forward.append(BLOCK)
            request.batch_sizes.append(len(group))
            request.forward_phases["commit"] += 1
            request.q_by_phase["commit"] += BLOCK
            request.context_version += 1
            request.committed_end = old_length + BLOCK
            request.block += 1
            request.step_in_block = 0
            request.post_steps = 0
            request.states = None
            request.established = set()
            request.dirty = set()
            request.dirty_since = {}
            request.last_read = {}
            request.clock = 0
            generated = request.x[:, request.prompt_length:request.committed_end]
            eos = any(bool((generated == e).any()) for e in request.eos_ids)
            if eos or request.committed_end == request.total:
                request.phase = "done"
                request.done = True
                request.cold = set()
            else:
                request.phase = "denoise"
                if request.committed_end > 0:
                    request.cold = set(range(BLOCK))
                    request.cold_since = {i: 0 for i in request.cold}
                else:
                    request.cold = set()
        trace = self._record_batch(dict(
            phase="commit", stage="commit", batch=[r.index for r in group],
            batch_size=len(group), q=BLOCK, h=h,
            padded=any(r.committed_end != h for r in group),
            decoder_q=sum(layer_q) // self._layers, layer_q=list(layer_q),
            history_lengths={r.index: r.committed_end for r in group},
            history_rows=BLOCK * len(group),
            history_edges=sum(BLOCK * r.committed_end for r in group),
            block_edges=len(group) * BLOCK * BLOCK,
            store_kv=True, full_read=True,
        ))
        for request in group:
            request.events.append(dict(
                phase="commit", stage="commit", block=request.block - 1,
                context_version=request.context_version - 1, q=BLOCK,
                decoder_q=BLOCK, history_q=BLOCK, history_rows=BLOCK,
                history_edges=BLOCK * (request.committed_end - BLOCK),
                block_edges=BLOCK * BLOCK, store_kv=True, full_read=True,
                batch=[r.index for r in group], batch_size=len(group),
                padded=any(r.committed_end != h for r in group), trace=trace,
                commit_with_cold=had_cold[request.index],
            ))

    # -- loop -------------------------------------------------------------
    def run(self):
        active = [request for request in self.requests if not request.done]
        for _round in range(self.max_rounds):
            if not active:
                return self.requests
            actions = {"prefill": [], "denoise": [], "commit": []}
            for request in active:
                actions[self._next_action(request)].append(request)
            self._used_keys = set()
            if actions["prefill"]:
                buckets = defaultdict(list)
                for request in actions["prefill"]:
                    buckets[request.committed_end].append(request)
                for h in sorted(buckets):
                    bucket = sorted(buckets[h], key=lambda r: r.index)
                    for i in range(0, len(bucket), self.batch_size):
                        self._prefill(bucket[i:i + self.batch_size])
            if actions["denoise"]:
                for group in self._groups(actions["denoise"],
                                          key=lambda r: r.committed_end):
                    self._denoise(group)
            if actions["commit"]:
                for group in self._groups(actions["commit"],
                                          key=lambda r: r.committed_end):
                    self._commit(group)
            # Drop groups that did not run this round; they are either stale
            # or belong to requests that changed phase.
            self._cache_pool = {key: value for key, value in self._cache_pool.items()
                                if key in self._used_keys}
            active = [request for request in active if not request.done]
        raise RuntimeError("history runner action limit exceeded")


def package_request(request, tokenizer, special_ids):
    ids = request.output_ids()
    eos_found = any(token in request.eos_ids for token in ids)
    n = sum(token not in special_ids for token in ids)
    f = request.generation_forwards
    q = request.generation_query_rows
    if f <= 0 or q != BLOCK * f:
        raise RuntimeError(f"invalid generation work for request {request.index}: {(f, q)}")
    return dict(
        status="success",
        index=request.index,
        task_id=request.task_id,
        dataset=request.dataset,
        seed=request.seed,
        original30=request.index < 30,
        output_ids=ids,
        output_text=tokenizer.decode(ids, skip_special_tokens=True),
        output_tokens=n,
        generation_forwards=f,
        generation_query_rows=q,
        generation_physical_padded_rows=q,
        max_q=BLOCK,
        tpf=n / f,
        token_efficiency=n / q,
        eos=eos_found,
        truncated=not eos_found,
        prompt_tokens=request.prompt_length,
        prefill_rows=request.prefill_rows,
        forwards_by_phase=dict(request.forward_phases),
        q_by_phase=dict(request.q_by_phase),
        q_histogram={str(BLOCK): f},
        history_rows=request.denoise_history_rows + request.commit_history_rows,
        denoise_history_rows=request.denoise_history_rows,
        commit_history_rows=request.commit_history_rows,
        history_q_per_forward=request.history_q_per_forward,
        decoder_q_per_forward=request.decoder_q_per_forward,
        history_kv_edges=request.history_edges,
        block_attention_edges=request.block_edges,
        stale_reused_rows=request.stale_reused,
        accepted_tokens=request.accepted,
        edited_tokens=request.edited,
        revoked_tokens=request.revoked,
        forced_accepts=request.forced_accepts,
        cold_blocked_rounds=request.cold_blocked_rounds,
        commit_with_cold=request.commit_with_cold,
        dirty_wait=request.dirty_wait,
        cold_wait=request.cold_wait,
        stale_age=request.stale_age,
        batch_sizes=request.batch_sizes,
        events=request.events,
    )


def summarize(rows):
    if not rows:
        return {}

    def block(rows):
        if not rows:
            return dict(requests=0)
        n = sum(row["output_tokens"] for row in rows)
        f = sum(row["generation_forwards"] for row in rows)
        q = sum(row["generation_query_rows"] for row in rows)
        return dict(
            requests=len(rows),
            correct=None,
            macro_tpf=mean(row["tpf"] for row in rows),
            pooled_token_efficiency=n / q if q else None,
            output_tokens=n,
            generation_forwards=f,
            generation_query_rows=q,
            prefill_rows=sum(row["prefill_rows"] for row in rows),
            history_rows=sum(row["history_rows"] for row in rows),
            history_kv_edges=sum(row["history_kv_edges"] for row in rows),
            block_attention_edges=sum(row["block_attention_edges"] for row in rows),
            history_q_per_forward=_percentiles(
                value for row in rows for value in row["history_q_per_forward"]),
            decoder_q_per_forward=_percentiles(
                value for row in rows for value in row["decoder_q_per_forward"]),
            stale_reused_rows=sum(row["stale_reused_rows"] for row in rows),
            dirty_wait=_percentiles(v for row in rows for v in row["dirty_wait"]),
            cold_wait=_percentiles(v for row in rows for v in row["cold_wait"]),
            stale_age=_percentiles(v for row in rows for v in row["stale_age"]),
            batch_sizes=dict(Counter(v for row in rows for v in row["batch_sizes"])),
            eos=sum(bool(row["eos"]) for row in rows),
            truncated=sum(bool(row["truncated"]) for row in rows),
        )

    return dict(all=block(rows),
                original30=block([row for row in rows if row["original30"]]),
                added60=block([row for row in rows if not row["original30"]]))

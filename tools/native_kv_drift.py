"""Per-position K/V drift diagnostic on the native frozen30 DiffusionGemma run.

This is an observational pass over the *native* (dense) trajectory.  It never
adds a forward pass and never draws a random number: the existing
``run_native`` call, its per-task seed, sampler and accounting are reused
verbatim.  A thin SDPA attention alias captures the tensors that the decoder
self-attention actually consumes, after ``k_norm`` + RoPE for K and after
``v_norm`` for V, at one selected decoder layer.  The alias delegates to the
pinned SDPA implementation unchanged.

For every consecutive pair of denoising steps inside the same canvas, the
diagnostic computes row-wise relative RMS L2 drift

    rms(K_t[i] - K_{t-1}[i]) / rms(K_{t-1}[i])

(and the same for V).  Only the non-fresh resolved pool visible at the
previous step is ranked.  ``hard_frontier`` scores are computed from the
previous step's ``last_write`` / ``last_query`` state with the production
``frontier._frontier_scores`` function.  The top-10% (or explicit top-k,
tie-aware) drift set is compared with the top-k frontier set; enrichment is
the observed intersection divided by the uniform expectation.  Each selected
position is classified as a true token-to-token edit (``t2t``), a pure
context change (``context``), or a revocation (``revoke``).

Non-fresh pool and cross-step phase
-----------------------------------
At the end of step ``t-1`` the diagnostic knows ``resolved`` and ``fresh``.
The pool for step ``t`` is ``resolved[i] and i not in fresh``: the non-fresh
resolved pool.  The frontier score uses only ``last_write`` and
``last_query`` produced by step ``t-1``; no current-step outcome is read.
Both clocks are *increasing* step ids (1, 2, ...), never the native
countdown ``cur_step``.  The raw countdown is kept only for validation and
audit.  The first step of every canvas has no same-canvas predecessor and is
skipped.  The drift criterion is the pre-registered joint
``max(relative_rms(dK), relative_rms(dV))``; K and V are both reported.
Relative drift below ``DRIFT_FLOOR`` is treated as no drift, so an
all-zero/near-zero step cannot create deterministic false positives.

Only small sufficient statistics are written: per-step counts, top positions,
per-position aggregates and per-task summaries.  No K/V activations or large
tensors are serialized.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
from contextlib import ExitStack, contextmanager
from copy import copy
from dataclasses import dataclass
from pathlib import Path

import torch

from bdlm_serving.evaluation.frozen import load_tasks
from bdlm_serving.methods.t2t.frontier import _frontier_scores, select_hard
from bdlm_serving.methods.t2t.gemma_sparse import _advance, _hard_touch
from bdlm_serving.methods.t2t.planner import History
from bdlm_serving.methods.t2t.trace import gemma_trace
from bdlm_serving.runtime.diffusiongemma import run_native

DEFAULT_LAYER = 15
DEFAULT_TOP_FRACTION = 0.10
DEFAULT_DRIFT_FLOOR = 1e-6
JOINT = "max_dk_dv"
ATTENTION_ALIAS = "bdlm_native_kv_drift_sdpa"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COVERAGE_OUT = REPO_ROOT / "experiments/t2t/placement/native-kv"
EPS = 1e-12
EVENTS = ("t2t_old", "t2t_fresh", "context", "revoke", "admit", "fresh_same", "mask_stay")
COVERAGE_STAGE = "dense_prestate_nonfresh_steps_only"
COVERAGE_ASSUMPTION = (
    "assume every non-fresh step is selectable (假设这些有非 fresh 的步均可选位); "
    "offline dense-prestate coverage only")
COVERAGE_DISCLAIMER = (
    "offline coverage diagnostic only; not an actual sparse verify, sparse rollout, "
    "or online quality measurement (不声称实际 sparse verify/在线质量)")


@dataclass
class PrevState:
    block: int
    clock: int
    step_id: int
    after: list
    k: torch.Tensor
    v: torch.Tensor
    last_write: list
    last_query: list


class CaptureStore:
    """Per-step K/V landing area; reset before every denoising step."""

    def __init__(self):
        self.active = False
        self.k = {}
        self.v = {}
        self.seen = {}

    def reset(self):
        self.k.clear()
        self.v.clear()
        self.seen.clear()


def relative_rms(current, previous, eps=EPS):
    """Row-wise relative RMS L2 drift for a batch-one [B, H, T, D] tensor."""
    if current.shape != previous.shape or current.shape[0] != 1:
        raise ValueError("Expected matching batch-one K/V tensors")
    cur = current.float()
    prev = previous.float()
    numerator = (cur - prev).pow(2).mean(dim=(1, 3))
    denominator = prev.pow(2).mean(dim=(1, 3)).clamp_min(eps)
    return (numerator / denominator).sqrt()[0].tolist()


def top_set(values, pool, fraction=DEFAULT_TOP_FRACTION, k=None,
            floor=0.0, tie_aware=True):
    """Return ``(selected, k, cutoff, tie_count, expanded)``.

    ``k`` is the nominal seat count derived from the *full* pool.  Positions
    below ``floor`` are not positives and cannot enter ``selected``.  If the
    cutoff is above the floor and tied, every positive tied at the cutoff is
    included.  ``tie_aware=False`` keeps exactly ``k`` positions with a
    position tie-break; the prediction end must always use this mode.
    """
    pool = list(pool)
    if not pool:
        return [], 0, None, 0, False
    if k is None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("top fraction must be in (0, 1]")
        k = max(1, math.ceil(fraction * len(pool)))
    k = min(int(k), len(pool))
    if k <= 0:
        raise ValueError("top-k must be positive")
    positives = [i for i in pool if values[i] >= floor]
    if not positives:
        return [], k, None, 0, False
    ordered = sorted(positives, key=lambda i: (-values[i], i))
    take = min(k, len(ordered))
    cutoff = values[ordered[take - 1]]
    tolerance = 1e-12 * max(1.0, abs(cutoff))
    tie_count = sum(1 for i in positives if abs(values[i] - cutoff) <= tolerance)
    selected = ordered[:take]
    expanded = False
    if tie_aware and cutoff > floor and tie_count > 1:
        selected = [i for i in ordered if values[i] >= cutoff - tolerance]
        expanded = len(selected) > take
    return selected, k, cutoff, tie_count, expanded


def classify(prev_resolved, prev_fresh, before, after, accepted, i):
    """Classify one position's transition; ``t2t_old`` is the true T2T."""
    if prev_resolved[i]:
        if not accepted[i]:
            return "revoke"
        if before[i] != after[i]:
            return "t2t_fresh" if i in prev_fresh else "t2t_old"
        return "fresh_same" if i in prev_fresh else "context"
    return "admit" if accepted[i] else "mask_stay"


def enrichment(pool, top, selected):
    """Hard-frontier selection enrichment against the uniform expectation."""
    pool = sorted(pool)
    top = list(top)
    selected = list(selected)
    if not pool or not top or not selected:
        return dict(pool_size=len(pool), top_count=len(top),
                    selected_count=len(selected), hits=0, precision=None,
                    recall=None, base_rate=0.0 if not top else None,
                    uniform_expected=0.0, lift=None,
                    uniform_positional_hits=0)
    hits = len(set(top) & set(selected))
    precision = hits / len(selected)
    recall = hits / len(top)
    base_rate = len(top) / len(pool)
    expected = len(selected) * base_rate
    return dict(
        pool_size=len(pool), top_count=len(top), selected_count=len(selected),
        hits=hits, precision=precision, recall=recall, base_rate=base_rate,
        uniform_expected=expected, lift=hits / expected if expected > 0 else None,
        uniform_positional_hits=len(set(top) & set(pool[:len(selected)])),
    )


class DriftState:
    """Exact native state machine plus one drift record per same-canvas step."""

    def __init__(self, width, layer, top_fraction=DEFAULT_TOP_FRACTION, top_k=None,
                 drift_floor=DEFAULT_DRIFT_FLOOR, *, lag_joint=False):
        if width <= 0:
            raise ValueError("canvas width must be positive")
        if not 0.0 < float(drift_floor) < float("inf"):
            raise ValueError("drift floor must be positive and finite")
        self.width = int(width)
        self.layer = int(layer)
        self.fraction = float(top_fraction)
        self.top_k = None if top_k is None else int(top_k)
        self.drift_floor = float(drift_floor)
        self.block = None
        self.clock = None
        self.step_id = 0
        self.before = None
        self.h = None
        self.last_write = None
        self.last_query = None
        self.prev = None
        self.records = []
        self.lag_frames = [] if lag_joint else None
        self.empty_pool_steps = 0
        self.pos = [dict(pool=0, top=0, selected=0, dk_sum=0.0, dv_sum=0.0,
                         joint_sum=0.0, dk_max=0.0, dv_max=0.0, joint_max=0.0,
                         score_sum=0.0, top_dk_sum=0.0, top_dv_sum=0.0,
                         top_joint_sum=0.0,
                         **{event: 0 for event in EVENTS})
                    for _ in range(self.width)]

    def begin(self, block, clock, before):
        block, clock = int(block), int(clock)
        before = list(before)
        if len(before) != self.width:
            raise ValueError("canvas width mismatch")
        if clock < 1:
            raise ValueError("denoising clock must be positive")
        new_block = self.block != block
        if new_block:
            if self.block is not None and block != self.block + self.width:
                raise RuntimeError("Unexpected canvas advance or second request")
            self.h = History([False] * self.width, [None] * self.width,
                             [0] * self.width, set(), set())
            self.last_write = [-1] * self.width
            self.last_query = [-1] * self.width
            self.prev = None
            self.step_id = 0
        elif self.prev is None:
            raise RuntimeError("Missing previous step")
        elif clock != self.prev.clock - 1:
            raise RuntimeError("Non-native denoising clock")
        self.step_id += 1
        self.block, self.clock, self.before = block, clock, before

    def end(self, after, accepted, k, v, seen=1):
        after = list(after)
        accepted = list(accepted)
        if len(after) != self.width or len(accepted) != self.width:
            raise ValueError("canvas width mismatch")
        if seen != 1:
            raise RuntimeError(
                f"selected layer saw {seen} decoder forwards; expected exactly one")
        if k is None or v is None:
            raise RuntimeError("selected layer did not capture K/V")
        if self.prev is not None:
            if self.prev.block != self.block:
                raise RuntimeError("Canvas changed without reset")
            if self.prev.after != self.before:
                raise RuntimeError("Canvas before-state is not the previous after-state")
            if self.clock != self.prev.clock - 1:
                raise RuntimeError("Non-native denoising clock")
            self._record(k, v, after, accepted)
        previous_h = self.h
        self.h, _ = _advance(self.h, self.before, after, accepted,
                             range(self.width), [None] * self.width)
        _hard_touch(self.last_write, self.last_query, range(self.width),
                    accepted, self.before, after, previous_h.resolved, self.step_id)
        self.prev = PrevState(self.block, self.clock, self.step_id, after, k, v,
                              list(self.last_write), list(self.last_query))

    def _record(self, k, v, after, accepted):
        prev = self.prev
        # Non-fresh resolved pool at the end of step t-1: resolved, but not
        # written/admitted on that step.  The score below sees only the
        # previous step's write/query history (increasing step ids).
        pool = [i for i in range(self.width)
                if self.h.resolved[i] and i not in self.h.fresh]
        if not pool:
            self.empty_pool_steps += 1
            if self.lag_frames is None:
                return
        dk = relative_rms(k, prev.k)
        dv = relative_rms(v, prev.v)
        joint = [max(dk[i], dv[i]) for i in range(self.width)]
        if self.lag_frames is not None:
            if not all(map(math.isfinite, dk)) or not all(map(math.isfinite, dv)):
                raise ValueError("Nonfinite native K/V drift")
            self.lag_frames.append(dict(block=self.block, clock=self.clock, joint=joint))
            if not pool:
                return
        if max(prev.last_write) > self.step_id or max(prev.last_query) > self.step_id:
            raise RuntimeError("write/query step id is ahead of the current step")
        scores = _frontier_scores(pool, prev.last_write, prev.last_query, self.step_id)
        nominal_k = (max(1, math.ceil(self.fraction * len(pool)))
                     if self.top_k is None else min(int(self.top_k), len(pool)))
        top, top_k, top_cutoff, top_tie, top_expanded = top_set(
            joint, pool, k=nominal_k, floor=self.drift_floor, tie_aware=True)
        selected, selected_k, selected_cutoff, selected_tie, selected_expanded = top_set(
            scores, pool, k=nominal_k, floor=0.0, tie_aware=False)
        if len(selected) != nominal_k or selected_expanded:
            raise RuntimeError("frontier prediction must be exactly k positions")
        top_set_, selected_set = set(top), set(selected)
        classes = {i: classify(self.h.resolved, self.h.fresh,
                               self.before, after, accepted, i) for i in pool}
        stats = enrichment(pool, top, selected)
        class_counts = {event: 0 for event in EVENTS}
        class_hits = {event: 0 for event in EVENTS}
        for i in top:
            class_counts[classes[i]] += 1
        for i in top_set_ & selected_set:
            class_hits[classes[i]] += 1
        positives_available = sum(1 for i in pool if joint[i] >= self.drift_floor)
        self.records.append(dict(
            block=self.block, clock=self.clock, prev_clock=prev.clock,
            raw_clock=self.clock, prev_raw_clock=prev.clock,
            step_id=self.step_id, prev_step_id=prev.step_id,
            phase="t-1_to_t_same_canvas", pool_kind="non_fresh_resolved",
            pool_size=stats["pool_size"], nominal_k=nominal_k, top_k=top_k,
            top_count=stats["top_count"], positives_available=positives_available,
            top_cutoff=top_cutoff, top_tie_count=top_tie, top_expanded=top_expanded,
            selected_count=stats["selected_count"], selected_cutoff=selected_cutoff,
            selected_tie_count=selected_tie, selected_expanded=selected_expanded,
            hits=stats["hits"], precision=stats["precision"], recall=stats["recall"],
            base_rate=stats["base_rate"], uniform_expected=stats["uniform_expected"],
            uniform_positional_hits=stats["uniform_positional_hits"], lift=stats["lift"],
            class_counts=class_counts, class_hits=class_hits,
            mean_dk_top=sum(dk[i] for i in top) / len(top) if top else None,
            mean_dv_top=sum(dv[i] for i in top) / len(top) if top else None,
            mean_joint_top=sum(joint[i] for i in top) / len(top) if top else None,
            mean_drift_top=sum(joint[i] for i in top) / len(top) if top else None,
            mean_dk_pool=sum(dk[i] for i in pool) / len(pool),
            mean_dv_pool=sum(dv[i] for i in pool) / len(pool),
            mean_joint_pool=sum(joint[i] for i in pool) / len(pool),
            mean_drift_pool=sum(joint[i] for i in pool) / len(pool),
            mean_score_selected=sum(scores[i] for i in selected) / len(selected),
            mean_score_pool=sum(scores[i] for i in pool) / len(pool),
            top_positions=[[i, dk[i], dv[i], joint[i], classes[i]]
                           for i in sorted(top, key=lambda i: (-joint[i], i))],
            selected_positions=[[i, scores[i]]
                                for i in sorted(selected, key=lambda i: (-scores[i], i))],
        ))
        for i in pool:
            p = self.pos[i]
            p["pool"] += 1
            p["dk_sum"] += dk[i]
            p["dv_sum"] += dv[i]
            p["joint_sum"] += joint[i]
            p["dk_max"] = max(p["dk_max"], dk[i])
            p["dv_max"] = max(p["dv_max"], dv[i])
            p["joint_max"] = max(p["joint_max"], joint[i])
            p["score_sum"] += scores[i]
            if i in top_set_:
                p["top"] += 1
                p["top_dk_sum"] += dk[i]
                p["top_dv_sum"] += dv[i]
                p["top_joint_sum"] += joint[i]
            if i in selected_set:
                p["selected"] += 1
        for i in range(self.width):
            self.pos[i][classify(self.h.resolved, self.h.fresh,
                                 self.before, after, accepted, i)] += 1

    def position_rows(self):
        rows = []
        for i, p in enumerate(self.pos):
            rows.append(dict(
                index=i, pool=p["pool"], top=p["top"], selected=p["selected"],
                mean_dk=p["dk_sum"] / p["pool"] if p["pool"] else None,
                mean_dv=p["dv_sum"] / p["pool"] if p["pool"] else None,
                mean_joint=p["joint_sum"] / p["pool"] if p["pool"] else None,
                max_dk=p["dk_max"] if p["pool"] else None,
                max_dv=p["dv_max"] if p["pool"] else None,
                max_joint=p["joint_max"] if p["pool"] else None,
                mean_score=p["score_sum"] / p["pool"] if p["pool"] else None,
                mean_top_dk=p["top_dk_sum"] / p["top"] if p["top"] else None,
                mean_top_dv=p["top_dv_sum"] / p["top"] if p["top"] else None,
                mean_top_joint=p["top_joint_sum"] / p["top"] if p["top"] else None,
                **{event: p[event] for event in EVENTS},
            ))
        return rows


def summarize_task(records, empty_pool_steps=0):
    pool_total = sum(r["pool_size"] for r in records)
    top_total = sum(r["top_count"] for r in records)
    selected_total = sum(r["selected_count"] for r in records)
    hits_total = sum(r["hits"] for r in records)
    uniform_expected_total = sum(r["uniform_expected"] for r in records)
    positives_available_total = sum(r["positives_available"] for r in records)
    empty_top_steps = sum(1 for r in records if r["top_count"] == 0)
    micro_precision = hits_total / selected_total if selected_total else None
    micro_recall = hits_total / top_total if top_total else None
    micro_base = top_total / pool_total if pool_total else None
    micro_lift = hits_total / uniform_expected_total if uniform_expected_total > 0 else None
    class_counts = {e: sum(r["class_counts"].get(e, 0) for r in records) for e in EVENTS}
    class_hits = {e: sum(r["class_hits"].get(e, 0) for r in records) for e in EVENTS}
    class_expected = {
        e: sum(r["selected_count"] * r["class_counts"].get(e, 0) / r["pool_size"]
               for r in records if r["pool_size"])
        for e in EVENTS}
    class_lift = {}
    for e in EVENTS:
        class_lift[e] = class_hits[e] / class_expected[e] if class_expected[e] > 0 else None
    mean_dk_top = (sum(r["mean_dk_top"] * r["top_count"]
                       for r in records if r["top_count"]) / top_total
                   if top_total else None)
    mean_dv_top = (sum(r["mean_dv_top"] * r["top_count"]
                       for r in records if r["top_count"]) / top_total
                   if top_total else None)
    mean_joint_top = (sum(r["mean_joint_top"] * r["top_count"]
                          for r in records if r["top_count"]) / top_total
                      if top_total else None)
    mean_drift_top = mean_joint_top
    mean_dk_pool = (sum(r["mean_dk_pool"] * r["pool_size"] for r in records) / pool_total
                    if pool_total else None)
    mean_dv_pool = (sum(r["mean_dv_pool"] * r["pool_size"] for r in records) / pool_total
                    if pool_total else None)
    mean_joint_pool = (sum(r["mean_joint_pool"] * r["pool_size"] for r in records) / pool_total
                       if pool_total else None)
    mean_drift_pool = mean_joint_pool
    mean_score_selected = (sum(r["mean_score_selected"] * r["selected_count"] for r in records)
                           / selected_total if selected_total else None)
    mean_score_pool = (sum(r["mean_score_pool"] * r["pool_size"] for r in records)
                       / pool_total if pool_total else None)
    t2t = class_counts["t2t_old"] + class_counts["t2t_fresh"]
    t2t_hits = class_hits["t2t_old"] + class_hits["t2t_fresh"]
    t2t_expected = sum(
        r["selected_count"] * (r["class_counts"].get("t2t_old", 0)
                               + r["class_counts"].get("t2t_fresh", 0)) / r["pool_size"]
        for r in records if r["pool_size"])
    step_lifts = [r["lift"] for r in records if r["lift"] is not None]
    return dict(
        transitions=len(records), empty_pool_steps=empty_pool_steps,
        pool_total=pool_total, top_total=top_total,
        selected_total=selected_total, hits_total=hits_total,
        uniform_expected_total=uniform_expected_total,
        positives_available_total=positives_available_total,
        empty_top_steps=empty_top_steps,
        micro_precision=micro_precision, micro_recall=micro_recall,
        micro_base=micro_base, micro_lift=micro_lift,
        macro_lift_step=sum(step_lifts) / len(step_lifts) if step_lifts else None,
        class_counts=class_counts, class_hits=class_hits, class_lift=class_lift,
        class_expected=class_expected,
        t2t_total=t2t, t2t_hits=t2t_hits,
        t2t_expected=t2t_expected,
        t2t_lift=t2t_hits / t2t_expected if t2t_expected > 0 else None,
        mean_dk_top=mean_dk_top, mean_dv_top=mean_dv_top,
        mean_joint_top=mean_joint_top,
        mean_drift_top=mean_drift_top, mean_drift_pool=mean_drift_pool,
        mean_dk_pool=mean_dk_pool, mean_dv_pool=mean_dv_pool,
        mean_joint_pool=mean_joint_pool,
        mean_score_selected=mean_score_selected, mean_score_pool=mean_score_pool,
    )


def shared_kv_layers(model):
    return [i for i, layer in enumerate(model.model.decoder.layers)
            if getattr(layer.self_attn, "v_proj", None) is None]


@contextmanager
def capture_attention(model, layers, canvas, store, impl=None):
    """Capture post-RoPE K and post-v_norm V without changing the forward."""
    if impl is None:
        from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
            ALL_ATTENTION_FUNCTIONS as impl,
        )
    alias = ATTENTION_ALIAS
    if alias in impl:
        raise RuntimeError("native KV drift attention is already active")
    original = impl["sdpa"]
    selected = {int(i) for i in layers}
    decoder_layers = list(model.model.decoder.layers)
    if not selected <= set(range(len(decoder_layers))):
        raise ValueError(f"selected layer outside decoder: {sorted(selected)}")
    for i in selected:
        attn = decoder_layers[i].self_attn
        if getattr(attn.config, "_attn_implementation", None) != "sdpa":
            raise ValueError("native KV drift requires the pinned SDPA backend")

    def capture(module, q, k, v, mask, **kwargs):
        idx = getattr(module, "layer_idx", None)
        if idx in selected and store.active:
            if q.shape[-2] != canvas:
                raise RuntimeError(f"decoder Q rows {q.shape[-2]} != canvas {canvas}")
            if k.shape[-2] < canvas or v.shape[-2] < canvas:
                raise RuntimeError("decoder KV is shorter than the canvas")
            store.seen[idx] = store.seen.get(idx, 0) + 1
            store.k[idx] = k[..., -canvas:, :].detach().clone()
            store.v[idx] = v[..., -canvas:, :].detach().clone()
        return original(module, q, k, v, mask, **kwargs)

    with ExitStack() as stack:
        impl[alias] = capture
        stack.callback(impl.__delitem__, alias)
        for i in selected:
            attn = decoder_layers[i].self_attn
            stack.callback(setattr, attn, "config", attn.config)
            attn.config = copy(attn.config)
            attn.config._attn_implementation = alias
        yield


def kv_drift_trace(out, layer, top_fraction=DEFAULT_TOP_FRACTION, top_k=None,
                   drift_floor=DEFAULT_DRIFT_FLOOR, *, lag_joint=False):
    """Drop-in ``trace_request`` for ``run_native``; writes per-task statistics."""
    @contextmanager
    def capture(model, task):
        state = DriftState(model.config.canvas_length, layer, top_fraction, top_k,
                           drift_floor, lag_joint=lag_joint)
        attn = model.model.decoder.layers[layer].self_attn
        state.layer_type = getattr(attn, "layer_type", None)
        state.shared_kv = getattr(attn, "v_proj", None) is None
        state.shared_kv_layers = shared_kv_layers(model)
        state.head_dim = int(attn.head_dim)
        state.kv_heads = int(attn.k_proj.out_features // state.head_dim)
        store = CaptureStore()
        with gemma_trace(out)(model, task):
            trace_step = model._denoising_step
            with capture_attention(model, {layer}, state.width, store):
                def step(**kw):
                    state.begin(int(kw["input_ids"].shape[1]), int(kw["cur_step"]),
                                kw["current_canvas"][0].tolist())
                    store.reset()
                    store.active = True
                    try:
                        result = trace_step(**kw)
                    finally:
                        store.active = False
                    state.end(result[0][0].tolist(),
                              kw["sampler"].accepted_token_mask[0].tolist(),
                              store.k.get(layer), store.v.get(layer),
                              seen=store.seen.get(layer, 0))
                    return result

                model._denoising_step = step
                try:
                    yield state
                finally:
                    model._denoising_step = trace_step
        write_task(out, task, state)

    return capture


def write_task(out, task, state):
    payload = dict(
        index=task["index"], task_id=task["task_id"], dataset=task["dataset"],
        seed=task["seed"], model="gemma", policy="native256", layer=state.layer,
        layer_type=getattr(state, "layer_type", None),
        shared_kv=getattr(state, "shared_kv", None),
        shared_kv_layers=getattr(state, "shared_kv_layers", []),
        head_dim=getattr(state, "head_dim", None),
        kv_heads=getattr(state, "kv_heads", None),
        canvas=state.width, top_fraction=state.fraction, top_k=state.top_k,
        joint=JOINT, drift_floor=state.drift_floor,
        phase="t-1_to_t_same_canvas", pool_kind="non_fresh_resolved",
        capture="decoder_self_attn_sdpa_input_post_rope_post_vnorm",
        steps=state.records, positions=state.position_rows(),
        summary=summarize_task(state.records, state.empty_pool_steps),
    )
    write_json(Path(out) / f"{task['index']:02d}-kvdrift-layer{state.layer}.json", payload)
    if state.lag_frames is not None:
        identity = {k: task[k] for k in ('index', 'dataset', 'task_id', 'seed')}
        write_json(Path(out) / f"{task['index']:02d}-lag-joint-layer{state.layer}.json",
                   dict(**identity, canvas=state.width, layer=state.layer, metric=JOINT,
                        frames=state.lag_frames))


def summary_row(payload):
    s = payload["summary"]
    return dict(
        index=payload["index"], dataset=payload["dataset"], task_id=payload["task_id"],
        seed=payload["seed"], layer=payload["layer"], layer_type=payload["layer_type"],
        shared_kv=payload["shared_kv"], transitions=s["transitions"],
        pool_total=s["pool_total"], top_total=s["top_total"],
        selected_total=s["selected_total"], hits_total=s["hits_total"],
        uniform_expected_total=s["uniform_expected_total"],
        positives_available_total=s["positives_available_total"],
        empty_top_steps=s["empty_top_steps"],
        empty_pool_steps=s["empty_pool_steps"],
        micro_precision=s["micro_precision"], micro_recall=s["micro_recall"],
        micro_lift=s["micro_lift"], t2t_total=s["t2t_total"], t2t_hits=s["t2t_hits"],
        t2t_lift=s["t2t_lift"], context_total=s["class_counts"].get("context", 0),
        context_hits=s["class_hits"].get("context", 0),
        revoke_total=s["class_counts"].get("revoke", 0),
        revoke_hits=s["class_hits"].get("revoke", 0),
        mean_dk_top=s["mean_dk_top"], mean_dv_top=s["mean_dv_top"],
        mean_joint_top=s["mean_joint_top"],
        mean_drift_top=s["mean_drift_top"],
        mean_dk_pool=s["mean_dk_pool"], mean_dv_pool=s["mean_dv_pool"],
        mean_joint_pool=s["mean_joint_pool"],
        mean_drift_pool=s["mean_drift_pool"],
    )


def aggregate_tasks(task_payloads, parity, layer):
    summaries = [p["summary"] for p in task_payloads]
    pool_total = sum(s["pool_total"] for s in summaries)
    top_total = sum(s["top_total"] for s in summaries)
    selected_total = sum(s["selected_total"] for s in summaries)
    hits_total = sum(s["hits_total"] for s in summaries)
    uniform_expected_total = sum(s["uniform_expected_total"] for s in summaries)
    positives_available_total = sum(s["positives_available_total"] for s in summaries)
    empty_top_steps = sum(s["empty_top_steps"] for s in summaries)
    empty_pool_steps = sum(s["empty_pool_steps"] for s in summaries)
    micro_precision = hits_total / selected_total if selected_total else None
    micro_recall = hits_total / top_total if top_total else None
    micro_base = top_total / pool_total if pool_total else None
    micro_lift = hits_total / uniform_expected_total if uniform_expected_total > 0 else None
    class_counts = {e: sum(s["class_counts"].get(e, 0) for s in summaries) for e in EVENTS}
    class_hits = {e: sum(s["class_hits"].get(e, 0) for s in summaries) for e in EVENTS}
    class_expected = {e: sum(s["class_expected"].get(e, 0.0) for s in summaries)
                      for e in EVENTS}
    class_lift = {}
    for e in EVENTS:
        class_lift[e] = class_hits[e] / class_expected[e] if class_expected[e] > 0 else None
    task_lifts = [s["micro_lift"] for s in summaries if s["micro_lift"] is not None]
    step_lifts = [r["lift"] for p in task_payloads for r in p["steps"]
                  if r["lift"] is not None]
    t2t = class_counts["t2t_old"] + class_counts["t2t_fresh"]
    t2t_hits = class_hits["t2t_old"] + class_hits["t2t_fresh"]
    t2t_expected = sum(s["t2t_expected"] for s in summaries)
    mean_dk_top = (sum(s["mean_dk_top"] * s["top_total"]
                       for s in summaries if s["top_total"]) / top_total
                   if top_total else None)
    mean_dv_top = (sum(s["mean_dv_top"] * s["top_total"]
                       for s in summaries if s["top_total"]) / top_total
                   if top_total else None)
    mean_joint_top = (sum(s["mean_joint_top"] * s["top_total"]
                          for s in summaries if s["top_total"]) / top_total
                      if top_total else None)
    mean_drift_top = mean_joint_top
    mean_dk_pool = (sum(s["mean_dk_pool"] * s["pool_total"] for s in summaries) / pool_total
                    if pool_total else None)
    mean_dv_pool = (sum(s["mean_dv_pool"] * s["pool_total"] for s in summaries) / pool_total
                    if pool_total else None)
    mean_joint_pool = (sum(s["mean_joint_pool"] * s["pool_total"] for s in summaries) / pool_total
                       if pool_total else None)
    mean_drift_pool = mean_joint_pool
    mean_score_selected = (
        sum(s["mean_score_selected"] * s["selected_total"]
            for s in summaries if s["selected_total"]) / selected_total
        if selected_total else None)
    mean_score_pool = (
        sum(s["mean_score_pool"] * s["pool_total"]
            for s in summaries if s["pool_total"]) / pool_total
        if pool_total else None)
    per_task = [dict(
        index=p["index"], dataset=p["dataset"], task_id=p["task_id"], seed=p["seed"],
        transitions=s["transitions"], pool_total=s["pool_total"], top_total=s["top_total"],
        selected_total=s["selected_total"], hits_total=s["hits_total"],
        uniform_expected_total=s["uniform_expected_total"],
        positives_available_total=s["positives_available_total"],
        empty_top_steps=s["empty_top_steps"],
        empty_pool_steps=s["empty_pool_steps"],
        micro_precision=s["micro_precision"], micro_recall=s["micro_recall"],
        micro_lift=s["micro_lift"], t2t_total=s["t2t_total"], t2t_hits=s["t2t_hits"],
        context_total=s["class_counts"].get("context", 0),
        context_hits=s["class_hits"].get("context", 0),
        revoke_total=s["class_counts"].get("revoke", 0),
        revoke_hits=s["class_hits"].get("revoke", 0),
        mean_dk_top=s["mean_dk_top"], mean_dv_top=s["mean_dv_top"],
        mean_joint_top=s["mean_joint_top"],
        mean_drift_top=s["mean_drift_top"],
        mean_dk_pool=s["mean_dk_pool"], mean_dv_pool=s["mean_dv_pool"],
        mean_joint_pool=s["mean_joint_pool"],
        mean_drift_pool=s["mean_drift_pool"],
    ) for p, s in zip(task_payloads, summaries, strict=True)]
    return dict(
        model="gemma", layer=layer, requests=len(task_payloads),
        joint=JOINT,
        drift_floor=(task_payloads[0].get("drift_floor", DEFAULT_DRIFT_FLOOR)
                     if task_payloads else DEFAULT_DRIFT_FLOOR),
        phase="t-1_to_t_same_canvas", pool_kind="non_fresh_resolved",
        shared_kv_layers=task_payloads[0]["shared_kv_layers"] if task_payloads else [],
        layer_type=task_payloads[0]["layer_type"] if task_payloads else None,
        selected_shared_kv=task_payloads[0]["shared_kv"] if task_payloads else None,
        transitions=sum(s["transitions"] for s in summaries),
        pool_total=pool_total, top_total=top_total, selected_total=selected_total,
        hits_total=hits_total, uniform_expected_total=uniform_expected_total,
        positives_available_total=positives_available_total,
        empty_top_steps=empty_top_steps,
        empty_pool_steps=empty_pool_steps,
        micro_precision=micro_precision,
        micro_recall=micro_recall, micro_base=micro_base, micro_lift=micro_lift,
        macro_lift_task=sum(task_lifts) / len(task_lifts) if task_lifts else None,
        macro_lift_step=sum(step_lifts) / len(step_lifts) if step_lifts else None,
        class_counts=class_counts, class_hits=class_hits, class_lift=class_lift,
        class_expected=class_expected,
        t2t_total=t2t, t2t_hits=t2t_hits,
        t2t_expected=t2t_expected,
        t2t_lift=t2t_hits / t2t_expected if t2t_expected > 0 else None,
        context_total=class_counts.get("context", 0),
        context_hits=class_hits.get("context", 0),
        revoke_total=class_counts.get("revoke", 0),
        revoke_hits=class_hits.get("revoke", 0),
        mean_dk_top=mean_dk_top, mean_dv_top=mean_dv_top,
        mean_joint_top=mean_joint_top,
        mean_drift_top=mean_drift_top,
        mean_dk_pool=mean_dk_pool, mean_dv_pool=mean_dv_pool,
        mean_joint_pool=mean_joint_pool,
        mean_drift_pool=mean_drift_pool,
        mean_score_selected=mean_score_selected,
        mean_score_pool=mean_score_pool,
        parity=parity, per_task=per_task,
    )


def verify_parity(output, reference, tasks):
    output, reference = Path(output), Path(reference)
    rows = []
    for task in tasks:
        name = f"{task['index']:02d}-native256.json"
        new = json.loads((output / name).read_text())
        old = json.loads((reference / name).read_text())
        if new != old:
            diff = sorted(k for k in set(new) | set(old) if new.get(k) != old.get(k))
            raise AssertionError(f"Native Gemma parity failed for {name}: {diff}")
        rows.append({k: new[k] for k in (
            "index", "task_id", "dataset", "seed", "output_tokens",
            "generation_forwards", "generation_query_rows")})
    if [r["index"] for r in rows] != list(range(len(rows))):
        raise AssertionError("Task indices are not contiguous")
    return rows


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    tmp.replace(path)


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(args, prepare=None):
    if getattr(args, "model", "gemma") != "gemma":
        raise ValueError("native_kv_drift is Gemma-only")
    if getattr(args, "reference", None) is None:
        raise ValueError("--reference is required")
    layer = getattr(args, "layer", None)
    if layer is None:
        layer = os.environ.get("DG_KV_LAYER", DEFAULT_LAYER)
    layer = int(layer)
    fraction = float(getattr(args, "top_fraction", DEFAULT_TOP_FRACTION))
    top_k = getattr(args, "top_k", None)
    if top_k is not None:
        top_k = int(top_k)
    drift_floor = float(getattr(args, "drift_floor", DEFAULT_DRIFT_FLOOR))
    if layer < 0:
        raise ValueError("layer must be non-negative")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("top fraction must be in (0, 1]")
    if top_k is not None and top_k <= 0:
        raise ValueError("top-k must be positive")
    if not 0.0 < drift_floor < float("inf"):
        raise ValueError("drift floor must be positive and finite")
    args.output = Path(args.output)
    args.reference = Path(args.reference)
    args.samples = Path(args.samples)
    args.checkpoint = Path(args.checkpoint)
    tasks = load_tasks(args.samples)
    if len(tasks) != 30:
        raise ValueError("native_kv_drift requires all 30 frozen tasks")
    run_native(args, before_cuda=prepare,
               trace_request=kv_drift_trace(args.output, layer, fraction, top_k,
                                            drift_floor, lag_joint=(getattr(args, 'lag_joint', False)
                                                                    or os.getenv('DG_KV_LAG_JOINT') == '1')))
    parity = verify_parity(args.output, args.reference, tasks)
    traces = list(args.output.glob("*.trace.jsonl.gz"))
    if len(traces) != 30:
        raise AssertionError(f"expected 30 traces, found {len(traces)}")
    payloads = [json.loads(
        (args.output / f"{t['index']:02d}-kvdrift-layer{layer}.json").read_text())
        for t in tasks]
    aggregate = aggregate_tasks(payloads, parity, layer)
    write_json(args.output / "kvdrift-aggregate.json", aggregate)
    write_csv(args.output / "kvdrift-summary.csv", [summary_row(p) for p in payloads])
    write_json(args.output / "verified.json",
               dict(model=args.model, matched_requests=len(tasks)))
    print(json.dumps({k: aggregate[k] for k in (
        "requests", "transitions", "pool_total", "hits_total", "micro_lift",
        "t2t_total", "context_total")}, ensure_ascii=False), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma", choices=("gemma",))
    for key in ("checkpoint", "samples", "output", "reference"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--lag-joint", action="store_true",
                        help="write per-row joint drift on every same-canvas transition (also DG_KV_LAG_JOINT=1)")
    parser.add_argument("--top-fraction", type=float, default=DEFAULT_TOP_FRACTION)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--drift-floor", type=float, default=DEFAULT_DRIFT_FLOOR)
    return parser.parse_args()


def _read_trace(path):
    with gzip.open(path, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    if not rows or "schema" not in rows[0]:
        raise ValueError(f"missing trace metadata: {path}")
    return rows[0], rows[1:]


def _load_diagnostic(folder, layer):
    paths = sorted(folder.glob(f"*-kvdrift-layer{layer}.json"))
    tasks = []
    for path in paths:
        diag = json.loads(path.read_text())
        if diag.get("layer") != layer:
            raise ValueError(f"wrong layer in {path}: {diag.get('layer')}")
        if diag.get("pool_kind") != "non_fresh_resolved":
            raise ValueError(f"wrong pool kind in {path}: {diag.get('pool_kind')}")
        if diag.get("phase") != "t-1_to_t_same_canvas":
            raise ValueError(f"wrong phase in {path}: {diag.get('phase')}")
        tasks.append(diag)
    tasks.sort(key=lambda d: d["index"])
    return tasks


def _coverage_for_step(pool, target, resolved, fresh, last_write, last_query,
                       clock, previous_strict, budget, policy, seed, width):
    """Coverage of one dense prestate by one Q/policy pair."""
    edit_slots = 0 if policy == "hard_noedit" else budget // 8
    if policy != "hard_noedit" and edit_slots <= 0:
        return None
    kwargs = {"previous_strict": previous_strict} if policy == "hard_trigger" else {}
    chosen, reserved = select_hard(
        list(range(width)), list(resolved), set(fresh), list(last_write),
        list(last_query), clock, budget, edit_slots, policy, seed, **kwargs)
    chosen_set, reserved_set = set(chosen), set(reserved)
    pool_set, target_set = set(pool), set(target)
    masks_total = sum(1 for i in range(width) if not resolved[i])
    mask_seats = sum(1 for i in chosen if not resolved[i])
    fresh_selected = sum(1 for i in chosen if i in fresh)
    neutral_backfill = sum(
        1 for i in chosen
        if i not in fresh and resolved[i] and i not in reserved_set)
    reserved_hits = len(reserved_set & target_set)
    total_hits = len(chosen_set & target_set)
    pool_size = len(pool_set)
    target_size = len(target_set)
    old_seats = len(reserved_set) + neutral_backfill
    return dict(
        policy=policy, budget=budget, edit_slots=edit_slots,
        mode=("dense_full_selection" if budget == width
              else "fixed_edit_seats"),
        pool_size=pool_size, target_count=len(target_set),
        fresh_count=len(fresh), masks_total=masks_total,
        seats_total=len(chosen_set), reserved_count=len(reserved_set),
        reserved_hits=reserved_hits, total_hits=total_hits,
        backfill_hits=total_hits - reserved_hits,
        random_expectation_reserved=(
            min(len(reserved_set), pool_size) * target_size / pool_size
            if pool_size else 0.0),
        random_expectation_total=(
            old_seats * target_size / pool_size if pool_size else 0.0),
        mask_seats=mask_seats, mask_lost=masks_total - mask_seats,
        neutral_backfill_count=neutral_backfill,
        fresh_selected=fresh_selected,
        stage=COVERAGE_STAGE, assumption=COVERAGE_ASSUMPTION,
    )


def _aggregate_coverage(records, width):
    by_key = {}
    for record in records:
        by_key.setdefault((record["budget"], record["policy"]), []).append(record)
    aggregates = []
    for (budget, policy), rows in sorted(by_key.items()):
        sums = {key: sum(row[key] for row in rows) for key in (
            "pool_size", "target_count", "seats_total", "reserved_count",
            "reserved_hits", "total_hits", "backfill_hits",
            "random_expectation_reserved", "random_expectation_total",
            "masks_total", "mask_seats", "mask_lost",
            "neutral_backfill_count", "fresh_selected")}
        target_count = sums["target_count"]
        pool_size = sums["pool_size"]
        total_hits = sums["total_hits"]
        reserved_hits = sums["reserved_hits"]
        random_total = sums["random_expectation_total"]
        random_reserved = sums["random_expectation_reserved"]
        masks_total = sums["masks_total"]
        aggregates.append(dict(
            budget=budget, policy=policy, steps=len(rows),
            nonfresh_steps=sum(1 for row in rows if row["pool_size"] > 0),
            empty_target_steps=sum(1 for row in rows if row["target_count"] == 0),
            dense_control=budget == width,
            mode=("dense_full_selection" if budget == width
                  else "fixed_edit_seats"),
            **sums,
            target_coverage=total_hits / target_count if target_count else None,
            reserved_coverage=reserved_hits / target_count if target_count else None,
            pool_coverage=total_hits / pool_size if pool_size else None,
            random_lift=total_hits / random_total if random_total > 0 else None,
            reserved_lift=(reserved_hits / random_reserved
                           if random_reserved > 0 else None),
            mask_lost_rate=sums["mask_lost"] / masks_total if masks_total else None,
        ))
    by_budget_policy = {(a["budget"], a["policy"]): a for a in aggregates}
    for aggregate in aggregates:
        noedit = by_budget_policy.get((aggregate["budget"], "hard_noedit"))
        if noedit is None:
            aggregate.update(delta_total_hits_vs_noedit=None,
                             delta_reserved_hits_vs_noedit=None,
                             delta_mask_lost_vs_noedit=None)
        else:
            aggregate.update(
                delta_total_hits_vs_noedit=(
                    aggregate["total_hits"] - noedit["total_hits"]),
                delta_reserved_hits_vs_noedit=(
                    aggregate["reserved_hits"] - noedit["reserved_hits"]),
                delta_mask_lost_vs_noedit=(
                    aggregate["mask_lost"] - noedit["mask_lost"]),
            )
    return aggregates


def _process_coverage_task(diag, folder, budgets, policies, seed):
    index = diag["index"]
    trace_path = folder / f"{index:02d}.trace.jsonl.gz"
    meta, rows = _read_trace(trace_path)
    for key in ("index", "task_id", "dataset", "seed"):
        if meta.get(key) != diag.get(key):
            raise ValueError(f"trace metadata mismatch for {index:02d}: {key}")
    width = int(diag["canvas"])
    diag_steps = {(step["block"], step["clock"]): step for step in diag["steps"]}
    if len(diag_steps) != len(diag["steps"]):
        raise ValueError(f"duplicate diagnostic step key for {index:02d}")
    h = History([False] * width, [None] * width, [0] * width, set(), set())
    last_write = [-1] * width
    last_query = [-1] * width
    step_id = 0
    previous_strict = set()
    current_block = None
    previous_after = None
    previous_clock = None
    records = []
    matched = set()
    for row in rows:
        block, clock = int(row["block"]), int(row["clock"])
        before, after, accepted = row["before"], row["after"], row["accepted"]
        if len(before) != width or len(after) != width or len(accepted) != width:
            raise ValueError(f"canvas width mismatch in {trace_path}")
        if block != current_block:
            if current_block is not None and block != current_block + width:
                raise ValueError(f"unexpected canvas advance in {trace_path}")
            h = History([False] * width, [None] * width, [0] * width, set(), set())
            last_write = [-1] * width
            last_query = [-1] * width
            step_id = 0
            previous_strict = set()
            current_block = block
            previous_after = None
            previous_clock = None
        else:
            if previous_after != before:
                raise ValueError(f"trace before-state mismatch in {trace_path}")
            if clock != previous_clock - 1:
                raise ValueError(f"non-native trace clock in {trace_path}")
        step_id += 1
        diag_step = diag_steps.get((block, clock))
        if diag_step is not None:
            matched.add((block, clock))
            if (diag_step.get("step_id") != step_id
                    or diag_step.get("prev_step_id") != step_id - 1
                    or diag_step.get("raw_clock") != clock):
                raise ValueError(f"diagnostic step identity mismatch in {trace_path}")
            pool = [i for i in range(width)
                    if h.resolved[i] and i not in h.fresh]
            if diag_step.get("pool_size") != len(pool):
                raise ValueError(f"diagnostic pool mismatch in {trace_path}")
            target = [int(position[0]) for position in diag_step["top_positions"]]
            if not set(target) <= set(pool):
                raise ValueError(f"target outside non-fresh pool in {trace_path}")
            for budget in budgets:
                # A budget equal to the canvas is the dense full-selection
                # control: fixed edit seats do not apply, so only noedit runs.
                policies_for_budget = (
                    ("hard_noedit",) if budget == width else policies)
                for policy in policies_for_budget:
                    record = _coverage_for_step(
                        pool, target, h.resolved, h.fresh, last_write, last_query,
                        step_id, previous_strict, budget, policy, seed, width)
                    if record is not None:
                        record.update(
                            index=index, task_id=diag["task_id"],
                            dataset=diag["dataset"], seed=diag["seed"],
                            block=block, clock=clock, step_id=step_id,
                            prev_step_id=step_id - 1,
                            phase=diag["phase"], pool_kind=diag["pool_kind"],
                            disclaimer=COVERAGE_DISCLAIMER,
                        )
                        records.append(record)
        previous_h = h
        h, _ = _advance(h, before, after, accepted, range(width), [None] * width)
        _hard_touch(last_write, last_query, range(width), accepted, before, after,
                    previous_h.resolved, step_id)
        previous_strict = {
            i for i in range(width)
            if previous_h.resolved[i] and accepted[i] and before[i] != after[i]}
        previous_after = after
        previous_clock = clock
    if matched != set(diag_steps):
        raise ValueError(f"unmatched diagnostic steps in {trace_path}: "
                         f"{sorted(set(diag_steps) - matched)}")
    return records


def offline_budget_coverage(
        folder, layer=DEFAULT_LAYER, budgets=(16, 32, 64, 128, 256),
        policies=("hard_frontier", "hard_slack", "hard_trigger", "hard_noedit"),
        out=None, seed=0, require_30=True, require_verified=True):
    """Offline Q-budget coverage of high-joint-drift old positions.

    Reads the verified diagnostic records and native traces in ``folder`` and
    replays the *same dense prestate* (the previous native round queried all
    rows).  For every non-fresh step, every budget and every policy, it calls
    the production ``select_hard`` and reports the pool, seats, reserved hits,
    the per-step uniform expectation based on the actual selected old rows
    ``(reserved_count + neutral_backfill_count) * target_count / pool_size``,
    total hits including neutral backfill, and the MASK positions lost relative
    to ``hard_noedit``.  A budget equal to the canvas is the dense
    full-selection control: only ``hard_noedit`` is run and fixed edit-seat
    concepts do not apply.

    This is an offline coverage diagnostic only.  It assumes every non-fresh
    step is selectable; it does not claim actual sparse verification, a sparse
    rollout, or online quality.
    """
    folder = Path(folder)
    out = Path(out) if out is not None else DEFAULT_COVERAGE_OUT
    budgets = tuple(int(budget) for budget in budgets)
    policies = tuple(policies)
    tasks = _load_diagnostic(folder, layer)
    if require_30 and (len(tasks) != 30
                       or [task["index"] for task in tasks] != list(range(30))):
        raise ValueError("offline coverage requires the 30 verified frozen tasks")
    if require_verified:
        verified = json.loads((folder / "verified.json").read_text())
        expected = dict(model="gemma", matched_requests=len(tasks))
        if verified != expected:
            raise ValueError(f"verified.json mismatch: {verified} != {expected}")
    step_records = []
    for diag in tasks:
        records = _process_coverage_task(diag, folder, budgets, policies, seed)
        step_records.extend(records)
    width = int(tasks[0]["canvas"]) if tasks else 0
    aggregate = _aggregate_coverage(step_records, width)
    for row in aggregate:
        row["stage"] = COVERAGE_STAGE
        row["assumption"] = COVERAGE_ASSUMPTION
        row["denominators"] = (
            "pool_total=non-fresh resolved rows; target_total=high joint drift old rows; "
            "seats_total=chosen seats; old_seats=reserved_count+neutral_backfill_count; "
            "random_expectation_total=old_seats*target_total/pool_total; "
            "mask_lost=masks_total-mask_seats")
        row["disclaimer"] = COVERAGE_DISCLAIMER
    metadata = dict(
        folder=str(folder), layer=layer, budgets=list(budgets),
        policies=list(policies), tasks=len(tasks), canvas=width,
        stage=COVERAGE_STAGE, assumption=COVERAGE_ASSUMPTION,
        disclaimer=COVERAGE_DISCLAIMER,
        denominators=(
            "pool_total = non-fresh resolved rows; target_total = high joint drift "
            "old rows; seats_total = chosen seats; old_seats = reserved_count + "
            "neutral_backfill_count; random_expectation_total = old_seats * "
            "target_total / pool_total; mask_lost = masks_total - mask_seats"),
        dense_control_note=(
            "A budget equal to the canvas is a dense full-selection control: "
            "every canvas row is selected, coverage is 100% by construction, "
            "and fixed edit-seat concepts do not apply. Only the hard_noedit "
            "dense row is reported for that budget."),
    )
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "kvdrift-budget-coverage.csv", aggregate)
    write_json(out / "kvdrift-budget-coverage.json",
               dict(metadata=metadata, aggregate=aggregate))
    return dict(metadata=metadata, aggregate=aggregate,
                step_records=len(step_records))


if __name__ == "__main__":
    main(parse_args())

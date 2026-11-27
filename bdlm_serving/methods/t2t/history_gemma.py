"""Batch-one DiffusionGemma 5.12 block-history adapter.

Only the decoder attention backend is replaced. The native cache selects the
sliding-window history, and all canvas rows still reach the LM head. A sparse
stop candidate requests a full-history check at the next native clock.
"""
from contextlib import ExitStack, contextmanager, nullcontext
from copy import copy
from dataclasses import dataclass, field

import torch

from .budget_family import choose
from .seat_audit import canvas_digest
from ...runtime.diffusiongemma import observe


@dataclass
class Rows:
    width: int
    valid: set = field(default_factory=set)
    dirty: dict = field(default_factory=dict)
    debt_since: dict = field(default_factory=dict)
    last_read: dict = field(default_factory=dict)

    def invalidate(self):
        self.valid.clear()
        self.last_read.clear()

    def select(self, budget, accepted, has_history, force_full=False, cold_share=0):
        all_rows = set(range(self.width))
        if not has_history:
            return ()
        if force_full or budget >= self.width:
            return tuple(range(self.width))
        unresolved = {i for i in all_rows if not accepted[i]}
        prefix = []
        for i, ok in enumerate(accepted):
            if not ok:
                break
            if i in self.valid and i not in self.dirty:
                prefix.append(i)
        # Older never-serviced writes win over new writes; tie-break by row.
        # choose orders mandatory IDs numerically, so give it the oldest seats
        # as the mandatory subset when demand exceeds the fixed budget.
        cold = all_rows - self.valid - set(self.dirty)
        reserve = min(len(cold), (budget * cold_share + 99) // 100)
        due = sorted(self.dirty, key=lambda i: (self.debt_since[i], i))[:budget-reserve]
        return choose(budget, self.width, due, unresolved, prefix,
                      cold=cold).rows

    def refresh(self, selected, tick):
        waits = [tick - self.debt_since[i] for i in selected if i in self.dirty]
        for i in selected:
            self.dirty.pop(i, None)
            self.debt_since.pop(i, None)
            self.last_read[i] = tick
        self.valid.update(selected)
        return waits

    def write(self, rows, tick):
        for i in rows:
            self.dirty[i] = tick
            self.debt_since.setdefault(i, tick)


def _bounds(mask):
    """Convert one native partition to the kernel's contiguous legal ranges."""
    width = mask.shape[-1]
    if not width:
        return None, (0,) * mask.shape[1]
    allowed = mask if mask.dtype == torch.bool else mask == 0
    idx = torch.arange(width, device=mask.device)
    lo = torch.where(allowed, idx, width).amin(-1)
    hi = torch.where(allowed, idx + 1, 0).amax(-1)
    counts = allowed.sum(-1)
    if not torch.equal(counts, hi - lo.minimum(hi)):
        raise ValueError("Noncontiguous native Gemma attention is unsupported")
    return (None if bool((counts == width).all()) else
            torch.stack((lo.minimum(hi), hi), -1).int().contiguous()), tuple(counts[0].tolist())


def _replace(stack, obj, name, replacement):
    if name in vars(obj):
        stack.callback(setattr, obj, name, vars(obj)[name])
    else:
        stack.callback(delattr, obj, name)
    setattr(obj, name, replacement)


@contextmanager
def _eligible_sampler(sampler, canvas, eligible, stopper):
    """Run native EB selection on eligible rows, preserving full RNG draws."""
    accept = sampler.accept_canvas
    with ExitStack() as stack:
        def restricted(current, proposal, logits, clock):
            # Native renoise still makes a whole-canvas RNG draw. Cold IDs are
            # restored after the step and never become accepted or committed.
            if eligible:
                index = torch.tensor(eligible, device=canvas.device)
                selected = accept(current.index_select(1, index), proposal.index_select(1, index),
                                  logits.index_select(1, index), clock)
                bits = torch.zeros_like(canvas, dtype=torch.bool)
                bits.index_copy_(1, index, sampler.accepted_token_mask)
                sampler.accepted_token_mask = bits
                return current.clone().index_copy_(1, index, selected)
            sampler.accepted_token_mask = torch.zeros_like(canvas, dtype=torch.bool)
            return current

        _replace(stack, sampler, "accept_canvas", restricted)
        # Do not feed a partly cold draft into the native stability history.
        yield None if stopper is None else lambda *_: torch.zeros(
            (1,), device=canvas.device, dtype=torch.bool)


def _numerics(left, right):
    a, b = left.float(), right.float()
    diff = (a - b).abs()
    metrics = torch.stack((diff.amax(), diff.square().mean().sqrt(), a.square().mean().sqrt()))
    peak, rmse, rms = metrics.tolist()
    return dict(max_abs=peak, rmse=rmse, rel_rmse=rmse / max(rms, 1e-8))


@dataclass
class Ledger:
    events: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    probes: list = field(default_factory=list)


def attention_backend(budget, width, saturating_sdpa=False):
    if budget is None:
        return "native_sdpa_reference"
    if saturating_sdpa and budget >= width:
        return "native_sdpa_saturating_dispatch"
    return "triton_block_history"


@contextmanager
def history_gemma(model, budget, *, audit=False, saturating_sdpa=False, layout="split", cold_share=0):
    """Instrument one exclusive, unpadded native generate call.

    budget=None is the native SDPA reference. Audit mode adds same-input SDPA
    attention and logit probes, and is not a timing/cost measurement.
    """
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import ALL_ATTENTION_FUNCTIONS as impl
    from .history_attention import SlotPlan, block_history, state_like

    width = model.config.canvas_length
    if model.training or (budget is not None and not 1 <= budget <= 2 * width):
        raise ValueError("Expected an eval model and a fixed Gemma history budget")
    backend = attention_backend(budget, width, saturating_sdpa)
    split = backend == "triton_block_history"
    if layout not in ("split", "rect", "halo") or cold_share not in (0, 25, 50, 100):
        raise ValueError("Unknown research arm")
    rectangular = split and layout != "split"
    if rectangular:
        backend = "sdpa_" + layout
    attn_layers = [layer.self_attn for layer in model.model.decoder.layers]
    native_step, sdpa = model._denoising_step, impl["sdpa"]
    alias = "bdlm_gemma_block_history"
    ledger = Ledger()
    states, geometry, scope = {}, {}, None
    rows = Rows(width)
    epoch = previous_epoch = 0
    prompt_length = last_block = last_cache = None
    resolved = [False] * width
    closing = closed = False

    def encoder_tick(_module, _args, _kwargs, _out):
        nonlocal epoch
        epoch += 1

    def attention(module, q, k, v, mask, **kw):
        if scope is None or scope["shadow"]:
            return sdpa(module, q, k, v, mask, **kw)
        i = module.layer_idx
        b, h = q.shape[-2], k.shape[-2] - q.shape[-2]
        if (i in scope["layers"] or q.shape[0] != 1 or b != width or h < 0
                or kw.get("is_causal") or kw.get("dropout") or kw.get("softcap") is not None):
            raise RuntimeError("Unexpected native Gemma decoder attention")
        cache = scope["cache"].layers[i]
        if (h != cache.keys.shape[-2] or cache.values.shape != cache.keys.shape or
                h != (min(scope["block"], module.sliding_window - 1)
                      if module.is_sliding else scope["block"])):
            raise RuntimeError("Unexpected encoder KV extent")
        key = (module.layer_type, h, id(mask))
        if key not in geometry:
            if mask is None:
                geometry[key] = (None, None, (b,) * b, (h,) * b)
            else:
                if mask.ndim != 4 or mask.shape[:2] != (1, 1) or mask.shape[-1] != h + b:
                    raise RuntimeError("Unexpected native Gemma decoder mask")
                legal = mask[:, 0].expand(1, b, h + b)
                bb, bc = _bounds(legal[..., h:])
                hb, hc = _bounds(legal[..., :h])
                geometry[key] = bb, hb, bc, hc
        bb, hb, bc, hc = geometry[key]
        selected = tuple(i for i in scope["selected"] if hc[i])
        heads = q.shape[1]
        scope["layers"][i] = (selected, sum(bc) * heads, sum(hc[j] for j in selected) * heads,
                              tuple(j for j, n in enumerate(hc) if n))
        if not split:
            out = sdpa(module, q, k, v, mask, **kw)
            if scope["probe"]:
                scope["attention_probe"][i] = _numerics(out[0], out[0])
            return out
        if rectangular:
            if mask is not None or scope["probe"]:
                raise ValueError("Rectangular research path expects the unpadded native mask")
            # Resident per-layer attention output. Off-Q rows do not attend;
            # their projections/FFN still run and are not advertised as sparse.
            old = states.get(i)
            if old is None:
                old = q.new_zeros((1, b, heads, q.shape[-1]))
            def attend(positions, kk, vv):
                if not positions:
                    return
                if positions not in scope["plans"]:
                    scope["plans"][positions] = torch.tensor(positions, device=q.device)
                ix = scope["plans"][positions]
                update = sdpa(module, q.index_select(2, ix), kk, vv, None, **kw)[0]
                old.index_copy_(1, ix, update)
            attend(selected, k, v)
            extra = scope["halo"]
            attend(extra, k[:, :, h:], v[:, :, h:])
            total = len(selected) + len(extra)
            scope["layers"][i] = (selected, total * b * heads, len(selected) * h * heads,
                                  tuple(j for j, n in enumerate(hc) if n))
            states[i] = old
            return old, None
        if selected not in scope["plans"]:
            scope["plans"][selected] = SlotPlan.make([selected], b, q.device)
        if i not in states:
            states[i] = state_like(q)
        out = block_history(q.contiguous(), k[:, :, h:].contiguous(), v[:, :, h:].contiguous(),
                            cache.keys.contiguous(), cache.values.contiguous(),
                            bb, hb, scope["plans"][selected], states[i], kw["scaling"])
        out = out.transpose(1, 2).contiguous()
        if scope["probe"]:
            sdpa_out, _ = sdpa(module, q, k, v, mask, **kw)
            scope["attention_probe"][i] = _numerics(sdpa_out, out)
        return out, None

    def step(**kw):
        nonlocal scope, rows, prompt_length, last_block, previous_epoch, resolved, closing, closed, last_cache
        canvas, prior_draft = kw["current_canvas"], kw["argmax_canvas"]
        prefix, clock = kw["input_ids"].shape[1], kw["cur_step"]
        cache = kw["past_key_values"]
        if (canvas.shape != (1, width) or kw["decoder_position_ids"].shape != canvas.shape
                or cache.is_compileable or cache.get_seq_length() != prefix):
            raise ValueError("Expected a native unpadded batch-one dynamic canvas")
        first = last_block != prefix
        if first:
            if last_block is not None and (not closed or prefix != last_block + width or epoch <= previous_epoch):
                raise RuntimeError("Canvas advanced without a close and new encoder version")
            rows, resolved = Rows(width), [False] * width
            prompt_length = prefix if prompt_length is None else prompt_length
            last_block, closing, closed = prefix, False, False
        elif closed or clock != ledger.trace[-1]["clock"] - 1:
            raise RuntimeError("Non-native denoising clock")
        invalidated = first or epoch != previous_epoch or cache is not last_cache
        if invalidated:
            rows.invalidate()
            states.clear()
            geometry.clear()
        previous_epoch, last_cache = epoch, cache
        # Native unpadded generation always supplies an all-legal mapping.
        # Explicit masks are supported by the attention hook, but not by the
        # request scheduler: fail rather than guess row-level cold eligibility.
        if any(mask is not None for mask in kw["mask_mapping"].values()):
            raise ValueError("Padded/explicitly masked generate is not qualified")
        forced = budget is not None and budget < width and (closing or clock == 1)
        selected = rows.select(width if budget is None else budget, resolved, prefix > 0, forced, cold_share)
        tick = len(ledger.trace)
        eligible = sorted((rows.valid | set(selected)) if prefix else range(width))
        cold_before = width - len(rows.valid) if prefix else 0
        no_sample = len(eligible) < width
        sampler = kw["sampler"]
        guard = (_eligible_sampler(sampler, canvas, eligible, kw["diffusion_stopping_criteria"])
                 if no_sample else nullcontext(kw["diffusion_stopping_criteria"]))
        scope = dict(selected=selected, cache=cache, block=prefix, layers={}, plans={}, shadow=False,
                     probe=audit and budget is not None and (first or clock == 1),
                     attention_probe={})
        remaining = set(range(width)) - set(selected)
        halo = tuple(sorted(remaining, key=lambda j: (resolved[j], j))[:len(selected)]) if layout == "halo" else ()
        scope["halo"] = halo
        forward = kw["decoder_forward"]

        def measured(**args):
            out = forward(**args)
            if scope["probe"]:
                scope["shadow"] = True
                try:
                    reference = forward(**args)
                finally:
                    scope["shadow"] = False
                ledger.probes.append(dict(round=tick, attention=scope["attention_probe"],
                                          logits=_numerics(reference.logits, out.logits)))
            return out

        if scope["probe"]:
            kw["decoder_forward"] = measured
        n = len(ledger.events)
        try:
            with guard as stopper:
                kw["diffusion_stopping_criteria"] = stopper
                after, draft, sc, finished = native_step(**kw)
            layers = scope["layers"]
        finally:
            scope = None
        expected = [("decoder", width)] * (2 if audit and budget is not None and (first or clock == 1) else 1)
        if ledger.events[n:] != expected or len(layers) != len(attn_layers):
            raise RuntimeError("Decoder work did not match native physical geometry")
        if len(expected) == 2:
            ledger.events.pop()  # Same-input gate probe is not part of generation.
        required = set().union(*(set(item[3]) for item in layers.values()))
        if required != (set(range(width)) if prefix else set()):
            raise RuntimeError("Actual layer history disagrees with sampler eligibility")
        if any(set(item[0]) != set(selected) & set(item[3]) for item in layers.values()):
            raise RuntimeError("History selection did not reach all applicable layers")
        selected_set = set(selected)
        reused = rows.valid - selected_set
        reused_age = [tick - rows.last_read[i] for i in reused]
        stale_dirty = len(reused & rows.dirty.keys())
        init = len(selected_set - rows.valid)
        waits = rows.refresh(selected, tick)
        if no_sample:
            active = torch.zeros_like(canvas, dtype=torch.bool)
            active[:, eligible] = True
            after = torch.where(active, after, canvas)
            draft = torch.where(active, draft, prior_draft)
        before_ids, after_ids = canvas[0].tolist(), after[0].tolist()
        accepted = sampler.accepted_token_mask[0].tolist()
        admitted = {i for i in eligible if accepted[i] and not resolved[i]}
        edits = {i for i in eligible if accepted[i] and resolved[i] and before_ids[i] != after_ids[i]}
        revoked = {i for i in eligible if resolved[i] and not accepted[i]}
        # Re-noising still-unresolved positions is the native unresolved lane,
        # not a fresh committed-input write (otherwise refresh starves cold).
        if prefix:
            rows.write(admitted | edits | revoked, tick)
        resolved = accepted
        stop = bool(finished.item())
        closing = stop and budget is not None and budget < width and not forced
        closed = not closing and (stop or clock == 1)
        if (any(accepted[i] or after_ids[i] != before_ids[i] for i in set(range(width)) - set(eligible))
                or (closed and no_sample)):
            raise RuntimeError("Cold row accepted, mutated or committed")
        record = dict(block=prefix - prompt_length, encoder_version=epoch, invalidated=invalidated,
                      attention_backend=backend,
                      clock=clock, q=width, history_q=len(selected),
                      history_layer_q=sum(len(item[0]) for item in layers.values()),
                      block_edges=sum(item[1] for item in layers.values()),
                      history_edges=sum(item[2] for item in layers.values()),
                      cold_before=cold_before, first_read=init, reread=len(selected) - init,
                      reused=len(reused), dirty_reuse=stale_dirty, max_reuse_age=max(reused_age, default=0),
                      dirty_pending=len(rows.dirty), dirty_wait=waits, admitted=len(admitted), edited=len(edits),
                      revoked=len(revoked), admitted_without_read=len(admitted - selected_set),
                      edited_without_read=len(edits - selected_set), close=forced,
                      drain_only=bool(selected) and len(waits) == len(selected),
                      close_excess=max(0, len(selected) - (budget or width)),
                      stop_candidate=stop, closed=closed)
        if rectangular or cold_share:
            record.update(layout=layout, cold_share=cold_share,
                          local_attention_q=len(selected)+len(halo) if rectangular else width)
        if audit:
            record.update(selected=selected, canvas=canvas_digest(after_ids, accepted),
                          draft=canvas_digest(draft[0].tolist(), accepted),
                          sampler_eligible=eligible, accepted_rows=[i for i, ok in enumerate(accepted) if ok],
                          admitted_rows=sorted(admitted), edited_rows=sorted(edits), revoked_rows=sorted(revoked),
                          pending_dirty_rows=sorted(rows.dirty))
        ledger.trace.append(record)
        return after, draft, sc, torch.zeros_like(finished) if closing else finished

    if alias in impl:
        raise RuntimeError("Another Gemma history request is already active")
    with ExitStack() as stack:
        ledger.events = stack.enter_context(observe(model))
        h = model.model.encoder.register_forward_hook(encoder_tick, with_kwargs=True)
        stack.callback(h.remove)
        impl[alias] = attention
        stack.callback(impl.__delitem__, alias)
        for layer in attn_layers:
            if layer.config._attn_implementation != "sdpa":
                raise ValueError("Expected pinned native SDPA")
            config = copy(layer.config)
            config._attn_implementation = alias
            _replace(stack, layer, "config", config)
        _replace(stack, model, "_denoising_step", step)
        yield ledger
        if not closed or len(ledger.trace) != sum(tag == "decoder" for tag, _ in ledger.events):
            raise RuntimeError("Missing native final decoder or inconsistent work counters")

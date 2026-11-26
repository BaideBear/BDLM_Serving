"""Batch-one sparse DiffusionGemma with a two-canvas serving window.

Offers are nominal decoder query rows, not hard caps. The engine charges
full initialization/verification, outstanding dirty rows and parked KV
refreshes to actual Q. It does not pad to fill an offer.

Each canvas uses a separate Torch random stream seeded with seed + ordinal.
This is deliberately not a replay of the historical request-global stream:
opening or parking a tail does not consume the head's random draws.
"""

from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from random import Random
from types import SimpleNamespace

import torch

from .gemma_wave_kv import WaveKV, wave_attention

_OFFERS = (32, 64, 128, 256, 512)


@dataclass(frozen=True)
class Tick:
    tick: int
    head: int
    resident: int
    mandatory: tuple[int, ...]
    may_open: bool


@dataclass
class _Canvas:
    ordinal: int
    ids: torch.Tensor
    draft: torch.Tensor
    sampler: object
    stop: object
    processor: object
    rng: torch.Tensor
    age: int = 0
    raw: torch.Tensor | None = None
    sc: torch.Tensor | None = None
    dirty: set[int] = field(default_factory=set)
    ratio: float = 0.0
    verify: bool = False
    tentative: bool = False
    promoted: bool = False


def _with_rng(state, device, fn):
    # Native initialization and sampling use the default generator. Swap only
    # this canvas's stream in and restore the caller's stream even on error.
    if device.type == "cpu":
        get, set_ = torch.random.get_rng_state, torch.random.set_rng_state
    else:
        get = lambda: torch.cuda.get_rng_state(device)
        set_ = lambda value: torch.cuda.set_rng_state(value, device)
    previous = get()
    try:
        set_(state)
        result = fn()
        return result, get()
    finally:
        set_(previous)


def _prefix_policy(plan):
    slots = plan.get("prefix_slots", {})
    if not isinstance(slots, Mapping):
        raise ValueError("prefix_slots must map offers to integer quotas")
    quotas = {}
    for key, count in slots.items():
        offer = int(key) if isinstance(key, str) and key in map(str, _OFFERS) else key
        if (type(offer) is not int or offer not in _OFFERS or offer in quotas
                or type(count) is not int or not 0 <= count <= min(offer, 256)
                or (offer >= 256 and count)):
            raise ValueError("Invalid prefix_slots; only sparse offers have optional prefix seats")
        quotas[offer] = count
    mode = plan.get("prefix_mode", "right" if any(quotas.values()) else "zero")
    if mode not in ("zero", "right", "random") or (mode == "zero" and any(quotas.values())):
        raise ValueError("Expected prefix_mode zero, right or random; zero requires zero quotas")
    return mode, quotas


def _choose(canvas, capacity, quota, mode, *, seed, tick):
    """Reserve dirty head rows, fill unresolved rows, then spend true slack."""
    accepted = canvas.sampler.accepted_token_mask[0].tolist()
    picked = set(canvas.dirty)
    unresolved = [i for i, ok in enumerate(accepted) if not ok]
    picked.update([r for r in unresolved if r not in picked][:max(0, capacity - len(picked))])
    prefix_end = unresolved[0] if unresolved else len(accepted)
    pool = [r for r in range(prefix_end) if r not in picked]
    count = min(quota, len(pool), max(0, capacity - len(picked)))
    optional = []
    if count:
        if mode == "random":
            Random(f"physical-prefix:{seed}:{canvas.ordinal}:{tick}").shuffle(pool)
            optional = sorted(pool[:count])
        elif mode == "right":
            optional = pool[-count:]
        else:
            raise RuntimeError("Zero mode cannot select optional prefix rows")
    return sorted(picked.union(optional)), optional, prefix_end


def _sample(model, s, rows, measured, prefix, cache, masks, steps, device):
    """Run the pinned full-width sampler, publishing state only for rows in Q."""
    block = s.ids.shape[1]
    full = len(rows) == block
    idx = torch.tensor(rows, device=device, dtype=torch.long)
    if full:
        s.raw = measured.contiguous()
    elif s.raw is None or s.sc is None:
        raise RuntimeError("Sparse sampling requires a full initialization")
    else:
        s.raw.index_copy_(1, idx, measured)
    old_ids, old_draft, old_sc = s.ids, s.draft, s.sc
    old_accepted = s.sampler.accepted_token_mask
    old_accepted = (
        old_accepted.clone() if old_accepted is not None else torch.zeros_like(s.ids, dtype=torch.bool)
    )
    active = torch.ones_like(old_accepted) if full else torch.zeros_like(old_accepted)
    if not full:
        active.index_fill_(1, idx, True)
    stop = s.stop
    if not full and stop is not None:
        def stop(draft, scores):
            return s.stop(torch.where(active, draft, old_draft), scores)

    # The native temperature processor mutates its input. Never pass the raw
    # cache itself, or a later sparse step will temperature-scale it again.
    raw = s.raw.clone()
    result, s.rng = _with_rng(s.rng, device, lambda: model._denoising_step(
        decoder_forward=lambda **_: SimpleNamespace(logits=raw),
        current_canvas=old_ids, argmax_canvas=old_draft, input_ids=prefix,
        decoder_position_ids=None, self_conditioning_logits=old_sc,
        mask_mapping=masks, past_key_values=cache,
        finished_denoising=torch.zeros(1, dtype=torch.bool, device=device),
        cur_step=steps - s.age, sampler=s.sampler, logits_processor=s.processor,
        diffusion_stopping_criteria=stop,
    ))
    ids, draft, sc, finished = result
    accepted = s.sampler.accepted_token_mask
    if (ids.shape != old_ids.shape or draft.shape != old_ids.shape
            or sc.shape != s.raw.shape or accepted is None
            or accepted.shape != old_ids.shape or accepted.dtype != torch.bool):
        raise RuntimeError("Incompatible native denoising state")
    if not full:
        ids = torch.where(active, ids, old_ids)
        draft = torch.where(active, draft, old_draft)
        sc = torch.where(active.unsqueeze(-1), sc, old_sc)
        accepted = torch.where(active, accepted, old_accepted)
        s.sampler.accepted_token_mask = accepted
    admitted = accepted & ~old_accepted
    edited = accepted & old_accepted & (ids != old_ids)
    revoked = old_accepted & ~accepted
    changes = tuple(torch.where(bits[0])[0].tolist() for bits in (admitted, edited, revoked))
    # Unaccepted renoise is intentionally stale off-Q; dirty refers to changes
    # in accepted state (admissions, strict edits and revocations).
    s.dirty.update(r for group in changes for r in group)
    s.ratio = len(changes[0]) / max(int((~old_accepted).sum()), 1)
    s.ids, s.draft, s.sc = ids, draft, sc
    s.age += 1
    stopped = bool(finished)
    return stopped, full and (stopped or s.age == steps), changes


@torch.inference_mode()
def generate(model, data, plan, *, next_budget, causal=True, seed=0, quote_costs=False):
    """Return generated IDs and a physical decoder/encoder tape.

    Trace row lists (``visible``, ``selected``, ``sampled``, ``refresh_only``,
    ``prefix_selected``)
    use absolute positions including the prompt offset. ``visible`` describes
    generated keys resident after this decoder call, before any eviction; the
    encoder prefix is implicit and causal visibility is query-dependent.
    Dirty, ages and sampler changes (``admitted``, ``strict_edits``, ``revoked``)
    are per resident canvas, in head-to-tail order, with local row indices.
    Set quote_costs=True to call next_budget(tick, options) before each
    decoder. Otherwise only next_budget(tick) is called, without building a menu.
    ``encoded`` and ``promoted`` list ordinals. ``phase`` describes the head.
    """
    block, cap = plan["block_size"], plan["cap"]
    threshold = plan["flowblock_spawn_threshold"]
    if (type(block) is not int or block != 256 or block != model.config.canvas_length
            or type(cap) is not int or cap < block or cap % block
            or plan.get("batch_size", 1) != 1 or not 0 <= threshold <= 1):
        raise ValueError("Expected a 256-token model canvas and a whole-canvas request cap")
    if type(quote_costs) is not bool:
        raise ValueError("quote_costs must be Boolean")
    if quote_costs:
        from .gemma_quote import quote_options
    if type(causal) is not bool or type(seed) is not int or not callable(next_budget):
        raise ValueError("Expected Boolean causal, integer seed and callable offer source")
    prefix_mode, prefix_slots = _prefix_policy(plan)
    if getattr(model, "training", False):
        raise ValueError("Unified generation requires an eval-mode model")

    prefix, mask = data["input_ids"], data["attention_mask"]
    if (prefix.ndim != 2 or prefix.shape[0] != 1 or not prefix.shape[1]
            or prefix.dtype != torch.long or mask.shape != prefix.shape
            or mask.device != prefix.device or not bool((mask == 1).all())):
        raise ValueError("Expected one unpadded text request")
    if set(data) - {"input_ids", "attention_mask", "mm_token_type_ids"}:
        raise ValueError("Unsupported encoder inputs; only unpadded text is supported")
    if "mm_token_type_ids" in data and not bool((data["mm_token_type_ids"] == 0).all()):
        raise ValueError("Multimodal requests are not supported")
    device, origin = prefix.device, prefix.shape[1]
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and single-device CUDA are supported")
    mask = mask.bool()
    cfg, kwargs = model._prepare_generation_config(None, max_new_tokens=cap)
    steps = cfg.max_denoising_steps
    if (kwargs or type(steps) is not int or steps < 1
            or getattr(cfg, "cache_implementation", None) not in (None, "dynamic", "dynamic_full")):
        raise ValueError("Incompatible native DiffusionGemma generation config")
    cache = model._prepare_cache_for_generation(cfg, 1, origin + cap - block)
    if cache.is_compileable:
        raise ValueError("A native dynamic encoder cache is required")
    masks = dict.fromkeys(model.config.text_config.layer_types)
    layer_count = len(model.model.decoder.layers)
    if not masks or not layer_count:
        raise ValueError("Missing native decoder attention layers")
    eos = cfg.eos_token_id
    eos = set() if eos is None else {eos} if isinstance(eos, int) else set(eos)
    bank = WaveKV(block)
    trace, live, output = [], [], []

    def encode(length):
        nonlocal cache
        pos = torch.arange(prefix.shape[1] - length, prefix.shape[1],
                           device=device, dtype=torch.int32)[None]
        ids, mapping = model._prepare_encoder_inputs(
            prefix, mask, pos, cache, length == origin and not output, block, 1,
        )
        if ids.shape != pos.shape:
            raise RuntimeError("Encoder Q disagrees with committed prefix")
        cache = model.model.encoder(input_ids=ids, attention_mask=mapping,
                                    past_key_values=cache, position_ids=pos).past_key_values
        if cache.get_seq_length() != prefix.shape[1]:
            raise RuntimeError("Encoder cache did not advance to the committed boundary")

    def spawn(ordinal):
        sampler = model._prepare_sampler(cfg)
        rng = torch.Generator(device=device).manual_seed((seed + ordinal) % (1 << 64)).get_state()
        ids, rng = _with_rng(rng, device, lambda: sampler.initialize_canvas(1, device))
        if ids.shape != (1, block) or ids.device != device or ids.dtype != torch.long:
            raise RuntimeError("Incompatible native canvas initialization")
        live.append(_Canvas(
            ordinal, ids, ids, sampler, model._prepare_diffusion_stopping_criteria(cfg),
            model._prepare_logits_processor(cfg), rng,
        ))

    encode(origin)
    spawn(0)
    may_open = False
    with wave_attention(model, bank):
        while live:
            head = live[0]
            state = Tick(len(trace), head.ordinal, len(live),
                         tuple(len(s.dirty) for s in live), may_open)
            if quote_costs:
                options = quote_options(live, may_open=may_open, steps=steps, cap=cap,
                                        prefix_slots=prefix_slots, choose=_choose)
                offer = next_budget(state, options)
            else:
                offer = next_budget(state)
            if type(offer) is not int or offer not in _OFFERS:
                raise ValueError(f"Unsupported decoder offer: {offer!r}")
            if offer == 2 * block and may_open:
                spawn(head.ordinal + 1)
            ages_before = [s.age for s in live]
            dirty_before = [sorted(s.dirty) for s in live]
            mandatory_q = sum(map(len, dirty_before))
            tail = live[1] if len(live) == 2 else None
            # A dirty parked tail spends the same shared offer as the head.
            head_capacity = max(0, offer - len(tail.dirty)) if tail is not None else offer

            optional, prefix_end, verify_reason = [], None, None
            phase = "verify"
            if head.promoted:
                verify_reason = "promotion"
            elif head.raw is None:
                phase = "bootstrap"
            elif head.age == steps - 1:
                verify_reason = "terminal_clock"
            elif head.verify:
                verify_reason = "sparse_stop"
            elif offer >= block:
                phase = "dense"
            elif head_capacity == 0 and not head.dirty:
                phase = "tail_dirty_drain"
                head_rows = []
            else:
                head_rows, optional, prefix_end = _choose(
                    head, head_capacity, prefix_slots.get(offer, 0), prefix_mode, seed=seed, tick=state.tick,
                )
                if head_rows:
                    phase = "sparse"
                else:
                    verify_reason = "empty_selection"
            if phase not in ("sparse", "tail_dirty_drain"):
                head_rows = list(range(block))
            projected = [(head, head_rows, True)] if head_rows else []
            if tail is not None:
                # A tail cannot spend its last step or honor an early stop
                # until its predecessor has been encoded.
                sample_tail = offer == 2 * block and not tail.tentative and tail.age < steps - 1
                if sample_tail:
                    projected.append((tail, list(range(block)), True))
                elif tail.dirty:
                    projected.append((tail, sorted(tail.dirty), False))

            selected, sampled, ids_parts, sc_parts, sc_valid = [], [], [], [], []
            admitted, strict_edits, revoked = ([[] for _ in live] for _ in range(3))
            for s, rows, do_sample in projected:
                if not rows or not s.dirty.issubset(rows):
                    raise RuntimeError("Projected canvas dropped mandatory rows")
                absolute = [origin + s.ordinal * block + r for r in rows]
                selected.extend(absolute)
                if do_sample:
                    sampled.extend(absolute)
                idx = torch.tensor(rows, dtype=torch.long, device=device)
                ids_parts.append(s.ids.index_select(1, idx))
                sc_parts.append(None if s.sc is None else s.sc.index_select(1, idx))
                sc_valid.extend([s.sc is not None] * len(rows))

            q = len(selected)
            if not q or (phase in ("sparse", "tail_dirty_drain") and q > max(offer, mandatory_q)):
                raise RuntimeError("Sparse Q exceeds the shared offer and mandatory debt")
            bank.begin(selected, origin, device, causal=causal)
            with ExitStack() as stack:
                sc = None
                if any(sc_valid):
                    ref = next(x for x in sc_parts if x is not None)
                    sc = torch.cat([
                        x if x is not None else ref.new_zeros((1, ids.shape[1], ref.shape[-1]))
                        for x, ids in zip(sc_parts, ids_parts, strict=True)
                    ], dim=1)
                    if not all(sc_valid):
                        # A fresh canvas uses zero soft embeddings, not softmax
                        # of all-zero logits (a uniform vocabulary embedding).
                        valid = torch.tensor(sc_valid, device=device)[None, :, None]
                        handle = model.model.decoder.self_conditioning.register_forward_pre_hook(
                            lambda _, args: (args[0], args[1].masked_fill(~valid, 0))
                        )
                        stack.callback(handle.remove)
                logits = model(
                    decoder_input_ids=torch.cat(ids_parts, dim=1),
                    decoder_position_ids=torch.tensor([selected], device=device, dtype=torch.int32),
                    self_conditioning_logits=sc, decoder_attention_mask=masks,
                    past_key_values=cache,
                ).logits
            if logits.ndim != 3 or logits.shape[:2] != (1, q):
                raise RuntimeError("Decoder/LM head did not physically project selected Q")
            if hasattr(bank, "visited") and len(bank.visited) != layer_count:
                raise RuntimeError("Wave attention did not intercept every decoder layer")

            visible = [
                pos for s in live if s.raw is not None or any(s is p for p, _, _ in projected)
                for pos in range(origin + s.ordinal * block, origin + (s.ordinal + 1) * block)
            ]
            offset = 0
            done = False
            for s, rows, do_sample in projected:
                measured = logits[:, offset:offset + len(rows)]
                offset += len(rows)
                s.dirty.difference_update(rows)
                if not do_sample:
                    continue
                stop, commit, changes = _sample(
                    model, s, rows, measured, prefix, cache, masks, steps, device,
                )
                i = s.ordinal - head.ordinal
                admitted[i], strict_edits[i], revoked[i] = changes
                if s is head:
                    done = commit
                    s.verify = stop and not commit
                    s.promoted = False
                else:
                    s.tentative = stop or s.age == steps - 1
            sampled_set = set(sampled)
            record = dict(
                tick=state.tick, offer=offer, q=q, head=head.ordinal, visible=visible,
                selected=selected, sampled=sampled, dirty_before=dirty_before,
                dirty_after=[sorted(s.dirty) for s in live], ages_before=ages_before,
                ages_after=[s.age for s in live], encoded=[], promoted=[],
                refresh_only=[p for p in selected if p not in sampled_set],
                phase=phase, verify_reason=verify_reason, prefix_end=prefix_end,
                prefix_quota=prefix_slots.get(offer, 0), prefix_mode=prefix_mode,
                prefix_selected=[origin + head.ordinal * block + p for p in optional],
                admitted=admitted, strict_edits=strict_edits, revoked=revoked,
                mandatory_q=mandatory_q,
                head_capacity=head_capacity if offer < block else block,
                overflow=max(0, q - offer), prefill=origin if state.tick == 0 else 0,
            )
            trace.append(record)
            del logits, measured, sc, sc_parts

            if done:
                tokens = head.draft[0].tolist()
                end = next((i + 1 for i, token in enumerate(tokens) if token in eos), None)
                output.extend(tokens if end is None else tokens[:end])
                if end is not None or len(output) == cap:
                    return output, trace
                prefix = torch.cat((prefix, head.draft), dim=1)
                mask = torch.nn.functional.pad(mask, (0, block), value=True)
                encode(block)
                record["encoded"].append(head.ordinal)
                bank.drop_before(prefix.shape[1])
                live.pop(0)
                if live:
                    new_head = live[0]
                    new_head.sc = None
                    new_head.stop = model._prepare_diffusion_stopping_criteria(cfg)
                    new_head.verify = new_head.tentative = False
                    new_head.promoted = True
                    record["promoted"].append(new_head.ordinal)
                else:
                    spawn(head.ordinal + 1)
            may_open = (
                len(live) == 1 and live[0].age > 0 and not live[0].promoted
                and steps > 1 and live[0].ratio >= threshold
                and live[0].ordinal + 1 < cap // block
            )
    raise RuntimeError("Request exhausted without EOS or output cap")

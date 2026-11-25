"""Batch-one, B32 LLaDA2.1 quota experiment on the native cached decoder.

Install serial.install_llada_rectangular_mask(model, allow_selected_cache=True)
on the raw eval checkpoint once. Do not pass a ModelFacade or share the model
with concurrent forwards (the projection audit uses temporary module hooks).

Bootstrap is the first native denoise pass, not a hidden extra forward. Selected
steps retain stale unselected intra-block KV; full verification and final commit
always recompute from the committed prefix. No future blocks are evaluated.
"""
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
import random

import torch
from transformers.cache_utils import DynamicCache

from bdlm_serving.methods.t2t.planner import History, allocate
from bdlm_serving.methods.t2t.frontier import GEOMETRY_POLICIES, HARD_POLICIES
from .seat_audit import MODES, SeatAudit, canvas_digest, digest
from bdlm_serving.runtime.batch import BatchCache, ForwardBatcher, ModelFacade
from bdlm_serving.runtime.rng import REFERENCE_RNG_LOCK
from bdlm_serving.runtime.serial import RequestState, SerialKernel


_GATED = ("near_miss", "gate_random")
_HAZARD = ("hard_hazard_fixed", "hard_hazard_gate")
from .exchange import ONLINE_MODES as EXCHANGE, PLACEBOS
_HARD = HARD_POLICIES + GEOMETRY_POLICIES + ("hard_oracle1",) + MODES + _HAZARD + tuple(EXCHANGE)


@dataclass
class _GateEvidence:
    # Local position -> (last sampled draft, one-based denoise clock).
    observed: dict = field(default_factory=dict)
    previous_t2t: set = field(default_factory=set)

    def categories(self, state, before, start):
        h, threshold = state.history, state.config["editing_threshold"]
        categories = []
        for i, token in enumerate(before):
            if i + start < state.prompt_length:
                tag = "prompt"
            elif token == state.mask_id:
                tag = "mask"
            elif i in h.fresh:
                tag = "fresh"
            elif (i in self.observed and self.observed[i][0] != token
                  and h.certainty[i] is not None and h.certainty[i] <= threshold):
                tag = "near_miss"
            else:
                tag = "other_resolved"
            categories.append(tag)
        return categories

    def record(self, state, tags, start, picked, proposed, before, after, reserved, released):
        clock = state.step_in_block + 1
        strict = {i for i in picked if tags[i] not in ("prompt", "mask")
                  and after[i] != before[i] and after[i] != state.mask_id}
        state.events[-1]["gate"] = dict(
            available_near_miss_count=tags.count("near_miss"),
            available_nonfresh_count=tags.count("near_miss") + tags.count("other_resolved"),
            reserved_positions=[start+i for i in reserved],
            reserved_near_miss_count=sum(tags[i] == "near_miss" for i in reserved),
            released_edit_slots_to_mask=released,
            previous_strict_edit_positions=[start+i for i in sorted(self.previous_t2t)],
            strict_edit_positions=[start+i for i in sorted(strict)],
            reserved_strict_edit_positions=[start+i for i in reserved if i in strict],
            selected_category_counts={t: sum(tags[i] == t for i in picked)
                                      for t in ("prompt", "mask", "fresh", "near_miss", "other_resolved")},
            # Aligned with absolute_positions. None = unseen, 1 = previous step.
            selected_evidence_age=[clock-self.observed[i][1] if i in self.observed else None
                                   for i in picked],
        )
        self.observed.update((i, (draft, clock)) for i, draft in zip(picked, proposed[0].tolist()))
        self.previous_t2t = strict


class _AuditedDenseCache(BatchCache):
    """One unpadded job, preserving the verified native batch-one KV layout."""

    def __init__(self, cache, q, end):
        super().__init__([{"q": q, "kv": end, "execution_cache": cache}], q, end)
        self.writes = []

    def update(self, k, v, layer_idx, cache_kwargs=None):
        self.writes.append((layer_idx, k.shape[-2], v.shape[-2]))
        return super().update(k, v, layer_idx, cache_kwargs)


class _SelectedCache(DynamicCache):
    def __init__(self, resident, selected, start, end, *, shadow=False):
        super().__init__()
        self.resident, self.selected, self.start = resident, selected, start
        self.shadow = shadow
        self.visible_end = end  # rectangular-mask adapter protocol
        self.writes = []

    def get_seq_length(self, layer_idx=0):
        return self.start

    def update(self, k, v, layer_idx, cache_kwargs=None):
        if (k.shape[-2] != self.selected.numel() or v.shape[-2] != self.selected.numel()
                or layer_idx != len(self.writes)):
            raise RuntimeError(f"Unexpected selected KV rows/order at layer {layer_idx}")
        self.writes.append((layer_idx, k.shape[-2], v.shape[-2]))
        key, value = self.resident[layer_idx]
        if self.shadow:
            return key.index_copy(-2, self.selected, k), value.index_copy(-2, self.selected, v)
        key.index_copy_(-2, self.selected, k)
        value.index_copy_(-2, self.selected, v)
        return key, value


@dataclass
class SparseRequestState(RequestState):
    seat_audit: SeatAudit | None = None
    history: History | None = None
    resident: tuple | None = None
    selection_seed: int = 0
    force_mask_next: bool = False
    gate_evidence: _GateEvidence | None = None
    last_write: list[int] | None = None
    last_query: list[int] | None = None
    previous_strict: set = field(default_factory=set)
    placebo_exchange: dict | None = None


class _RankControl(History):
    """Override only ordering, so all arms retain shared allocate/backfill."""

    def __init__(self, history, mode, seed):
        super().__init__(**vars(history))
        self.mode, self.seed = mode, seed

    def rank(self, eligible, signal):
        if not eligible or self.mode == "ranked":
            return super().rank(eligible, signal)
        if eligible[0] in self.fresh:
            return super().rank(eligible, signal)
        ordered = sorted(eligible)
        if self.mode == "random":
            random.Random(self.seed).shuffle(ordered)
        return ordered


class _GateOrder(History):
    def __init__(self, history, reserved):
        super().__init__(**vars(history))
        self.reserved = set(reserved)

    def rank(self, eligible, signal):
        if not eligible or eligible[0] in self.fresh:
            return super().rank(eligible, signal)
        # Shared, neutral positional backfill after the explicitly reserved seats.
        return sorted(eligible, key=lambda i: (i not in self.reserved, i))


class LLaDASparseKernel(SerialKernel):
    """Same create/step/run/output interface as SerialKernel (cached only).

    ranked/targeted: History ranking. random: same fresh ranking, uniform
    permutation of the non-fresh edit pool. undirected: positional edit order.
    noedit: zero reserved edit slots, with positional ordering for non-fresh
    resolved backfill. All arms apply native updates to every computed row.
    q_budget=32 bypasses sparse selection.

    near_miss/gate_random: release unsupported edit seats to MASK. The former
    ranks observed low-confidence disagreements by distance to the immediately
    preceding pass's strict T2Ts, then confidence and position. The latter draws
    the same number on its own state uniformly from all non-fresh resolved rows.
    Only these policies add event["gate"] to sampled passes. Its position lists
    are absolute; selected_evidence_age is pre-pass age in denoise calls.

    hard_frontier/hard_random/hard_noedit: refresh the preceding pass's admits
    and strict T2Ts, then allocate remaining seats without proposal confidence.
    Hard events report mandatory rows, absolute reserved edits and strict hits,
    and physical Q over the nominal budget (including dense passes).
    """

    def __init__(self, model, q_budget, fresh_slots=0, mask_slots=1,
                 policy="ranked", signal="uncertain", selection_seed=None, hazard_weights=None,
                 exchange_rates=None):
        super().__init__(model, "llada21")
        policy = {"targeted": "ranked"}.get(policy, policy)
        if policy not in ("ranked", "random", "undirected", "noedit", *_GATED, *_HARD):
            raise ValueError(f"Unknown LLaDA quota policy: {policy}")
        if signal not in ("uncertain", "young", "neighbor", "recent_then_uncertain"):
            raise ValueError(f"Unknown History signal: {signal}")
        if (not 1 <= q_budget <= 32 or min(fresh_slots, mask_slots) < 0
                or fresh_slots + mask_slots > q_budget):
            raise ValueError("Invalid B32 budget/quotas")
        if policy == "noedit" and fresh_slots + mask_slots != q_budget:
            raise ValueError("noedit requires zero reserved edit seats")
        if model.training or isinstance(model, ModelFacade):
            raise ValueError("Pass the raw eval LLaDA2.1 model, not a facade")
        self.projections = tuple(layer.attention.query_key_value for layer in model.model.layers)
        if not self.projections:
            raise ValueError("No native LLaDA2.1 QKV projections")
        self.q_budget, self.fresh_slots, self.mask_slots = q_budget, fresh_slots, mask_slots
        self.policy, self.signal, self.selection_seed = policy, signal, selection_seed
        self.hazard_weights = hazard_weights
        self.exchange_rates = exchange_rates
        if policy in EXCHANGE and not exchange_rates:
            raise ValueError("Missing exchange calibration")
        if policy in _HAZARD and (hazard_weights is None or len(hazard_weights) != 25):
            raise ValueError("Missing frozen hazard calibration")

    def create(self, input_ids, config, gen_length, seed, mask_id, eos_ids, cached=True):
        if not cached or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("This backend requires cached batch-one input")
        if bool((input_ids == mask_id).any()):
            raise ValueError("Prompt cannot contain the generation mask ID")
        base = super().create(input_ids, config, gen_length, seed, mask_id, eos_ids, cached=True)
        return SparseRequestState(**vars(base),
                                  seat_audit=SeatAudit(self.policy) if self.policy in MODES else None,
                                  selection_seed=seed if self.selection_seed is None else self.selection_seed)

    def _execute(self, state, positions, end, cache, phase, stage, store):
        q = positions.numel()
        rows = [[0, None] for _ in self.projections]

        def observe(layer, inputs):
            shape = inputs[0].shape
            if len(shape) != 3 or shape[:2] != (1, q):
                raise RuntimeError(f"Layer {layer} physical Q shape {shape}, expected (1,{q},D)")
            rows[layer][0] += 1
            rows[layer][1] = shape[0] * shape[1]

        keys = torch.arange(end, device=positions.device)
        visibility = (positions[:, None] // 32 >= keys[None, :] // 32)[None, None]
        # Gather before embedding, QKV, attention and MLP. No padded model inputs.
        ids = state.x.index_select(1, positions)
        embeds = self.model.get_input_embeddings()(ids)
        with ExitStack() as stack:
            for i, projection in enumerate(self.projections):
                hook = projection.register_forward_pre_hook(
                    lambda _module, inputs, layer=i: observe(layer, inputs))
                stack.callback(hook.remove)
            out = self.model(inputs_embeds=embeds, position_ids=positions[None],
                             attention_mask=visibility, past_key_values=cache,
                             use_cache=True, output_attentions=True)
        if rows != [[1, q]] * len(rows) or cache.writes != [
                (i, q, q) for i in range(len(rows))]:
            raise RuntimeError(f"Q audit mismatch: projections={rows}, KV={cache.writes}")
        if out.logits.ndim != 3 or out.logits.shape[:2] != (1, q):
            raise RuntimeError("Model did not return one logit row per physical query")
        absolute = positions.tolist()
        state.events.append(dict(
            phase=phase, block=state.block, q_start=absolute[0], q_end=absolute[-1]+1,
            q_tokens=q, prefix_kv_length=state.committed_end if phase != "prefill" else 0,
            store_kv=store, context_version=state.context_version, stage=stage,
            absolute_positions=absolute, actual_q=q, physical_q=rows[0][1],
            layer_physical_q=[r[1] for r in rows], F=1, visible_kv=end,
            q_budget=self.q_budget, approximate_same_block_kv=stage == "selected",
        ))
        if self.policy in _HARD:
            clock = len(state.events)
            if state.last_query is not None:
                start = state.block * 32
                for pos in absolute:
                    state.last_query[pos - start] = clock
            state.events[-1].update(
                hard_clock=clock,
                mandatory_fresh=len(state.history.fresh) if state.history else 0,
                reserved_extra_edit_positions=[],
                reserved_strict_edit_positions=[],
                overflow=max(0, q - self.q_budget),
                hard_strict_edits=0,
            )
        return out.logits

    def _forward(self, state, q_start, q_end, store, phase, stage=None):
        if state.cache.get_seq_length() != q_start:
            raise RuntimeError("Committed KV / dense Q start mismatch")
        resident_needed = not store and self.q_budget < 32
        # DynamicCache append rebinds tensors; only the temporary wrapper grows.
        cache = state.cache if store else DynamicCache.from_legacy_cache(state.cache.to_legacy_cache())
        audit = _AuditedDenseCache(cache, q_end - q_start, q_end)
        pos = torch.arange(q_start, q_end, device=state.x.device)
        logits = self._execute(state, pos, q_end, audit, phase, stage or phase, store)
        if cache.get_seq_length() != q_end:
            raise RuntimeError("Dense pass did not populate complete KV")
        if resident_needed:
            state.resident = cache.to_legacy_cache()
        return logits

    def _quotas(self, state):
        fresh, mask = self.fresh_slots, self.mask_slots
        if state.force_mask_next:
            mask = max(1, mask)
            fresh = min(fresh, self.q_budget - mask)
        return fresh, mask

    def _pick(self, state, editable, start):
        rank = _RankControl(state.history, self.policy,
                            f"{state.selection_seed}:{start}:{state.step_in_block}")
        fresh, mask = self._quotas(state)
        picked = allocate(rank, editable, self.q_budget, fresh, mask, self.signal)
        if len(picked) != min(self.q_budget, len(editable)):
            raise RuntimeError("Shared allocate did not fill the requested seats")
        return picked

    def _gate_pick(self, state, editable, start, tags):
        fresh, mask = self._quotas(state)
        capacity = self.q_budget - fresh - mask
        pool = [i for i in editable if tags[i] in ("near_miss", "other_resolved")]
        positives = [i for i in pool if tags[i] == "near_miss"]
        n = min(capacity, len(positives))
        if self.policy == "gate_random":
            rng = random.Random(f"gate_random:{state.selection_seed}:{start}:{state.step_in_block}")
            reserved = rng.sample(pool, n)
        else:
            prev = state.gate_evidence.previous_t2t
            reserved = sorted(positives, key=lambda i: (
                min((abs(i-j) for j in prev), default=32), state.history.certainty[i], i))[:n]
        released = capacity - n
        picked = allocate(_GateOrder(state.history, reserved), editable, self.q_budget,
                          fresh, mask + released, self.signal)
        if len(picked) != min(self.q_budget, len(editable)) or not set(reserved).issubset(picked):
            raise RuntimeError("Gate reservation and shared allocation disagree")
        # Include an edit seat redirected by the existing force-mask rule, if any.
        return picked, reserved, self.q_budget - self.fresh_slots - self.mask_slots - n

    def _hard_pick(self, state, editable, resolved):
        fresh = state.history.fresh.intersection(editable)
        if len(fresh) > self.q_budget:
            return sorted(fresh), []

        from bdlm_serving.methods.t2t.frontier import select_hard

        edit_slots = self.q_budget - self.fresh_slots - self.mask_slots
        oracle = self.policy == "hard_oracle1"
        if self.policy in EXCHANGE:
            from .exchange import choose, randomize_candidate
            clock = len(state.events)+1
            picked, reserved, offers = choose(list(editable), resolved, fresh, state.previous_strict,
                state.last_write, clock, self.q_budget, edit_slots,
                self.exchange_rates, EXCHANGE[self.policy])
            if self.policy in PLACEBOS:
                tag = f"exchange-placebo:{PLACEBOS[self.policy]}:{state.selection_seed}:{state.block}:{clock}"
                picked, reserved, state.placebo_exchange = randomize_candidate(
                    editable, resolved, fresh, picked, reserved, offers, tag)
            return picked, reserved
        if self.policy in _HAZARD:
            from .hazard import choose
            return choose(list(editable), resolved, fresh, state.last_write, state.previous_strict,
                len(state.events)+1, self.q_budget, edit_slots, self.hazard_weights,
                .05 if self.policy == "hard_hazard_gate" else 0.)
        picked, reserved = select_hard(
            list(editable), resolved, fresh, state.last_write, state.last_query,
            len(state.events) + 1, self.q_budget, 0 if oracle else edit_slots,
            "hard_noedit" if oracle or state.seat_audit else self.policy, state.selection_seed,
            previous_strict=state.previous_strict,
        )
        if state.seat_audit:
            picked, reserved = state.seat_audit.choose(
                picked, editable, resolved, fresh, state.last_write, state.last_query,
                len(state.events)+1, self.q_budget, "llada21", state.selection_seed, state.previous_strict)
            if state.seat_audit.current:
                fingerprint = digest(state.x, state.cache.to_legacy_cache(), state.resident,
                    state.rng_cpu, state.rng_cuda, state.block, state.step_in_block, state.post_steps,
                    state.committed_end, state.context_version, state.prompt_length, state.last_write,
                    state.last_query, sorted(fresh), sorted(state.previous_strict), resolved)
                state.seat_audit.begin(fingerprint, state.resident[15], state.block * 32)
        if oracle:
            from .shadow import screened_swap
            picked, reserved = screened_swap(picked, resolved, fresh, editable,
                lambda rows, pool, stage: self._shadow_probe(state, rows, pool, stage))
        picked, reserved = list(picked), list(reserved)
        if (len(picked) != min(self.q_budget, len(editable))
                or picked != sorted(set(picked))
                or not set(picked).issubset(editable)
                or not fresh.issubset(picked)
                or not set(reserved).issubset(set(picked) - fresh)):
            raise RuntimeError("select_hard returned invalid B32 positions")
        return picked, reserved

    def _shadow_probe(self, state, rows, pool, stage):
        # Diagnostic forwards read the live pre-step tokens and resident KV;
        # they never update that cache, the token canvas, or the real RNG.
        start = state.block * 32
        pos = torch.tensor(rows, dtype=torch.long, device=state.x.device) + start
        scratch = replace(state, events=[], last_query=None)
        cache = _SelectedCache(state.resident, pos, start, start+32, shadow=True)
        logits = self._execute(scratch, pos, start+32, cache, "diagnostic", stage, False)
        proposed, confidence = self._sample(scratch, logits)
        old = state.x.index_select(1, pos)[0]
        ok = ((old != state.mask_id) & (proposed[0] != state.mask_id) & (proposed[0] != old)
              & (confidence[0] > state.config["editing_threshold"]))
        pool = set(pool)
        hits = {i: float(c) for i, hit, c in zip(rows, ok.tolist(), confidence[0].tolist())
                if hit and i in pool}
        event, = scratch.events
        event.update(diagnostic=True, hard_clock=len(state.events)+1,
                     screened_candidates=len(pool), screen_hits=len(hits))
        state.events.append(event)
        return hits

    def _sample(self, state, logits):
        with REFERENCE_RNG_LOCK, torch.random.fork_rng(devices=self._rng_devices()):
            torch.set_rng_state(state.rng_cpu)
            if self._rng_devices():
                torch.cuda.set_rng_state(state.rng_cuda, self.model.device)
            proposed, confidence = self.model._sample_with_temperature_topk_topp(
                logits, temperature=0.0, top_k=None, top_p=None)
            state.rng_cpu = torch.get_rng_state().clone()
            if self._rng_devices():
                state.rng_cuda = torch.cuda.get_rng_state(self.model.device).clone()
        return proposed, confidence

    @torch.inference_mode()
    def step(self, state):
        if not isinstance(state, SparseRequestState) or state.model_kind != self.kind:
            raise ValueError("Request/model mismatch")
        self.last_logits = None
        if state.phase != "denoise":
            action = super().step(state)
            if action == "commit":
                state.history, state.resident, state.force_mask_next = None, None, False
                state.gate_evidence = None
                state.last_write = state.last_query = None
                state.previous_strict.clear()
            return action

        start = state.block * 32
        end = start + 32
        cur = state.x[:, start:end]
        before = cur[0].tolist()
        resolved = [t != state.mask_id for t in before]
        count = resolved.count(False)
        if not count:
            state.post_steps += 1
            if state.post_steps > state.config.get("max_post_steps", 16):
                state.phase = "commit"
                return self.step(state)

        first = state.history is None
        if first:
            state.history = History(resolved, [None] * 32, [0] * 32, set(), set())
            if self.policy in _GATED:
                state.gate_evidence = _GateEvidence()
            elif self.policy in _HARD:
                state.last_write, state.last_query = [-1] * 32, [-1] * 32
        gate = state.gate_evidence
        hard = self.policy in _HARD
        if hard:
            hard_reserved = []
        if gate is not None:
            tags = gate.categories(state, before, start)
            reserved, released = [], 0
        if not count or first or self.q_budget == 32:
            stage = "verify" if not count else "bootstrap" if first else "dense"
            picked = list(range(32))
            logits = self._forward(state, start, end, False, "denoise", stage)
        else:
            stage = "selected"
            editable = range(max(0, state.prompt_length - start), 32)
            if hard:
                picked, hard_reserved = self._hard_pick(state, editable, resolved)
            elif gate is not None:
                picked, reserved, released = self._gate_pick(state, editable, start, tags)
            else:
                picked = self._pick(state, editable, start)

        local = torch.tensor(picked, dtype=torch.long, device=state.x.device)
        if stage == "selected":
            positions = local + start
            cache = _SelectedCache(state.resident, positions, start, end)
            logits = self._execute(state, positions, end, cache, "denoise", stage, False)
        if self.capture_logits:
            self.last_logits = logits.detach().clone()
        proposed, confidence = self._sample(state, logits)
        if not bool(torch.isfinite(confidence).all()):
            raise RuntimeError(f"Nonfinite confidence in {stage} at block {state.block}")
        old = cur.index_select(1, local)
        masked = old == state.mask_id
        selected_masks = sum(not resolved[i] for i in picked)
        threshold = state.config.get("confidence_threshold", state.config.get("threshold"))
        accept = masked & (confidence > threshold)
        if selected_masks and not bool(accept.any()):
            order = confidence.masked_fill(~masked, -torch.inf)
            accept[0, torch.topk(order[0], k=1).indices] = True
        edit = (~masked & ((local[None] + start) >= state.prompt_length)
                & (proposed != old) & (confidence > state.config["editing_threshold"]))
        cur.index_copy_(1, local, torch.where(accept | edit, proposed, old))

        after = cur[0].tolist()
        if gate is not None:
            gate.record(state, tags, start, picked, proposed, before, after, reserved, released)
        h = state.history
        changed = {i for i in picked if before[i] != after[i]}
        now_resolved = [t != state.mask_id for t in after]
        if state.seat_audit:
            state.events[-1]["trajectory_sha256"] = canvas_digest(after, now_resolved)
            state.events[-1]["accepted_sha256"] = canvas_digest(after, now_resolved, accepted_only=True)
            if state.seat_audit.current:
                state.seat_audit.end(state.events[-1], after, now_resolved, before, resolved, state.resident[15])
                state.events[-1]["intervention"]["post_rng_sha256"] = digest(state.rng_cpu, state.rng_cuda)
        fresh = {i for i in changed if now_resolved[i]}
        certainty = h.certainty
        if hard:
            event = state.events[-1]
            if state.placebo_exchange is not None:
                event["placebo_exchange"] = state.placebo_exchange
                state.placebo_exchange = None
            # Writes happen after Q at the same tick; last_write >= last_query
            # detects an input row whose latest write has not been queried.
            for i in changed:
                state.last_write[i] = event["hard_clock"]
            strict = {i for i in fresh if resolved[i]}
            state.previous_strict = strict
            if self.policy == "hard_oracle1" and not set(hard_reserved) <= strict:
                raise RuntimeError("Exact-width shadow swap failed to reproduce a strict T2T")
            event.update(
                hard_strict_edits=len(strict),
                reserved_extra_edit_positions=[start+i for i in hard_reserved],
                reserved_strict_edit_positions=[start+i for i in hard_reserved if i in strict],
            )
        else:
            certainty = h.certainty.copy()
            for i, c in zip(picked, confidence[0].tolist()):
                certainty[i] = c  # previous proposal confidence, also for newly written tokens
        state.history = History(now_resolved, certainty,
                                [h.streak[i]+1 if now_resolved[i] and i not in fresh else 0
                                 for i in range(32)], changed, fresh)
        # Legacy quotas reserve a mask seat after a pass without mask queries.
        # Hard policies use their configured split and never displace a write.
        state.force_mask_next = bool(count and not selected_masks)
        state.step_in_block += 1
        edited = int(edit.sum())
        state.events[-1].update(masked_before=count, selected_masked_before=selected_masks,
                                accepted=int(accept.sum()), edited=edited,
                                masked_after=now_resolved.count(False), step_in_block=state.step_in_block,
                                post_steps=state.post_steps)
        if not count and not edited:
            state.phase = "commit"
        return "denoise"


def qualify_full_budget(model, ids, cfg, gen_length, seed, mask_id, eos_ids, *,
                        max_actions=4096, rtol=0.0, atol=0.0, reference=None):
    """Fail closed against the same batch-one native route as trace_native.py.

    Optionally supply a native archive row with output_ids/events/calls.
    Compare exact decisions, full canvas, RNG and counters per action; compare
    logits every denoise and committed KV every prefill/commit at rtol/atol.
    Returns a small JSON-compatible success report. Does not run a low-Q arm.
    """
    challenger = LLaDASparseKernel(model, q_budget=32)
    batcher = ForwardBatcher(model, "llada21", 1)
    facade = ModelFacade(batcher, "llada_sparse_qualification", seed)
    native = SerialKernel(facade, "llada21")
    native.capture_logits = challenger.capture_logits = True
    args = (ids, cfg, gen_length, seed, mask_id, eos_ids)

    def run():
        a, b = native.create(*args), challenger.create(*args)
        commits = 0
        for tick in range(1, max_actions+1):
            prev = len(a.events)
            action = native.step(a)
            if challenger.step(b) != action:
                raise AssertionError(f"Full-budget action mismatch at action {tick}")
            for name in ("x", "rng_cpu", "rng_cuda"):
                if not torch.equal(getattr(a, name), getattr(b, name)):
                    raise AssertionError(f"Full-budget {name} mismatch at action {tick}")
            for name in ("phase", "block", "step_in_block", "post_steps", "committed_end",
                         "context_version"):
                if getattr(a, name) != getattr(b, name):
                    raise AssertionError(f"Full-budget {name} mismatch at action {tick}")
            if len(a.events) != len(b.events) or a.cache.get_seq_length() != b.cache.get_seq_length():
                raise AssertionError(f"Full-budget work/KV length mismatch at action {tick}")
            if len(a.events) > prev:
                ne, se, call = a.events[-1], b.events[-1], facade.calls[-1]
                if any(se[k] != v for k, v in ne.items()) or (
                        se["actual_q"], se["physical_q"], se["F"], se["visible_kv"]) != (
                            call["rows"], call["physical_rows"], 1, call["kv_rows"]):
                    raise AssertionError(f"Full-budget events/work mismatch at action {tick}")
            if action == "denoise":
                torch.testing.assert_close(challenger.last_logits, native.last_logits,
                                           rtol=rtol, atol=atol)
            if action in ("prefill", "commit"):
                torch.testing.assert_close(b.cache.to_legacy_cache(), a.cache.to_legacy_cache(),
                                           rtol=rtol, atol=atol)
                commits += action == "commit"
            if a.phase == "done":
                output = challenger.output(b)[0].tolist()
                if reference is not None and (
                        output != reference["output_ids"] or a.events != reference["events"]
                        or any([c[k] for c in facade.calls] != [c[k] for c in reference["calls"]]
                               for k in ("rows", "physical_rows", "batch"))):
                    raise AssertionError("Full-budget result differs from frozen native archive")
                return dict(actions=tick, forwards=sum(e["F"] for e in b.events),
                            actual_q=sum(e["actual_q"] for e in b.events),
                            physical_q=sum(e["physical_q"] for e in b.events),
                            max_q=max((e["physical_q"] for e in b.events), default=0),
                            commits=commits, output_ids=output)
        raise RuntimeError("Qualification action limit exceeded")

    return batcher.run([run])[0]

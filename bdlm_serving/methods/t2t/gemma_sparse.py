"""Batch-one query selection inside Transformers 5.12 DiffusionGemma.generate.

The neural forward (including self-conditioning, MoE and LM head) receives Q
rows. Attention reads the native encoder prefix and *all* B resident canvas
KV slots; skipped decoder KV is stale. The checkpoint's global entropy bound,
temperature schedule, multinomial and renoise draw still operate on B rows,
using the last measured RAW logits for skipped rows. This is not a sparse
sampler. Only Q rows may change canvas/acceptance/draft/self-conditioning:
off-Q random draws are consumed but discarded, including re-noising of MASKs.
MASK is a boolean state, never a vocabulary ID.

Each canvas starts full. A sparse stop is a request for a full native step at
the next clock, not permission to commit. The last clock is always full. The
outer generate commits that step's argmax, not sampled IDs or acceptance bits.
At full budget, native denoising and SDPA run unmodified and no RNG is added.
"""
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from random import Random
from types import SimpleNamespace

import torch

from .planner import History, allocate
from ...runtime.diffusiongemma import observe
from ...runtime.diffusiongemma_optimus import CAP, Resident, resident_attention


from .frontier import HARD_POLICIES as _HARD_POLICIES
from .seat_audit import GEMMA_CONDITIONAL, GEMMA_FACTORIAL, MODES, SeatAudit, canvas_digest, digest


@dataclass
class GemmaLedger:
    events: list = field(default_factory=list)  # observe(): includes pure prompt prefill
    trace: list = field(default_factory=list)   # one record per decoder forward

    @property
    def work(self):
        """Observed (stage, physical Q), excluding only the first encoder."""
        if not self.events:
            return []
        if self.events[0][0] != "encoder":
            raise RuntimeError("Missing native prefill")
        result, j = [], 0
        for phase, q in self.events[1:]:
            if phase == "encoder":
                result.append(("reencode", q))
            elif phase == "decoder" and j < len(self.trace) and self.trace[j]["q"] == q:
                result.append((self.trace[j]["stage"], q))
                j += 1
            else:
                raise RuntimeError("Observed work disagrees with sparse decoder ledger")
        if j != len(self.trace):
            raise RuntimeError("Decoder call missing from observer")
        return result


class _Ranking(History):
    def rank(self, eligible, signal):
        if self.policy == "fresh_random":
            ordered = sorted(eligible)
            if ordered and ordered[0] in self.fresh:
                self.rng.shuffle(ordered)
            return ordered
        if not eligible or eligible[0] in self.fresh or self.policy == "targeted":
            return super().rank(eligible, signal)
        ordered = sorted(eligible)
        if self.policy == "random":
            self.rng.shuffle(ordered)
        return ordered


def _pick(h, budget, fresh, mask, policy, signal, rng):
    rank = _Ranking(h.resolved, h.certainty, h.streak, h.changed, h.fresh)
    rank.policy, rank.rng = policy, rng
    # No-edit means no reserved edit seats, NOT a veto on accepting an edit.
    if policy in ("noedit", "fresh_random"):
        mask = budget - fresh
    picked = allocate(rank, range(len(h.resolved)), budget, fresh, mask, signal)
    if len(picked) != budget or picked != sorted(set(picked)):
        raise RuntimeError("Invalid physical Q allocation")
    return picked


def _hard_pick(h, width, q_budget, fresh_slots, mask_slots, policy, seed,
               clock, last_write, last_query, previous_strict=()):
    """Delegate sparse seats to frontier.select_hard, enforcing mandatory fresh.

    The selector sees only measured per-block metadata. Every position that was
    fresh on the preceding step stays in the physical Q. If that mandatory set
    alone exceeds q_budget, it must be returned whole and the excess is reported.
    """
    from .frontier import select_hard
    edit_slots = q_budget - fresh_slots - mask_slots
    if policy == "hard_noedit":
        edit_slots = 0
    mandatory = sorted(h.fresh)
    if len(mandatory) > q_budget:
        return mandatory, [], len(mandatory), len(mandatory), len(mandatory) - q_budget, edit_slots
    positions, reserved = select_hard(
        list(range(width)), list(h.resolved), set(mandatory),
        list(last_write), list(last_query), clock, q_budget, edit_slots,
        policy, seed, **({"previous_strict": previous_strict} if policy.startswith("hard_trigger") else {}))
    positions, reserved = list(positions), list(reserved)
    assert positions == sorted(set(positions)), "select_hard positions must be sorted unique"
    assert all(type(i) is int and 0 <= i < width for i in positions), \
        "select_hard position outside block"
    assert len(reserved) == len(set(reserved)), "select_hard reserved positions must be unique"
    assert all(type(i) is int and 0 <= i < width for i in reserved), \
        "select_hard reserved position outside block"
    assert set(reserved) <= set(positions), "select_hard reserved positions are not selected"
    assert set(mandatory) <= set(positions), "select_hard dropped mandatory fresh rows"
    assert len(positions) == q_budget, \
        f"select_hard filled {len(positions)} of {q_budget} sparse seats"
    return positions, sorted(reserved), len(mandatory), len(mandatory), 0, edit_slots


def _hard_touch(last_write, last_query, picked, accepted, before, after,
                resolved, clock):
    """Record latest queries and accepted/rewritten/revoked changes only.

    Unaccepted draft noise never updates last_write because it was not accepted.
    """
    for i in picked:
        last_query[i] = clock
    for i in picked:
        if (bool(accepted[i]) != bool(resolved[i])
                or (accepted[i] and before[i] != after[i])):
            last_write[i] = clock


class _CanvasKV(Resident):
    """Reuse Optimus scatter/concatenation, with full-canvas visibility."""
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.bypass = False

    def begin(self, picked, device):
        self.select(0, picked, device)
        if not self.initialized[:self.width].all():
            raise RuntimeError("Full bootstrap required before sparse KV reads")
        # Optimus normally stops at the rightmost query. Gemma's canvas does not.
        self.end = self.width

    def extend(self, layer, q, k, v):
        if layer not in self.buffers:
            if self.index.numel() != self.width:
                raise RuntimeError("Layer missed its full initialization")
            # A single current canvas, not CAP resident positions per layer.
            self.buffers[layer] = tuple(
                t.new_empty((*t.shape[:-2], self.width, t.shape[-1])) for t in (k, v)
            )
        return super().extend(layer, q, k, v)

    def candidate_state(self, row, *, clone=False):
        views = [t[..., row:row+1, :] for layer in sorted(self.buffers) for t in self.buffers[layer]]
        return [v.clone() for v in views] if clone else views

    def other_state(self, row):
        return [view for layer in sorted(self.buffers) for t in self.buffers[layer]
                for view in (t[..., :row, :], t[..., row+1:, :])]


def _advance(h, before, after, accepted, picked, margin):
    """Only consecutive accepted, changed sample IDs count as T2T."""
    edits = {i for i in picked if h.resolved[i] and accepted[i] and before[i] != after[i]}
    admits = {i for i in picked if not h.resolved[i] and accepted[i]}
    revoked = {i for i in picked if h.resolved[i] and not accepted[i]}
    fresh = edits | admits
    certainty = h.certainty.copy()
    for i, c in zip(picked, margin, strict=True):
        certainty[i] = c
    streak = [s + 1 if a and i not in fresh else 0
              for i, (s, a) in enumerate(zip(h.streak, accepted, strict=True))]
    return History(accepted, certainty, streak, fresh | revoked, fresh), (len(admits), len(edits), len(revoked))


def _publish_scores(raw, index, measured, freeze_index):
    previous = raw.index_select(1, freeze_index) if freeze_index is not None else None
    raw.index_copy_(1, index, measured)
    if previous is not None:
        raw.index_copy_(1, freeze_index, previous)


@contextmanager
def sparse_gemma(model, q_budget, fresh_slots=0, mask_slots=0, policy="targeted",
                 signal="uncertain", selection_seed=0):
    """Wrap **one** native generate call on an exclusive eval model.

    Caller seeds torch for sampling. selection_seed seeds only a private Python
    RNG. Integer seats: edit=q_budget-fresh_slots-mask_slots; noedit gives all
    edit seats to MASK. `ledger.work` is ready after generate. Its length and
    sum are generation F and physical Q. Decoder-only rows are in ledger.trace.
    hard_* policies delegate seat choice to frontier.select_hard; all preceding
    step fresh rows are mandatory and overflow is recorded, never truncated.
    Requires unpadded batch one, dynamic cache, native SDPA, single device and
    the checkpoint's default logits processor/sampler/stopping configuration.
    """
    width = model.config.canvas_length
    if not (all(type(n) is int for n in (q_budget, fresh_slots, mask_slots))
            and 1 <= q_budget <= width <= CAP and min(fresh_slots, mask_slots) >= 0
            and fresh_slots + mask_slots <= q_budget):
        raise ValueError("Invalid Gemma canvas budget or quota")
    refresh_only = policy == "hard_trigger_refresh"
    if policy not in ("targeted", "random", "noedit", "fresh_random", "hard_trigger_refresh", *_HARD_POLICIES, *MODES, *GEMMA_FACTORIAL, GEMMA_CONDITIONAL) or signal not in (
            "uncertain", "young", "neighbor", "recent_then_uncertain"):
        raise ValueError("Invalid Gemma selection policy or ranking signal")
    if model.training:
        raise ValueError("Expected an eval model")
    dense_only = q_budget == width
    all_rows = list(range(width))
    native = model._denoising_step
    rng = Random(selection_seed)
    bank = None if dense_only else _CanvasKV(width)
    h = raw = previous_clock = block = None
    last_write = last_query = None
    hard_clock = 0
    previous_strict = set()
    need_check = False
    closed = True
    ledger = GemmaLedger()
    factorial = policy in GEMMA_FACTORIAL
    if (factorial or policy == GEMMA_CONDITIONAL) and (width, q_budget, fresh_slots, mask_slots) != (256, 32, 0, 32):
        raise ValueError("This one-seat policy requires Gemma 32:0:32")
    audit = SeatAudit("hard_once_rank") if factorial else SeatAudit(policy) if policy in (*MODES, GEMMA_CONDITIONAL) else None

    def step(**kw):
        nonlocal h, raw, previous_clock, block, need_check, closed
        nonlocal last_write, last_query, hard_clock, previous_strict
        canvas, prior_draft = kw["current_canvas"], kw["argmax_canvas"]
        pos, sc = kw["decoder_position_ids"], kw["self_conditioning_logits"]
        prefix = kw["input_ids"].shape[1]
        clock = int(kw["cur_step"])
        if canvas.shape != (1, width) or pos.shape != canvas.shape or clock < 1:
            raise ValueError("Expected one native canvas with explicit positions and clock")
        if kw["past_key_values"].is_compileable or any(m is not None for m in kw["mask_mapping"].values()):
            raise ValueError("Only an unpadded dynamic encoder cache is supported")
        new_block = block != prefix
        if new_block:
            if not closed or (block is not None and prefix != block + width):
                raise RuntimeError("Unexpected canvas advance or second request")
            h = History([False]*width, [None]*width, [0]*width, set(), set())
            block, need_check, closed, raw = prefix, False, False, None
            last_write, last_query, hard_clock = [-1]*width, [-1]*width, 0
            previous_strict = set()
            stage = "bootstrap"
        elif closed or clock != previous_clock - 1:
            raise RuntimeError("Non-native denoising clock")
        elif dense_only:
            stage = "dense"
        elif need_check or clock == 1:
            stage = "verify"
        else:
            stage = "selected"
        previous_clock = clock
        hard = policy in _HARD_POLICIES or audit is not None or refresh_only
        if hard:
            hard_clock += 1
        selected = stage == "selected"
        prev_fresh = sorted(h.fresh) if hard else []
        mandatory_fresh = len(prev_fresh)
        reserved, overflow = [], 0
        candidate = stale_kv = None
        frozen_rows = []
        edit_slots = q_budget - fresh_slots - mask_slots
        if selected and hard:
            (picked, reserved, mandatory_fresh, fresh_selected,
             overflow, edit_slots) = _hard_pick(
                h, width, q_budget, fresh_slots, mask_slots,
                "hard_noedit" if audit else "hard_trigger" if refresh_only else policy,
                selection_seed, hard_clock, last_write, last_query, previous_strict)
            if refresh_only and reserved:
                neutral, *_ = _hard_pick(h, width, q_budget, 0, q_budget, "hard_noedit",
                    selection_seed, hard_clock, last_write, last_query, previous_strict)
                frozen_rows = sorted(set(picked) - set(neutral))
                if (not set(frozen_rows) <= set(reserved) or set(frozen_rows) & h.fresh
                        or any(not h.resolved[i] for i in frozen_rows)):
                    raise RuntimeError("Refresh-only mode may freeze only newly reserved old rows")
            if audit:
                picked, reserved = audit.choose(picked, all_rows, h.resolved, h.fresh, last_write,
                    last_query, hard_clock, q_budget, "gemma", selection_seed, previous_strict)
                if audit.current:
                    stopper = kw["diffusion_stopping_criteria"]
                    layers = kw["past_key_values"].layers
                    fingerprint = digest(canvas, prior_draft, kw["input_ids"], pos, sc, raw,
                        kw["finished_denoising"], kw["sampler"].accepted_token_mask,
                        getattr(stopper, "argmax_canvas_history", None), torch.get_rng_state(),
                        torch.cuda.get_rng_state(canvas.device), block, clock, hard_clock, last_write,
                        last_query, h.resolved, sorted(h.fresh), sorted(previous_strict),
                        [(i, *bank.buffers[i]) for i in sorted(bank.buffers)],
                        [(layer.keys, layer.values, getattr(layer, "cumulative_length", None))
                         for layer in layers])
                    audit.begin(fingerprint, bank.buffers[15], 0)
                    if factorial:
                        candidate = audit.current["candidate"]
                        assert candidate in picked and candidate not in h.fresh and h.resolved[candidate]
                        stale_kv = bank.candidate_state(candidate, clone=True)
                        if policy[-1] == "d":
                            frozen_rows = [candidate]
                        audit.current.update(pre_row_sha256=digest(stale_kv),
                            persist_kv=policy[-2] == "k", use_score=policy[-1] == "k")
        else:
            picked = (_pick(h, q_budget, fresh_slots, mask_slots, policy, signal, rng)
                      if selected else all_rows)
            fresh_selected = sum(i in picked for i in prev_fresh)
        q = len(picked)
        n_events = len(ledger.events)
        if bank is not None:
            bank.begin(picked, canvas.device)
            index = bank.index
            freeze_index = index.new_tensor(frozen_rows) if frozen_rows else None
            forward = kw["decoder_forward"]

            def measured(**args):
                nonlocal raw
                if selected:
                    if raw is None or sc is None:
                        raise RuntimeError("Sparse decoder requires a full initialization")
                    for name in ("decoder_input_ids", "self_conditioning_logits", "decoder_position_ids"):
                        args[name] = args[name].index_select(1, index)
                out = forward(**args)
                if (out.logits.shape[:2] != (1, q)
                        or len(bank.visited) != len(model.model.decoder.layers)):
                    raise RuntimeError("Selected decoder Q did not reach all layers and head")
                if selected:
                    _publish_scores(raw, index, out.logits, freeze_index)
                else:
                    raw = out.logits.detach().clone()
                # The pinned native temperature processor is out-of-place.
                return SimpleNamespace(logits=raw) if selected else out
            kw["decoder_forward"] = measured

        sampler = kw["sampler"]
        if selected:
            active = torch.zeros_like(canvas, dtype=torch.bool)
            active.index_fill_(1, index, True)
            if frozen_rows:
                active.index_fill_(1, freeze_index, False)
            old_acceptance = sampler.accepted_token_mask.clone()
            stopper = kw["diffusion_stopping_criteria"]
            if stopper is not None:
                # The stable draft actually exposed to generate freezes off-Q.
                kw["diffusion_stopping_criteria"] = lambda draft, scores: stopper(
                    torch.where(active, draft, prior_draft), scores)
        after, draft, new_sc, finished = native(**kw)
        if ledger.events[n_events:] != [("decoder", q)]:
            raise RuntimeError("Denoising step did not execute exactly its planned physical Q")
        proposed_stop = bool(finished.item())
        if selected:
            after = torch.where(active, after, canvas)
            draft = torch.where(active, draft, prior_draft)
            new_sc = torch.where(active.unsqueeze(-1), new_sc, sc)
            sampler.accepted_token_mask = torch.where(active, sampler.accepted_token_mask, old_acceptance)
            finished = torch.zeros_like(finished)
        if candidate is not None:
            row = bank.candidate_state(candidate)
            p = audit.current
            p["forward_row_sha256"] = digest(row)
            if not p["persist_kv"]:
                for storage, old in zip(row, stale_kv, strict=True):
                    storage.copy_(old)
            p["after_row_sha256"] = digest(row)
            if p["after_row_sha256"] != p["forward_row_sha256" if p["persist_kv"] else "pre_row_sha256"]:
                raise RuntimeError("Candidate persistence control failed")
            p["post_other_kv_sha256"] = digest(bank.other_state(candidate))
            p["post_nonkv_sha256"] = digest(raw, after, draft, new_sc, sampler.accepted_token_mask,
                proposed_stop, finished, getattr(stopper, "argmax_canvas_history", None))
            p["action_rows"] = torch.where(active[0])[0].tolist()
        need_check = selected and proposed_stop
        closed = not selected and (clock == 1 or proposed_stop)
        before_ids, after_ids = canvas[0].tolist(), after[0].tolist()
        accepted = sampler.accepted_token_mask[0].tolist()
        # Only newly measured margins are features, never stale rows rescaled
        # at the current temperature. Dense bridge and hard policies have no
        # planner to feed, so skip the extra topk there.
        if dense_only or hard:
            margin = [None] * q
        else:
            scores = new_sc[0].index_select(0, index) if selected else new_sc[0]
            top = scores.topk(2, dim=-1).values.float()
            margin = (top[:, 0] - top[:, 1]).tolist()
        previous = h
        h, (admitted, edited, revoked) = _advance(h, before_ids, after_ids, accepted, picked, margin)
        if hard:
            _hard_touch(last_write, last_query, picked, accepted, before_ids,
                        after_ids, previous.resolved, hard_clock)
            strict_edits = {i for i in picked
                            if previous.resolved[i] and accepted[i]
                            and before_ids[i] != after_ids[i]}
            reserved_strict = sorted(set(reserved) & strict_edits)
            previous_strict = strict_edits
        record = dict(
            block=block, clock=clock, stage=stage, q=q, positions=picked.copy(),
            selected_fresh=sum(i in previous.fresh for i in picked),
            available_fresh=len(previous.fresh),
            edited_in_fresh=sum(accepted[i] and before_ids[i] != after_ids[i]
                                for i in picked if i in previous.fresh),
            selected_mask=sum(not previous.resolved[i] for i in picked),
            admitted=admitted, edited=edited, revoked=revoked, resolved=sum(accepted),
            stop_candidate=proposed_stop, finished=bool(finished.item()), terminal_clock=clock == 1,
        )
        if hard:
            record.update(
                reserved_extra_edit_positions=reserved,
                reserved_strict_edit_positions=reserved_strict,
                hard_strict_edits=len(strict_edits),
                mandatory_fresh=mandatory_fresh,
                fresh_selected=fresh_selected,
                edit_slots=edit_slots,
                overflow=overflow,
                hard_clock=hard_clock,
            )
        if refresh_only:
            record["kv_refresh_only_positions"] = frozen_rows
            if any(before_ids[i] != after_ids[i] or previous.resolved[i] != accepted[i] for i in frozen_rows):
                raise RuntimeError("KV-only refresh leaked a token or acceptance update")
        if audit:
            record["trajectory_sha256"] = canvas_digest(after_ids, accepted)
            record["accepted_sha256"] = canvas_digest(after_ids, accepted, accepted_only=True)
            if audit.current:
                audit.end(record, after_ids, accepted, before_ids, previous.resolved, bank.buffers[15])
                record["intervention"]["post_rng_sha256"] = digest(torch.get_rng_state(),
                    torch.cuda.get_rng_state(canvas.device))
        ledger.trace.append(record)
        return after, draft, new_sc, finished

    with ExitStack() as stack:
        ledger.events = stack.enter_context(observe(model))
        if bank is not None:
            stack.enter_context(resident_attention(model, bank))
        # Restore an existing instance-level instrument as well as class methods.
        if "_denoising_step" in vars(model):
            stack.callback(setattr, model, "_denoising_step", vars(model)["_denoising_step"])
        else:
            stack.callback(delattr, model, "_denoising_step")
        model._denoising_step = step
        yield ledger
        if not closed:
            raise RuntimeError("Generation returned without a full final decoder check")
        _ = ledger.work

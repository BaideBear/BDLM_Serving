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
    if policy == "noedit":
        mask = budget - fresh
    picked = allocate(rank, range(len(h.resolved)), budget, fresh, mask, signal)
    if len(picked) != budget or picked != sorted(set(picked)):
        raise RuntimeError("Invalid physical Q allocation")
    return picked


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


@contextmanager
def sparse_gemma(model, q_budget, fresh_slots=0, mask_slots=0, policy="targeted",
                 signal="uncertain", selection_seed=0):
    """Wrap **one** native generate call on an exclusive eval model.

    Caller seeds torch for sampling. selection_seed seeds only a private Python
    RNG. Integer seats: edit=q_budget-fresh_slots-mask_slots; noedit gives all
    edit seats to MASK. `ledger.work` is ready after generate. Its length and
    sum are generation F and physical Q. Decoder-only rows are in ledger.trace.
    Requires unpadded batch one, dynamic cache, native SDPA, single device and
    the checkpoint's default logits processor/sampler/stopping configuration.
    """
    width = model.config.canvas_length
    if not (all(type(n) is int for n in (q_budget, fresh_slots, mask_slots))
            and 1 <= q_budget <= width <= CAP and min(fresh_slots, mask_slots) >= 0
            and fresh_slots + mask_slots <= q_budget):
        raise ValueError("Invalid Gemma canvas budget or quota")
    if policy not in ("targeted", "random", "noedit") or signal not in (
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
    need_check = False
    closed = True
    ledger = GemmaLedger()

    def step(**kw):
        nonlocal h, raw, previous_clock, block, need_check, closed
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
        selected = stage == "selected"
        picked = (_pick(h, q_budget, fresh_slots, mask_slots, policy, signal, rng)
                  if selected else all_rows)
        q = len(picked)
        n_events = len(ledger.events)
        if bank is not None:
            bank.begin(picked, canvas.device)
            index = bank.index
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
                    raw.index_copy_(1, index, out.logits)
                else:
                    raw = out.logits.detach().clone()
                # The pinned native temperature processor is out-of-place.
                return SimpleNamespace(logits=raw) if selected else out
            kw["decoder_forward"] = measured

        sampler = kw["sampler"]
        if selected:
            active = torch.zeros_like(canvas, dtype=torch.bool)
            active.index_fill_(1, index, True)
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
        need_check = selected and proposed_stop
        closed = not selected and (clock == 1 or proposed_stop)
        before_ids, after_ids = canvas[0].tolist(), after[0].tolist()
        accepted = sampler.accepted_token_mask[0].tolist()
        # Only newly measured margins are features, never stale rows rescaled
        # at the current temperature. Dense bridge has no planner to feed.
        if dense_only:
            margin = [None] * q
        else:
            scores = new_sc[0].index_select(0, index) if selected else new_sc[0]
            top = scores.topk(2, dim=-1).values.float()
            margin = (top[:, 0] - top[:, 1]).tolist()
        previous = h
        h, (admitted, edited, revoked) = _advance(h, before_ids, after_ids, accepted, picked, margin)
        ledger.trace.append(dict(
            block=block, clock=clock, stage=stage, q=q, positions=picked.copy(),
            selected_fresh=sum(i in previous.fresh for i in picked),
            selected_mask=sum(not previous.resolved[i] for i in picked),
            admitted=admitted, edited=edited, revoked=revoked, resolved=sum(accepted),
            stop_candidate=proposed_stop, finished=bool(finished.item()), terminal_clock=clock == 1,
        ))
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

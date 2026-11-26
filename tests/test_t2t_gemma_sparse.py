"""CPU-only: no checkpoint or GPU. Integration cases use the pinned 5.12 mixin."""
import inspect
import random
from random import Random
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from bdlm_serving.methods.t2t.gemma_sparse import (
    _advance, _CanvasKV, _pick, GemmaLedger, sparse_gemma)
from bdlm_serving.methods.t2t.planner import History
from bdlm_serving.runtime.diffusiongemma import observe


def test_equal_budget_controls_share_only_fresh_ranking():
    h = History([True]*7 + [False]*2, [1, 0, 8, 7, 6, 5, 4, None, None],
                [0]*9, {0, 1}, {0, 1})
    def pick(policy, seed=0, budget=4):
        return _pick(h, budget, 1, 1, policy, "uncertain", Random(seed))
    assert pick("targeted") == [1, 5, 6, 7]
    # Two MASKs, then the next fresh and no resolved backfill.
    assert pick("noedit") == [0, 1, 7, 8]
    assert pick("noedit", budget=5) == [0, 1, 2, 7, 8]
    assert pick("random", 43) == pick("random", 43)
    seen = {tuple(pick("random", seed)) for seed in range(20)}
    assert len(seen) > 1
    assert all(1 in p and 7 in p and 0 not in p and 8 not in p and len(p) == 4 for p in seen)


def test_no_noise_or_argmax_labels():
    h = History([True, True, False, False, True], [0]*5, [3]*5, set(), set())
    new, counts = _advance(h, [9]*5, [3, 3, 9, 3, 9],
                           [True, False, True, False, True], [0, 1, 2, 3], [1, 2, 3, 4])
    assert counts == (1, 1, 1)  # admission even when noise ID equals accepted ID
    assert new.fresh == {0, 2} and new.changed == {0, 1, 2}
    assert new.streak == [0, 0, 0, 0, 4] and new.certainty == [1, 2, 3, 4, 0]


def test_zero_edit_pair_only_changes_fresh_ranking():
    h = History([True]*6 + [False]*4, [4, 3, 2, 1, 0, 0, None, None, None, None],
                [0]*10, {0, 1, 2, 3}, {0, 1, 2, 3})
    def choose(policy, seed):
        return set(_pick(h, 5, 2, 3, policy, "uncertain", Random(seed)))
    assert choose("noedit", 0) == {2, 3, 6, 7, 8}
    choices = [choose("fresh_random", seed) for seed in range(30)]
    assert all(s - h.fresh == {6, 7, 8} and len(s & h.fresh) == 2 for s in choices)
    assert choices[0] == choose("fresh_random", 0)
    assert len({tuple(sorted(s)) for s in choices}) == 6


def test_resident_full_visible_tail_and_multiple_layers():
    bank = _CanvasKV(5)
    with pytest.raises(RuntimeError, match="bootstrap"):
        bank.begin([0], "cpu")
    bank.begin(list(range(5)), "cpu")
    t = torch.arange(7).view(1, 1, 7, 1)
    for i in range(2):
        kv = t + i * 10
        a, b = bank.extend(i, kv[..., 2:, :], kv, kv + 100)
        assert torch.equal(a, kv) and torch.equal(b, kv + 100)
    bank.begin([1, 3], "cpu")
    for i in range(2):
        kv = torch.tensor([14, 15, 50, 51]).view(1, 1, 4, 1) + i * 10
        a, b = bank.extend(i, kv[..., 2:, :], kv, kv + 100)
        assert a.flatten().tolist() == [x + i * 10 for x in [14, 15, 2, 50, 4, 51, 6]]
        assert torch.equal(b, a + 100)


def test_factorial_snapshot_is_row_local_across_layers_and_parser_is_gemma_q32_only():
    from bdlm_serving.methods.t2t.seat_audit import GEMMA_FACTORIAL, digest
    from bdlm_serving.methods.t2t.protocol import Arm
    bank = _CanvasKV(4)
    bank.buffers = {j: tuple(torch.arange(24).reshape(1, 2, 4, 3) + j + k
                             for k in range(2)) for j in (2, 0)}
    for i in (0, 2, 3):
        views = bank.candidate_state(i)
        snap = bank.candidate_state(i, clone=True)
        rest = digest(bank.other_state(i))
        for v in views:
            v.add_(7)
        assert digest(views) != digest(snap)
        assert digest(bank.other_state(i)) == rest
        for v, old in zip(views, snap, strict=True):
            v.copy_(old)
        assert digest(views) == digest(snap)
    for mode in GEMMA_FACTORIAL:
        assert Arm.parse(f"32:0:32:{mode}", 256).selection == mode
        for q, width in ((4, 32), (32, 32), (16, 256)):
            with pytest.raises(ValueError):
                Arm.parse(f"{q}:0:{q}:{mode}", width)


def test_newly_added_kv_only_rows_keep_raw_scores_before_any_sampler_operation():
    from bdlm_serving.methods.t2t.gemma_sparse import _publish_scores
    from bdlm_serving.methods.t2t.protocol import Arm
    old = torch.arange(18).reshape(1, 6, 3).clone()
    idx = torch.tensor([0, 2, 4, 5])
    measured = 100 + old.index_select(1, idx)
    for freeze in (None, torch.tensor([2, 5])):
        raw = old.clone()
        _publish_scores(raw, idx, measured, freeze)
        allowed = set(idx.tolist()) - (set(freeze.tolist()) if freeze is not None else set())
        for i in range(6):
            assert torch.equal(raw[:, i], old[:, i] + (100 if i in allowed else 0))
    assert Arm.parse("32:0:28:hard_trigger_refresh", 256).edit == 4
    with pytest.raises(ValueError):
        Arm.parse("32:0:28:hard_trigger_refresh", 32)


class _Encoder(nn.Module):
    def create_masks_for_generate(self, **kw):
        return {}

    def forward(self, input_ids, past_key_values, **kw):
        past_key_values.length += input_ids.shape[1]
        return NS(past_key_values=past_key_values)


class _Decoder(nn.Module):
    def __init__(self, registry):
        super().__init__()
        self.registry = registry
        self.embed_tokens = nn.Embedding(8, 1)
        self.layers = nn.ModuleList([nn.Module(), nn.Module()])
        for i, layer in enumerate(self.layers):
            layer.self_attn = nn.Identity()
            layer.self_attn.config = NS(_attn_implementation="sdpa")
            layer.self_attn.layer_idx = i
            layer.self_attn.is_causal = False
            layer.self_attn.num_key_value_groups = 1
        self.calls = []
        self.uncertain = False
        self.last_prefix, self.tick = None, 0

    def create_diffusion_decoder_attention_mask(self, decoder_attention_mask, **kw):
        return {"full_attention": None if decoder_attention_mask.all() else decoder_attention_mask}

    def forward(self, decoder_input_ids, self_conditioning_logits, decoder_position_ids,
                decoder_attention_mask, past_key_values):
        prefix = past_key_values.length
        if prefix != self.last_prefix:
            self.tick, self.last_prefix = 0, prefix
        self.tick += 1
        q = decoder_input_ids.shape[1]
        assert decoder_position_ids.shape == (1, q)
        assert self_conditioning_logits is None or self_conditioning_logits.shape == (1, q, 8)
        self.calls.append((decoder_position_ids.clone(), decoder_input_ids.clone()))
        x = self.embed_tokens(decoder_input_ids).unsqueeze(1)
        k = torch.nn.functional.pad(x, (0, 0, prefix, 0))
        for layer in self.layers:
            attn = layer.self_attn
            y, _ = self.registry[attn.config._attn_implementation](
                attn, x, k, k, decoder_attention_mask["full_attention"], is_causal=False)
            assert y.shape == (1, q, 1, 1)
        # Native stopper says stop on clock 2's call, but the next full
        # verification changes *all* drafts and must keep generating.
        token = 2 if self.tick < 3 else 3
        ids = torch.full_like(decoder_input_ids, token)
        if prefix > 2:
            ids[decoder_position_ids == prefix + 2] = 1  # EOS in canvas two
        logits = torch.zeros(1, q, 8)
        logits.scatter_(2, ids[..., None], 0.25 if self.uncertain else 30.)
        return NS(logits=logits)


@pytest.fixture
def tiny():
    g = pytest.importorskip("transformers.models.diffusion_gemma.generation_diffusion_gemma")
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import ALL_ATTENTION_FUNCTIONS

    class Native(nn.Module, g.DiffusionGemmaGenerationMixin):
        def __init__(self):
            super().__init__()
            self.config = NS(canvas_length=6, image_token_id=-99, text_config=NS(vocab_size=8))
            self.generation_config = g.DiffusionGemmaGenerationConfig(
                max_denoising_steps=7, pad_token_id=0, eos_token_id=[1],
                stability_threshold=1, confidence_threshold=0.005)
            self.dtype = torch.float32
            self.model = nn.Module()
            self.model.encoder = _Encoder()
            self.model.decoder = _Decoder(ALL_ATTENTION_FUNCTIONS)

        def _prepare_cache_for_generation(self, **kw):
            return NS(is_compileable=False, length=0)

        def forward(self, **kw):
            return self.model.decoder(**kw)

    return Native().eval()


PROMPT = torch.tensor([[4, 5]])


@pytest.mark.parametrize("width,q", [(6, 4), (6, 6), (256, 256)])
@pytest.mark.parametrize("fully_accepted", [False, True])
def test_physical_prefix_underfilled_swap_executes_same_width(tiny, width, q, fully_accepted):
    tiny.config.canvas_length = width
    native = tiny._denoising_step

    def controlled_acceptance(**kw):
        _, draft, sc, done = native(**kw)
        # Bootstrap admits a strict prefix and a noncontiguous old row, but
        # the next pass has no writes so those positions become non-fresh.
        mask = ([True] * width if fully_accepted else
                [True, True, True, False, True] + [False] * (width - 5))
        kw["sampler"].accepted_token_mask[:] = torch.tensor([mask], dtype=torch.bool)
        return kw["current_canvas"].clone(), draft, sc, torch.zeros_like(done)

    tiny._denoising_step = controlled_acceptance
    for mode, mask_slots in (("hard_prefix_zero", q), ("hard_prefix_frontier_right", q - 2)):
        tiny.model.decoder.calls.clear()
        torch.manual_seed(13)
        with sparse_gemma(tiny, q, 0, mask_slots, policy=mode) as ledger:
            tiny.generate(input_ids=PROMPT, max_new_tokens=width, max_denoising_steps=4)
        event = ledger.trace[2]
        if fully_accepted:
            assert ledger.work == [("bootstrap", width), ("selected", width),
                                   ("verify", width), ("verify", width)]
            assert event["prefix_empty_verify_plan"]["zero_width"] == 0
            assert "prefix_plan" not in event
            expected = list(range(width))
        else:
            zero = list(range(3, width))
            assert ledger.work == [("bootstrap", width), ("selected", q),
                                   ("selected", len(zero)), ("verify", width)]
            audit = event["prefix_plan"]
            assert audit["zero_rows"] == zero and audit["underfill"] == q - len(zero)
            expected = (zero if mode == "hard_prefix_zero" else
                        [1, 2, 3, *range(5, width - 1)])
            assert event["positions"] == audit["selected_rows"] == expected
        assert tiny.model.decoder.calls[2][0][0].tolist() == [2 + i for i in expected]


@pytest.mark.parametrize("seed", [1, 13, 149])
def test_full_budget_parity_output_events_and_rng(tiny, seed):
    tiny.model.decoder.uncertain = True
    torch.manual_seed(seed)
    with observe(tiny) as native_events:
        native = tiny.generate(input_ids=PROMPT, max_new_tokens=12)
    end_rng = torch.get_rng_state()
    torch.manual_seed(seed)
    with sparse_gemma(tiny, 6, policy="random", selection_seed=997) as ledger:
        result = tiny.generate(input_ids=PROMPT, max_new_tokens=12)
    assert torch.equal(result.sequences, native.sequences)
    assert torch.equal(result.tokens_per_forward, native.tokens_per_forward)
    assert torch.equal(end_rng, torch.get_rng_state()) and ledger.events == native_events
    assert len(ledger.work) == len(native_events) - 1
    assert sum(q for _, q in ledger.work) == sum(q for _, q in native_events[1:])
    assert "_denoising_step" not in vars(tiny)


def test_bootstrap_sparse_false_convergence_verify_and_outer_eos(tiny):
    with sparse_gemma(tiny, 2, fresh_slots=1, mask_slots=1, policy="noedit") as ledger:
        out = tiny.generate(input_ids=PROMPT, max_new_tokens=12)
    # First selected step is a stop candidate. Verify discovers 6 real edits,
    # then the second candidate is confirmed. Native outer loop handles EOS.
    expected = [("bootstrap", 6), ("selected", 2), ("verify", 6),
                ("selected", 2), ("verify", 6)]
    assert ledger.work == expected + [("reencode", 6)] + expected
    assert ledger.events[0] == ("encoder", 2)
    assert out.sequences.tolist() == [[4, 5, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0, 0, 0]]
    assert ledger.trace[1]["stop_candidate"] and not ledger.trace[1]["finished"]
    assert not ledger.trace[2]["finished"] and ledger.trace[2]["edited"] == 6
    assert ledger.trace[4]["finished"] and ledger.trace[5]["block"] == 8
    for event, (pos, _) in zip(ledger.trace, tiny.model.decoder.calls, strict=True):
        assert pos[0].tolist() == [event["block"] + i for i in event["positions"]]


def test_clock_cap_forces_verify_and_unselected_state_is_frozen(tiny):
    tiny.model.decoder.uncertain = True
    snapshots = []
    # An outer wrapper sees what native generate sees, not the speculative
    # dense sampler result prior to off-Q masking.
    with sparse_gemma(tiny, 2, 0, 1) as ledger:
        adapter = tiny._denoising_step
        def snapshot(**kw):
            after, draft, sc, done = adapter(**kw)
            snapshots.append((after.clone(), draft.clone(), sc.clone(),
                              kw["sampler"].accepted_token_mask.clone()))
            return after, draft, sc, done
        tiny._denoising_step = snapshot
        out = tiny.generate(input_ids=PROMPT, max_new_tokens=6, max_denoising_steps=4)
    assert ledger.work == [("bootstrap", 6), ("selected", 2), ("selected", 2), ("verify", 6)]
    assert ledger.trace[-1]["terminal_clock"] and not ledger.trace[-1]["stop_candidate"]
    assert torch.equal(out.sequences[:, 2:], snapshots[-1][1])  # argmax, not sampled canvas
    for t in [1, 2]:
        skipped = sorted(set(range(6)) - set(ledger.trace[t]["positions"]))
        for previous, current in zip(snapshots[t-1], snapshots[t], strict=True):
            assert torch.equal(previous[:, skipped], current[:, skipped])


def test_restore_on_error(tiny):
    original = tiny._denoising_step
    tiny._denoising_step = original
    attns = [l.self_attn for l in tiny.model.decoder.layers]
    cfg = [a.config for a in attns]
    with pytest.raises(ValueError, match="unpadded"):
        with sparse_gemma(tiny, 2):
            tiny.generate(input_ids=PROMPT, attention_mask=torch.tensor([[0, 1]]), max_new_tokens=6)
    assert tiny._denoising_step is original
    assert all(a.config is c for a, c in zip(attns, cfg, strict=True))
    assert not tiny.model.decoder._forward_pre_hooks and not tiny.model.encoder._forward_pre_hooks


def test_public_api_signature_and_ledger_shape():
    """Pinned to the orchestrator call in methods/t2t/online.py."""
    signature = inspect.signature(sparse_gemma)
    assert [p.name for p in signature.parameters.values()] == [
        "model", "q_budget", "fresh_slots", "mask_slots", "policy", "signal", "selection_seed"]
    assert [signature.parameters[n].default for n in ("fresh_slots", "mask_slots")] == [0, 0]
    assert (signature.parameters["policy"].default, signature.parameters["signal"].default,
            signature.parameters["selection_seed"].default) == ("targeted", "uncertain", 0)
    ledger = GemmaLedger()
    assert ledger.events == [] and ledger.trace == [] and ledger.work == []
    assert isinstance(GemmaLedger.work, property)


def test_selected_rows_gather_canvas_sc_and_cache_raw_logits(tiny):
    """Physical Q gather, cached off-Q logits/acceptance and the native Q decision."""
    tiny.model.decoder.uncertain = True
    # A per-forward offset makes a recomputed row differ from a cached one.
    real_forward, counter = tiny.forward, [0]
    def forward(**kw):
        counter[0] += 1
        return NS(logits=real_forward(**kw).logits + 0.01 * counter[0])
    tiny.forward = forward
    raw = []
    inner = tiny._prepare_logits_processor(tiny.generation_config)
    def processor(input_ids, scores, **kw):
        raw.append((int(kw["cur_step"]), scores.detach().clone()))
        return inner(input_ids, scores, **kw)
    tiny._prepare_logits_processor = lambda cfg, lp=None: processor
    native_masks = []
    real_prepare = tiny._prepare_sampler
    def prepare(cfg):
        sampler = real_prepare(cfg)
        real_accept = sampler.accept_canvas
        def accept(current, denoiser, logits, step):
            out = real_accept(current, denoiser, logits, step)
            native_masks.append(sampler.accepted_token_mask.clone())
            return out
        sampler.accept_canvas = accept
        return sampler
    tiny._prepare_sampler = prepare
    canvases, scs, sc_out, masks = [], [], [], []
    torch.manual_seed(5)
    with sparse_gemma(tiny, 2, policy="targeted", selection_seed=3) as ledger:
        # Spy on the sparse step itself so the recorded mask is post-restore.
        adapter = tiny._denoising_step
        def spy(**kw):
            canvases.append(kw["current_canvas"].clone())
            sc = kw["self_conditioning_logits"]
            scs.append(None if sc is None else sc.clone())
            out = adapter(**kw)
            sc_out.append(out[2].clone())
            masks.append(kw["sampler"].accepted_token_mask.clone())
            return out
        tiny._denoising_step = spy
        tiny.generate(input_ids=PROMPT, max_new_tokens=6, max_denoising_steps=4)
    assert ledger.work[0] == ("bootstrap", 6) and ledger.work[-1][1] == 6
    assert any(event["stage"] == "selected" for event in ledger.trace)
    assert set(ledger.trace[0]) == {
        "block", "clock", "stage", "q", "positions", "selected_fresh", "selected_mask",
        "admitted", "edited", "revoked", "resolved", "stop_candidate", "finished", "terminal_clock",
        "available_fresh", "edited_in_fresh"}
    for i, event in enumerate(ledger.trace):
        _, ids = tiny.model.decoder.calls[i]
        if event["stage"] != "selected":
            continue
        active = torch.zeros(6, dtype=torch.bool)
        active[event["positions"]] = True
        # The decoder saw exactly the canvas rows at the picked positions.
        assert ids[0].tolist() == canvases[i][0, event["positions"]].tolist()
        # Their self-conditioning is the previous step's output at those rows.
        assert torch.equal(scs[i][0, event["positions"]], sc_out[i - 1][0, event["positions"]])
        # Off-Q raw logits are bitwise the previous step's; Q rows were recomputed.
        current, previous = raw[i][1], raw[i - 1][1]
        assert torch.equal(current[0][~active], previous[0][~active])
        assert not torch.equal(current[0][active], previous[0][active])
        # Off-Q acceptance is frozen; Q rows keep the native sampler's decision.
        assert torch.equal(masks[i][0][~active], masks[i - 1][0][~active])
        assert torch.equal(masks[i][0][active], native_masks[i][0][active])


SLACK_POLICIES = ("hard_prefix_slack_right", "hard_prefix_slack_random", "hard_prefix_slack_oldest")


def _controlled(tiny, width, *, fully_accepted=False):
    native = tiny._denoising_step

    def controlled(**kw):
        _, draft, sc, done = native(**kw)
        # Bootstrap admits a strict prefix and a noncontiguous old row; the
        # next pass has no writes so those positions become non-fresh.
        mask = ([True] * width if fully_accepted else
                [True, True, True, False, True] + [False] * (width - 5))
        kw["sampler"].accepted_token_mask[:] = torch.tensor([mask], dtype=torch.bool)
        return kw["current_canvas"].clone(), draft, sc, torch.zeros_like(done)

    tiny._denoising_step = controlled


def _gemma_run(tiny, width, q, mask_slots, policy):
    tiny.model.decoder.calls.clear()
    torch.manual_seed(13)
    with sparse_gemma(tiny, q, 0, mask_slots, policy=policy) as ledger:
        tiny.generate(input_ids=PROMPT, max_new_tokens=width, max_denoising_steps=4)
    return ledger


@pytest.mark.parametrize("policy", SLACK_POLICIES)
@pytest.mark.parametrize("k", [1, 2, 4])
def test_slack_prefix_positive_slack_projects_real_rows(tiny, policy, k):
    width, q = 6, 6
    tiny.config.canvas_length = width
    _controlled(tiny, width)
    zero = _gemma_run(tiny, width, q, q, "hard_prefix_zero")
    slack = _gemma_run(tiny, width, q, q - k, policy)
    ze = next(e for e in zero.trace if e.get("prefix_plan", {}).get("zero_width") == 3)
    se = next(e for e in slack.trace if e.get("prefix_plan", {}).get("zero_width") == 3)
    za, sa = ze["prefix_plan"], se["prefix_plan"]
    assert sa["zero_rows"] == za["zero_rows"] == [3, 4, 5]
    assert sa["eligible_prefix_rows"] == za["eligible_prefix_rows"] == [0, 1, 2]
    assert sa["evicted_rows"] == []
    assert set(za["selected_rows"]) <= set(sa["selected_rows"])
    assert sa["added_rows"] == sorted(set(sa["selected_rows"]) - set(za["selected_rows"]))
    assert len(sa["added_rows"]) == min(k, 3, 3)
    assert set(sa["added_rows"]) <= {0, 1, 2}
    if policy == "hard_prefix_slack_right":
        assert sa["added_rows"] == ([2] if k == 1 else [1, 2] if k == 2 else [0, 1, 2])
    assert sa["actual_width"] == 3 + len(sa["added_rows"])
    assert sa["underfill"] == q - sa["actual_width"]
    assert se["positions"] == sa["selected_rows"]
    assert se["q"] == sa["actual_width"]
    assert se["selected_mask"] == len(za["zero_mask_rows"])
    assert se["selected_fresh"] == len(za["fresh_rows"])
    assert zero.trace.index(ze) == slack.trace.index(se)
    idx = slack.trace.index(se)
    assert tiny.model.decoder.calls[idx][0][0].tolist() == [2 + i for i in sa["selected_rows"]]
    assert tiny.model.decoder.calls[idx][0][0].numel() == sa["actual_width"]
    # One widened selected forward; no extra call merely to spend K.
    assert len(slack.work[:idx + 1]) == len(zero.work[:idx + 1])
    assert slack.work[idx][1] == za["actual_width"] + len(sa["added_rows"])


@pytest.mark.parametrize("policy", SLACK_POLICIES)
@pytest.mark.parametrize("k", [1, 2])
def test_slack_prefix_zero_slack_is_identical_no_extra_calls(tiny, policy, k):
    width, q = 6, 4
    tiny.config.canvas_length = width
    _controlled(tiny, width)
    zero = _gemma_run(tiny, width, q, q, "hard_prefix_zero")
    before = random.getstate()
    slack = _gemma_run(tiny, width, q, q - k, policy)
    assert random.getstate() == before
    ze, se = zero.trace[1], slack.trace[1]
    assert ze["stage"] == se["stage"] == "selected"
    assert se["prefix_plan"]["added_rows"] == []
    assert se["prefix_plan"]["selected_rows"] == ze["prefix_plan"]["selected_rows"]
    assert se["prefix_plan"]["actual_width"] == ze["prefix_plan"]["actual_width"] == 4
    assert se["positions"] == ze["positions"] and se["q"] == ze["q"] == 4
    # The zero-slack step is locally identical; later steps may use real slack.
    assert slack.work[:2] == zero.work[:2]
    assert [(e["stage"], e["q"], e["positions"]) for e in slack.trace[:2]] == [
        (e["stage"], e["q"], e["positions"]) for e in zero.trace[:2]]


@pytest.mark.parametrize("policy", SLACK_POLICIES)
def test_slack_prefix_zero_width_uses_full_native_verify(tiny, policy):
    width, q = 6, 6
    tiny.config.canvas_length = width
    _controlled(tiny, width, fully_accepted=True)
    zero = _gemma_run(tiny, width, q, q, "hard_prefix_zero")
    slack = _gemma_run(tiny, width, q, q - 2, policy)
    assert zero.work == slack.work == [
        ("bootstrap", 6), ("selected", 6), ("verify", 6), ("verify", 6)]
    for ledger in (zero, slack):
        event = ledger.trace[2]
        audit = event["prefix_empty_verify_plan"]
        assert event["stage"] == "verify" and event["q"] == 6
        assert audit["zero_width"] == 0 and audit["actual_width"] == 0
        assert audit["added_rows"] == [] and audit["selected_rows"] == []
        assert audit["underfill"] == 6
    assert [(e["stage"], e["q"], e["positions"]) for e in slack.trace] == [
        (e["stage"], e["q"], e["positions"]) for e in zero.trace]


def test_slack_prefix_added_rows_are_read_from_the_real_canvas(tiny):
    width, q, k = 6, 6, 2
    tiny.config.canvas_length = width
    _controlled(tiny, width)
    canvases = []
    inner = tiny._denoising_step

    def spy(**kw):
        canvases.append(kw["current_canvas"].clone())
        return inner(**kw)

    tiny._denoising_step = spy
    slack = _gemma_run(tiny, width, q, q - k, "hard_prefix_slack_right")
    se = next(e for e in slack.trace if e.get("prefix_plan", {}).get("zero_width") == 3)
    idx = slack.trace.index(se)
    added = se["prefix_plan"]["added_rows"]
    assert added
    local = [se["positions"].index(i) for i in added]
    assert tiny.model.decoder.calls[idx][1][0, local].tolist() == canvases[idx][0, added].tolist()

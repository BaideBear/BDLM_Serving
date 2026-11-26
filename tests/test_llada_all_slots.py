"""Adversarial per-forward contracts for the five-slot LLaDA executor.

Target::

    experiments.t2t.serving_slots.llada_all_slots.AllSlotKernel(
        model, *, prefix_slots=None, prefix_mode="zero", fuse_growth=True)

The kernel keeps the TopTransition request contract:

* ``create(input_ids, config, gen_length, seed, mask_id, eos_ids, cached=True)``
  returns a state; the first ``step`` performs the prompt prefill and one
  generation forward in one offer.
* ``step(state, offer)`` consumes exactly one integer offer in
  ``(4, 8, 16, 32, 64)`` and returns its audited event, or ``None`` at EOS/end.
* ``run(state, next_budget, max_actions=8192)`` and ``output(state)`` behave as
  in ``TopTransitionKernel``.

Event contract exercised below (all positions are absolute, including the
prompt offset):

* ``physical_q == actual_q == len(absolute_positions)`` and
  ``layer_physical_q == [physical_q] * layers``; one real model call per event.
* ``refresh_only_positions`` is projected but never sampled.  At low offers it
  is exactly the parked tail's mandatory M2T/T2T/T2M rows.
* ``sampled_q == len(sampled_positions) ==
  len(absolute_positions) - len(refresh_only_positions)``.
* ``overflow == max(0, physical_q - offer)`` and
  ``underfill == max(0, offer - physical_q)``.
* ``overflow_reasons`` splits overflow into a dense protocol pass
  (bootstrap/verify/promotion/commit) and mandatory refresh.
* ``mandatory_head_positions`` and ``mandatory_tail_positions`` partition
  ``refreshed_pending`` and are subsets of ``absolute_positions``.

Low offers 4/8/16 share one budget: mandatory head and tail rows are reserved
first, then MASK/backfill head rows, and only true slack may be spent on the
strictly block-local optional prefix.  Offers 32/64 keep the original
TopTransition routes (Q32 head plus tail-refresh overflow; Q64 pair).  No
forward pads to the offer, and no parked tail row is sampled below Q64.

The tiny fixtures are imported from the existing CPU suites:
``CPUModel`` from ``test_llada_top_transition``, the mask-taping ``_Toy`` from
``test_llada_dual``, and the embedding-taping ``_Toy`` from
``test_t2t_llada_sparse``.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from experiments.t2t.serving_slots.llada_all_slots import AllSlotKernel

from bdlm_serving.methods.t2t.planner import History
from test_llada_dual import _Toy as DualToy
from test_llada_top_transition import CPUModel, create as top_create
from test_t2t_llada_sparse import _Toy as SparseToy
from experiments.t2t.serving_slots.llada_top_transition import TopTransitionKernel


BLOCK = 32
PAIR = 64
OFFERS = (4, 8, 16, 32, 64)
LOW = (4, 8, 16)
CFG = dict(confidence_threshold=.8, editing_threshold=.8, max_post_steps=16)


class TapedCPUModel(CPUModel):
    """CPUModel plus an attention-mask tape; no behavior change."""

    def __init__(self):
        super().__init__()
        self.masks = []

    def forward(self, *, attention_mask, **kwargs):
        self.masks.append(attention_mask.detach().clone())
        return super().forward(attention_mask=attention_mask, **kwargs)


def make(kernel, *, prompt=32, length=64, max_post=16, eos=(), fill=None,
         token=5, seed=11):
    if fill is not None:
        kernel.model.fill = set(fill)
    config = dict(CFG, max_post_steps=max_post)
    return kernel.create(torch.full((1, prompt), token), config, length,
                         seed, 0, eos)


def pending(state):
    start = state.block * BLOCK
    return {start + i for i, (w, r) in enumerate(zip(state.last_write, state.last_query))
            if w >= 0 and w >= r}


def generation(state):
    return [e for e in state.events if e["phase"] != "prefill"]


def tail_snapshot(state):
    start = (state.block + 1) * BLOCK
    return SimpleNamespace(
        x=state.x[:, start:].clone(),
        history=state.right_history,
        post=state.right_post_steps,
        last_write=list(state.last_write[BLOCK:]),
        last_query=list(state.last_query[BLOCK:]),
    )


def assert_parked_tail_unchanged(state, before):
    """A low offer may refresh tail KV but must not sample or advance the tail."""
    start = (state.block + 1) * BLOCK
    torch.testing.assert_close(state.x[:, start:], before.x)
    assert state.right_history is before.history
    assert state.right_post_steps == before.post


def assert_clock(state, offers):
    events = generation(state)
    assert state.events[0]["phase"] == "prefill"
    assert len(events) == state.offer_tick == len(offers)
    assert [e["offer_tick"] for e in events] == list(range(len(offers)))
    assert [e["q_budget"] for e in events] == list(offers)
    assert [e["previous_offer"] for e in events] == [None] + list(offers[:-1])


def assert_event(event, offer):
    positions = event["absolute_positions"]
    assert positions == sorted(set(positions))
    assert all(type(p) is int for p in positions)
    assert event["physical_q"] == event["actual_q"] == len(positions)
    assert event["layer_physical_q"] == [len(positions)] * len(event["layer_physical_q"])
    assert event["F"] == 1
    assert 0 < len(positions) <= PAIR
    refresh = event["refresh_only_positions"]
    sampled = event["sampled_positions"]
    assert refresh == sorted(set(refresh))
    assert sampled == sorted(set(sampled))
    assert set(refresh) <= set(positions)
    if event["phase"] == "commit":
        assert sampled == []
    else:
        assert set(sampled) == set(positions) - set(refresh)
    assert event["sampled_q"] == len(sampled)
    assert event["sampled_q"] == (0 if event["phase"] == "commit"
                                  else len(positions) - len(refresh))
    assert set(event["refreshed_pending"]) <= set(positions)
    assert set(refresh) <= set(event["refreshed_pending"])
    assert (event["mandatory_head_positions"] + event["mandatory_tail_positions"]
            == event["refreshed_pending"])
    assert not set(event["mandatory_head_positions"]) & set(event["mandatory_tail_positions"])
    assert event["q_budget"] == event["nominal_q"] == offer
    assert event["overflow"] == max(0, len(positions) - offer)
    assert event["underfill"] == max(0, offer - len(positions))
    assert sum(event["overflow_reasons"].values()) == event["overflow"]
    assert event["visible_kv"] >= (positions[-1] + 1 if positions else 0)
    if event["dense_reason"] is not None:
        dense = set(event["dense_required_positions"])
        assert event["dense_required_q"] == len(dense)
        assert dense <= set(positions)
        assert dense.isdisjoint(refresh)
        assert set(positions) == dense | set(refresh)


def assert_tape(model, state):
    calls = getattr(model, "calls", None)
    if calls is not None:
        assert len(calls) == len(state.events)
        for event, (positions, end) in zip(state.events, calls, strict=True):
            assert event["absolute_positions"] == positions
            assert event["visible_kv"] == end
    rows = getattr(model, "embedding_rows", None)
    if rows is not None:
        assert len(rows) == len(state.events)
        for event, width in zip(state.events, rows, strict=True):
            assert width == event["physical_q"] == event["actual_q"]
    masks = getattr(model, "masks", None)
    if masks is not None:
        assert len(masks) == len(state.events)
        for event, mask in zip(state.events, masks, strict=True):
            assert mask.shape[-2] == event["physical_q"]
            assert mask.shape[-1] == event["visible_kv"]


def assert_causal_masks(model, state, prompt_length):
    """The exact rectangular block-causal map, with head/tail isolation."""
    masks = getattr(model, "masks", None)
    if masks is None:
        return
    assert len(masks) == len(state.events)
    for event, mask in zip(state.events, masks, strict=True):
        positions = torch.tensor(event["absolute_positions"], dtype=torch.long)
        end = event["visible_kv"]
        expected = (positions[:, None] // BLOCK
                    >= torch.arange(end)[None, :] // BLOCK)[None, None]
        torch.testing.assert_close(mask, expected)
        if event["phase"] == "prefill":
            assert all(p < prompt_length for p in event["absolute_positions"])
            continue
        block = event["block"]
        head_end, tail_end = (block + 1) * BLOCK, (block + 2) * BLOCK
        for i, p in enumerate(event["absolute_positions"]):
            if p < head_end:
                assert not mask[0, 0, i, head_end:].any()
            else:
                assert not mask[0, 0, i, tail_end:].any()
        assert event["visible_kv"] <= tail_end
        assert all(block * BLOCK <= p < tail_end
                   for p in event["absolute_positions"])
        if event["stage"] == "selected":
            assert all(p >= prompt_length for p in event["absolute_positions"])


def assert_prompt_frozen(state):
    for event in state.events:
        if event["phase"] != "prefill" and event["stage"] == "selected":
            assert all(p >= state.prompt_length for p in event["absolute_positions"])


# ---------------------------------------------------------------------------
# Offer contract
# ---------------------------------------------------------------------------


def test_constructor_defaults_and_keyword_only_prefix_arguments():
    signature = inspect.signature(AllSlotKernel)
    assert [p.name for p in signature.parameters.values()] == [
        "model", "prefix_slots", "prefix_mode", "fuse_growth"]
    for name, default in (("prefix_slots", None), ("prefix_mode", "zero"),
                          ("fuse_growth", True)):
        assert signature.parameters[name].default == default
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("bad", [0, 2, 3, 5, 6, 7, 12, 24, 48, 128, -4,
                                 True, 4.0, "4", None])
def test_invalid_offer_fails_closed_without_consuming_the_clock(bad):
    model = CPUModel().eval()
    kernel = AllSlotKernel(model)
    state = make(kernel)
    with pytest.raises(ValueError, match="integer offer"):
        kernel.step(state, bad)
    assert state.events == [] and state.phase == "prefill"
    assert state.offer_tick == 0 and state.previous_offer is None
    assert not model.calls


@pytest.mark.parametrize("offer", OFFERS)
def test_each_slot_is_accepted_and_charges_one_generation_forward(offer):
    model = CPUModel().eval()
    kernel = AllSlotKernel(model)
    state = make(kernel)
    event = kernel.step(state, offer)
    assert event["q_budget"] == event["nominal_q"] == offer
    assert event["offer_tick"] == 0 and event["previous_offer"] is None
    assert state.events[0]["phase"] == "prefill"
    assert state.phase == "denoise" and state.offer_tick == 1
    assert_event(event, offer)
    assert_tape(model, state)
    if offer == PAIR:
        assert event["top_action"] == "pair_bootstrap"
        assert event["physical_q"] == 64
    else:
        assert event["top_action"] == "head_bootstrap"
        assert event["physical_q"] == 32
        assert event["overflow"] == 32 - offer
        assert event["overflow_reasons"] == (
            {"bootstrap": 32 - offer} if 32 - offer else {})


# ---------------------------------------------------------------------------
# Mixed offers, downgrade, shared budget, parked tail
# ---------------------------------------------------------------------------


def test_mixed_offers_share_one_clock_and_never_pad_to_the_offer():
    model = SparseToy(edit=False).eval()
    kernel = AllSlotKernel(model)
    state = make(kernel, prompt=34, length=96, token=3)
    offers = [64, 8, 8, 32, 64, 16, 4]
    for q in offers:
        kernel.step(state, q)
    assert_clock(state, offers)
    assert_tape(model, state)
    assert_causal_masks(model, state, state.prompt_length)
    assert_prompt_frozen(state)
    events = generation(state)
    assert [e["top_action"] for e in events] == [
        "pair_bootstrap", "head_denoise", "head_denoise", "head_denoise",
        "pair_denoise", "head_denoise", "head_denoise"]
    assert [e["physical_q"] for e in events] == [64, 8, 8, 30, 62, 16, 4]
    assert [e["sampled_q"] for e in events] == [64, 4, 8, 30, 62, 15, 4]
    assert [e["refresh_only_positions"] for e in events] == [
        [], [64, 65, 66, 67], [], [], [], [86], []]
    assert [e["overflow"] for e in events] == [0] * 7
    assert [e["underfill"] for e in events] == [0, 0, 0, 2, 2, 0, 0]
    for event in events:
        assert set(event["sampled_positions"]).isdisjoint(event["refresh_only_positions"])
        if event["q_budget"] < PAIR:
            assert all(p < (state.block + 1) * BLOCK
                       for p in event["sampled_positions"])


def test_first_q64_projects_two_real_blocks_then_downgrade_keeps_tail_parked():
    model = SparseToy(edit=False).eval()
    kernel = AllSlotKernel(model)
    state = make(kernel, prompt=34, length=96, token=3)
    first = kernel.step(state, 64)
    assert first["top_action"] == "pair_bootstrap"
    assert first["absolute_positions"] == list(range(32, 96))
    assert first["physical_q"] == first["sampled_q"] == 64
    assert first["refresh_only_positions"] == []
    assert first["blocks"] == [1, 2] and first["visible_kv"] == 96
    assert state.resident[0][0].shape[-2] == 96
    assert state.right_history is not None and state.right_post_steps == 0
    before = tail_snapshot(state)
    down = kernel.step(state, 8)
    assert down["top_action"] == "head_denoise"
    assert down["q_budget"] == 8 and down["offer_tick"] == 1
    assert down["previous_offer"] == 64
    assert down["physical_q"] == 8 and down["sampled_q"] == 4
    assert down["refresh_only_positions"] == [64, 65, 66, 67]
    assert all(32 <= p < 64 for p in down["sampled_positions"])
    assert down["visible_kv"] == 96
    assert_parked_tail_unchanged(state, before)
    up = kernel.step(state, 64)
    assert up["top_action"] == "pair_denoise"
    assert up["physical_q"] == up["sampled_q"] == 62
    assert up["refresh_only_positions"] == [] and up["blocks"] == [1, 2]
    assert_causal_masks(model, state, state.prompt_length)


def test_low_offer_reserves_mandatory_rows_before_discretionary_and_never_pads():
    model = CPUModel().eval()
    kernel = AllSlotKernel(model)
    state = make(kernel, prompt=32, length=96, fill={2})
    kernel.step(state, 64)  # pair bootstrap: 32 fresh tail rows + one head row
    before = pending(state)
    assert len(before) == 33
    event = kernel.step(state, 8)
    assert event["top_action"] == "head_denoise"
    assert event["physical_q"] == 33
    assert event["sampled_q"] == 1
    assert event["refresh_only_positions"] == list(range(64, 96))
    assert event["mandatory_head_positions"] == sorted(p for p in before if p < 64)
    assert event["mandatory_tail_positions"] == list(range(64, 96))
    assert event["sampled_positions"] == sorted(p for p in before if p < 64)
    assert event["overflow"] == 25 and event["underfill"] == 0
    assert event["overflow_reasons"] == {"mandatory_refresh": 25}
    assert pending(state) == set()
    clean = kernel.step(state, 8)
    assert clean["physical_q"] == clean["sampled_q"] == 8
    assert clean["refresh_only_positions"] == [] and clean["underfill"] == 0
    assert clean["overflow_reasons"] == {}


def test_low_offer_underfills_instead_of_fabricating_rows():
    model = CPUModel().eval()
    kernel = AllSlotKernel(model)
    state = make(kernel, prompt=60, length=4, fill={2})
    kernel.step(state, 4)  # prefill + full head bootstrap
    event = kernel.step(state, 16)
    assert event["top_action"] == "head_denoise"
    assert event["physical_q"] == event["sampled_q"] == 4
    assert event["underfill"] == 12 and event["overflow"] == 0
    assert event["absolute_positions"] == [60, 61, 62, 63]
    assert event["layer_physical_q"] == [4, 4]
    assert model.calls[-1] == ([60, 61, 62, 63], 64)


def test_parked_tail_m2t_t2t_and_t2m_are_refresh_only_never_sampled():
    model = CPUModel().eval()
    kernel = AllSlotKernel(model)
    state = make(kernel, prompt=32, length=96, fill={2})
    kernel.step(state, 64)  # pair bootstrap
    # Pass A writes one strict T2T (64: 1 -> 2) and one T2M (65: 1 -> MASK).
    model.propose_at = {64: 2, 65: 0}
    kernel.step(state, 64)
    model.propose_at = {}
    before = tail_snapshot(state)
    tail_tokens = state.x[0, 64:67].clone()
    low1 = kernel.step(state, 8)
    assert low1["refresh_only_positions"] == [64, 65]
    assert low1["physical_q"] == 8 and low1["sampled_q"] == 6
    assert all(32 <= p < 64 for p in low1["sampled_positions"])
    assert model.draws[-1].numel() == 6
    assert_parked_tail_unchanged(state, before)
    torch.testing.assert_close(state.x[0, 64:67], tail_tokens)
    assert state.resident[0][0][0, 0, 64:66, 0].tolist() == tail_tokens[:2].tolist()
    assert low1["mandatory_tail_positions"] == [64, 65]
    # Pass B writes one M2T (65: MASK -> 1) and one strict T2T (66: 1 -> 2).
    model.propose_at = {65: 1, 66: 2}
    kernel.step(state, 64)
    model.propose_at = {}
    before = tail_snapshot(state)
    tail_tokens = state.x[0, 64:67].clone()
    low2 = kernel.step(state, 8)
    assert low2["refresh_only_positions"] == [65, 66]
    assert low2["physical_q"] == 8 and low2["sampled_q"] == 6
    assert all(32 <= p < 64 for p in low2["sampled_positions"])
    assert model.draws[-1].numel() == 6
    assert_parked_tail_unchanged(state, before)
    torch.testing.assert_close(state.x[0, 64:67], tail_tokens)
    assert state.resident[0][0][0, 0, 64:67, 0].tolist() == tail_tokens.tolist()
    assert low2["mandatory_tail_positions"] == [65, 66]


def test_tail_debt_can_consume_the_whole_offer_without_a_head_pass():
    model = CPUModel().eval()
    kernel = AllSlotKernel(model)
    state = make(kernel, prompt=32, length=96, fill={2})
    kernel.step(state, 64)
    model.propose_at = {64: 2, 65: 0}
    kernel.step(state, 64)
    model.propose_at = {}
    kernel.step(state, 8)
    model.propose_at = {p: 2 for p in range(65, 70)}
    kernel.step(state, 64)
    model.propose_at = {}
    # Fixture: a clean head, five still-pending tail rows, offer 4.
    if state.left_history is not None:
        state.left_history.fresh = set()
    state.last_write[:BLOCK] = [-1] * BLOCK
    state.last_query[:BLOCK] = [-1] * BLOCK
    before = tail_snapshot(state)
    draws, history = len(model.draws), state.left_history
    event = kernel.step(state, 4)
    assert event["top_action"] == "tail_dirty_drain"
    assert event["physical_q"] == 5 and event["sampled_q"] == 0
    assert event["sampled_positions"] == []
    assert event["refresh_only_positions"] == [65, 66, 67, 68, 69]
    assert event["mandatory_head_positions"] == []
    assert event["mandatory_tail_positions"] == [65, 66, 67, 68, 69]
    assert event["overflow"] == 1 and event["underfill"] == 0
    assert len(model.draws) == draws
    assert state.left_history is history
    assert_parked_tail_unchanged(state, before)


def test_head_verify_at_low_offer_shares_the_forward_with_tail_refresh():
    model = CPUModel().eval()
    kernel = AllSlotKernel(model)
    state = make(kernel, prompt=32, length=96, fill={2})
    kernel.step(state, 64)  # pair bootstrap
    # Leave exactly one head MASK; nine tail rows are edited below.
    state.x[0, 32:64] = 1
    state.x[0, 63] = 0
    state.left_history = History([True] * 31 + [False], [None] * BLOCK,
                                 [0] * BLOCK, set(), set())
    state.left_done = False
    state.left_post_steps = 0
    state.last_write[:BLOCK] = [-1] * BLOCK
    state.last_query[:BLOCK] = [-1] * BLOCK
    model.propose_at = {p: 2 for p in range(64, 73)}
    model.propose_at[63] = 1
    kernel.step(state, 64)
    model.propose_at = {}
    assert not state.left_done  # the pass started with one MASK
    before = tail_snapshot(state)
    tail_tokens = state.x[0, 64:73].clone()
    event = kernel.step(state, 4)
    assert event["top_action"] == "head_verify"
    assert event["physical_q"] == 41 and event["sampled_q"] == 32
    assert event["refresh_only_positions"] == list(range(64, 73))
    assert event["dense_reason"] == "verify" and event["dense_required_q"] == 32
    assert event["overflow"] == 37
    assert event["overflow_reasons"] == {"verify": 28, "mandatory_refresh": 9}
    assert all(32 <= p < 64 for p in event["sampled_positions"])
    assert model.draws[-1].numel() == 32
    assert_parked_tail_unchanged(state, before)
    torch.testing.assert_close(state.x[0, 64:73], tail_tokens)


def test_fixed_lower_offer_control_is_head_only_and_charges_dense_phases():
    model = CPUModel().eval()
    kernel = AllSlotKernel(model)
    state = make(kernel, prompt=60, length=4, fill={2})
    offers = [4] * 7
    for q in offers:
        if state.phase == "done":
            break
        kernel.step(state, q)
    assert state.phase == "done"
    assert_clock(state, offers[:state.offer_tick])
    assert_tape(model, state)
    events = generation(state)
    assert all(e["q_budget"] == 4 and len(e["blocks"]) == 1 for e in events)
    assert state.right_history is None
    assert kernel.output(state).tolist() == [[1, 1, 1, 1]]
    for event in events:
        assert_event(event, 4)
        if event["top_action"] in ("head_bootstrap", "head_verify", "commit_head"):
            assert event["physical_q"] == 32 and event["overflow"] == 28
            assert sum(event["overflow_reasons"].values()) == 28
        else:
            assert event["top_action"] == "head_denoise"
            assert event["physical_q"] == event["sampled_q"] <= 4
            assert event["underfill"] == 4 - event["physical_q"]
            assert event["refresh_only_positions"] == [] and event["overflow"] == 0
    assert {e["top_action"] for e in events} >= {
        "head_bootstrap", "head_denoise", "head_verify", "commit_head"}


@pytest.mark.parametrize("offer", [32, 64])
def test_fixed_32_and_64_match_pinned_top_transition_exactly(offer):
    models = [CPUModel().eval(), CPUModel().eval()]
    all_slot = AllSlotKernel(models[0])
    top = TopTransitionKernel(models[1])
    a = top_create(all_slot, prompt=44, length=52)
    b = top_create(top, prompt=44, length=52)
    all_slot.run(a, lambda _: offer)
    top.run(b, lambda _: offer)
    torch.testing.assert_close(all_slot.output(a), top.output(b))
    torch.testing.assert_close(a.rng_cpu, b.rng_cpu)
    assert a.committed_end == b.committed_end
    assert a.cache.get_seq_length() == b.cache.get_seq_length()
    assert models[0].calls == models[1].calls
    assert len(a.events) == len(b.events)
    for ea, eb in zip(a.events, b.events, strict=True):
        for key in ("phase", "stage", "absolute_positions", "physical_q", "actual_q",
                    "layer_physical_q", "visible_kv", "refresh_only_positions",
                    "blocks", "top_action", "q_budget", "overflow", "sampled_q"):
            assert ea.get(key) == eb.get(key), (offer, key)


# ---------------------------------------------------------------------------
# Causality, absolute positions and physical projection
# ---------------------------------------------------------------------------


def test_block_causal_mask_keeps_tail_invisible_to_head_and_prompt_frozen():
    model = DualToy().eval()
    kernel = AllSlotKernel(model)
    state = kernel.create(torch.full((1, 34), 3), CFG, 96, 29, 0, ())
    for q in (64, 8, 32, 64, 4, 16):
        kernel.step(state, q)
    assert_causal_masks(model, state, state.prompt_length)
    assert_prompt_frozen(state)
    assert_tape(model, state)
    for event in state.events:
        if event["phase"] == "prefill":
            continue
        block = event["block"]
        assert all(block * BLOCK <= p < (block + 2) * BLOCK
                   for p in event["absolute_positions"])
        assert event["visible_kv"] <= (block + 2) * BLOCK


def test_mask_tape_with_tail_refresh_keeps_head_rows_isolated():
    model = TapedCPUModel().eval()
    kernel = AllSlotKernel(model)
    state = make(kernel, prompt=32, length=96, fill={2})
    kernel.step(state, 64)  # pair bootstrap; the tail resolves fully
    event = kernel.step(state, 8)  # one head row + 32 tail refresh rows
    assert event["physical_q"] == 33 and event["sampled_q"] == 1
    assert event["refresh_only_positions"] == list(range(64, 96))
    assert_causal_masks(model, state, state.prompt_length)
    assert_tape(model, state)


def test_every_forward_is_one_real_projection_with_exact_rows():
    model = SparseToy(edit=False).eval()
    kernel = AllSlotKernel(model)
    state = kernel.create(torch.full((1, 34), 3), CFG, 96, 29, 0, ())
    for q in (64, 8, 32, 64, 4, 16):
        kernel.step(state, q)
    assert len(model.embedding_rows) == len(state.events)
    for event, rows in zip(state.events, model.embedding_rows, strict=True):
        assert rows == event["physical_q"] == event["actual_q"]
        assert event["layer_physical_q"] == [rows] * 2
        assert event["F"] == 1
    assert_tape(model, state)


# ---------------------------------------------------------------------------
# Strictly block-local optional prefix
# ---------------------------------------------------------------------------


def prefix_fixture(kernel, *, prompt=44, length=20):
    assert prompt == 44 and length == 20  # the forced rows below are absolute
    state = make(kernel, prompt=prompt, length=length, fill={2})
    bootstrap = kernel.step(state, 16)
    assert bootstrap["top_action"] == "head_bootstrap"
    assert state.resident is not None
    # Generated rows 44..51 are resolved; 52..63 stay MASK; prompt is frozen.
    state.x[0, 44:52] = 1
    state.x[0, 52:64] = 0
    state.left_history = History([True] * 20 + [False] * 12, [None] * BLOCK,
                                 [0] * BLOCK, set(), set())
    state.last_write[:BLOCK] = [-1] * BLOCK
    state.last_query[:BLOCK] = [-1] * BLOCK
    state.left_done = False
    state.left_post_steps = 0
    return state


def test_optional_prefix_is_strictly_block_local_and_only_fills_true_slack():
    model = CPUModel().eval()
    kernel = AllSlotKernel(model, prefix_slots={16: 4}, prefix_mode="right")
    state = prefix_fixture(kernel)
    before = state.x.clone()
    event = kernel.step(state, 16)
    plan = event["prefix_plan"]
    assert event["top_action"] == "head_denoise"
    assert event["physical_q"] == event["sampled_q"] == 16
    assert event["refresh_only_positions"] == []
    assert plan["quota"] == 4 and plan["prefix_end"] == 52
    assert plan["eligible_prefix_rows"] == list(range(44, 52))
    assert plan["added_rows"] == [48, 49, 50, 51]
    assert plan["slack"] == 4
    assert plan["zero_rows"] == list(range(52, 64))
    assert plan["selected_rows"] == list(range(48, 64))
    assert event["absolute_positions"] == event["sampled_positions"] == list(range(48, 64))
    assert set(plan["added_rows"]) <= set(plan["eligible_prefix_rows"])
    assert all(44 <= p < 64 for p in plan["added_rows"])
    assert all(before[0, p] != 0 for p in plan["added_rows"])  # resolved before Q

    # No true slack: the quota cannot displace mandatory/MASK head rows.
    tight_kernel = AllSlotKernel(CPUModel().eval(), prefix_slots={8: 4}, prefix_mode="right")
    tight = prefix_fixture(tight_kernel)
    event = tight_kernel.step(tight, 8)
    plan = event["prefix_plan"]
    assert event["physical_q"] == event["sampled_q"] == 8
    assert plan["added_rows"] == [] and plan["slack"] == 0
    assert plan["selected_rows"] == list(range(52, 60))
    assert event["absolute_positions"] == list(range(52, 60))

    # Zero mode underfills rather than inventing prefix rows.
    zero_kernel = AllSlotKernel(CPUModel().eval(), prefix_slots={16: 0}, prefix_mode="zero")
    zero = prefix_fixture(zero_kernel)
    event = zero_kernel.step(zero, 16)
    plan = event["prefix_plan"]
    assert event["physical_q"] == event["sampled_q"] == 12
    assert event["underfill"] == 4
    assert plan["added_rows"] == [] and plan["slack"] == 4
    assert plan["selected_rows"] == list(range(52, 64))

    # Random mode is seeded, block-local and leaves the global RNG untouched.
    random_model = CPUModel().eval()
    random_kernel = AllSlotKernel(random_model, prefix_slots={16: 4},
                                  prefix_mode="random")
    first = prefix_fixture(random_kernel)
    rng = torch.get_rng_state().clone()
    event = random_kernel.step(first, 16)
    plan = event["prefix_plan"]
    assert len(plan["added_rows"]) == 4
    assert set(plan["added_rows"]) <= set(plan["eligible_prefix_rows"])
    assert all(44 <= p < 64 for p in plan["added_rows"])
    added = list(plan["added_rows"])
    assert torch.equal(torch.get_rng_state(), rng)
    second_kernel = AllSlotKernel(CPUModel().eval(), prefix_slots={16: 4},
                                  prefix_mode="random")
    second = prefix_fixture(second_kernel)
    event = second_kernel.step(second, 16)
    assert event["prefix_plan"]["added_rows"] == added


@pytest.mark.parametrize("slots,mode", [
    (None, "zero"), ({}, "zero"), ({16: 4}, "right"), ({16: 4}, "random"),
    ({"16": 4}, "right"), ({16: 0}, "zero"), ({32: 0}, "zero"),
    ({64: 0}, "zero"),
])
def test_valid_prefix_configuration(slots, mode):
    kernel = AllSlotKernel(CPUModel().eval(), prefix_slots=slots, prefix_mode=mode)
    assert kernel.prefix_mode == mode


@pytest.mark.parametrize("slots,mode", [
    ({32: 1}, "right"), ({64: 1}, "right"), ({16: 4}, "zero"),
    ({16: 20}, "right"), ({16: -1}, "right"), ({16: 1.0}, "right"),
    ({16: True}, "right"), ({5: 1}, "right"), ({16: 4, "16": 2}, "right"),
    ([(16, 4)], "right"), ({16: 4}, "left"),
])
def test_invalid_prefix_configuration_fails_closed(slots, mode):
    with pytest.raises(ValueError):
        AllSlotKernel(CPUModel().eval(), prefix_slots=slots, prefix_mode=mode)


# ---------------------------------------------------------------------------
# EOS and promotion
# ---------------------------------------------------------------------------


def test_eos_in_head_ends_after_commit_and_never_commits_the_tail():
    model = CPUModel().eval()
    model.fill = {1, 2}
    model.propose = lambda block, token, prior: (4 if block == 1 else 1) if not token else token
    kernel = AllSlotKernel(model)
    state = make(kernel, prompt=32, length=64, eos=(4,))
    assert kernel.step(state, 64)["top_action"] == "pair_bootstrap"
    assert kernel.step(state, 64)["top_action"] == "pair_denoise"
    assert state.phase == "commit"
    with pytest.raises(RuntimeError, match="not final"):
        kernel.output(state)
    event = kernel.step(state, 64)
    assert event["top_action"] == "commit_head" and state.phase == "done"
    assert kernel.output(state).tolist() == [[4]]
    assert state.cache.get_seq_length() == state.committed_end == 64
    assert state.x[0, 64:].tolist() == [1] * 32  # tail never committed
    assert kernel.step(state, 4) is None


def test_eos_in_parked_tail_waits_for_the_new_prefix_reencode():
    model = CPUModel().eval()
    model.fill = {1, 2}
    model.propose = lambda block, token, prior: (4 if block == 2 else 1) if not token else token
    kernel = AllSlotKernel(model)
    state = make(kernel, prompt=32, length=96, eos=(4,))
    assert kernel.step(state, 64)["top_action"] == "pair_bootstrap"
    assert kernel.step(state, 64)["top_action"] == "pair_denoise"
    with pytest.raises(RuntimeError, match="not final"):
        kernel.output(state)
    commit = kernel.step(state, 64)
    assert commit["top_action"] == "commit_head"
    assert state.phase == "reencode" and state.resident is None
    assert state.committed_end == 64
    assert state.x[0, 64:96].tolist() == [4] * 32  # EOS tail not committed
    assert state.x[0, 96:].tolist() == [0] * 32
    with pytest.raises(RuntimeError, match="not final"):
        kernel.output(state)
    verify = kernel.step(state, 8)
    assert verify["top_action"] == "promote_verify"
    assert verify["physical_q"] == verify["sampled_q"] == 32
    assert verify["overflow"] == 24 and verify["dense_reason"] == "promotion_verify"
    assert verify["absolute_positions"] == list(range(64, 96))
    assert state.phase == "commit"
    final = kernel.step(state, 8)
    assert final["top_action"] == "commit_head" and state.phase == "done"
    assert kernel.output(state).tolist() == [[1] * 32 + [4]]
    assert state.cache.get_seq_length() == state.committed_end == 96


# ---------------------------------------------------------------------------
# Fused and split growth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fuse_growth", [True, False])
def test_fused_and_split_growth_share_tokens_and_split_adds_one_forward(fuse_growth):
    model = CPUModel().eval()
    kernel = AllSlotKernel(model, fuse_growth=fuse_growth)
    state = make(kernel, prompt=32, length=64)
    rng = torch.get_rng_state().clone()

    def offer(info):
        model.fill = set() if not info.tick else {1, 2}
        return 32 if not info.tick else 64

    assert kernel.run(state, offer) is state
    torch.testing.assert_close(torch.get_rng_state(), rng)
    events = generation(state)
    assert [e["q_budget"] for e in events] == [32] + [64] * (len(events) - 1)
    if fuse_growth:
        fused = events[1]
        assert fused["top_action"] == "fused_growth"
        assert fused["physical_q"] == fused["sampled_q"] == 64
        assert fused["refresh_only_positions"] == []
        assert fused["dense_reason"] == "bootstrap"
        assert len(events) == 6
        assert sum(e["physical_q"] for e in events) == 256
    else:
        split = events[1]
        assert split["top_action"] == "split_growth"
        assert split["physical_q"] == 33 and split["sampled_q"] == 32
        assert split["refresh_only_positions"] == [54]
        assert split["dense_reason"] == "bootstrap"
        assert split["dense_required_q"] == 32
        assert split["dense_required_positions"] == list(range(64, 96))
        assert len(events) == 7
        assert sum(e["physical_q"] for e in events) == 257
    assert kernel.output(state).tolist() == [[1] * 64]

"""CPU model and request factory from the repository's TopTransition tests."""
from contextlib import ExitStack
import random
from unittest.mock import patch

import pytest
import torch
from test_llada_top_transition import CPUModel, create
from experiments.t2t.serving_slots import llada_all_slots as slots
from experiments.t2t.serving_slots.llada_quote import quote


def freeze(x):
    if torch.is_tensor(x):
        return (id(x), x.shape, x.tolist())
    if hasattr(x, "to_legacy_cache"):
        return (id(x), freeze(x.to_legacy_cache()))
    if isinstance(x, dict):
        return {k: freeze(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return tuple(map(freeze, x))
    if hasattr(x, "__dict__"):
        return (id(x), freeze(vars(x)))
    return x.copy() if isinstance(x, set) else x


def check(k, s, offer):
    before, py_rng, rng = freeze(s), random.getstate(), torch.get_rng_state()
    calls, draws = len(k.model.calls), len(k.model.draws)
    with ExitStack() as stack:
        for obj, name in [(k, m) for m in ("step", "_prefill", "_execute", "_dual_pick",
                                           "_low_denoise", "_prepare_hard_metadata", "_sample")] + [
                (slots, "Random"), (k.model, "forward")]:
            stack.enter_context(patch.object(obj, name, side_effect=AssertionError(name)))
        a = quote(k, s, offer)
        assert quote(k, s, offer) == a
    assert freeze(s) == before and py_rng == random.getstate() and torch.equal(rng, torch.get_rng_state())
    assert calls == len(k.model.calls) and draws == len(k.model.draws)
    e = k.step(s, offer)
    widths = [len(pos) for pos, _ in k.model.calls[calls:]]
    assert widths == ([a["prefill_q"]] if a["prefill_q"] else []) + ([a["physical_q"]] if e else [])
    assert a["exact"]
    if e is None:
        assert (a["physical_q"], a["phase"], a["mandatory_q"]) == (0, "done", 0)
    else:
        assert (a["offer_tick"], a["phase"], a["action"], a["physical_q"]) == (
            e["offer_tick"], e["phase"], e["top_action"], e["physical_q"])
        assert (a["mandatory_q"], a["mandatory_positions"]) == (e["mandatory_q"], e["refreshed_pending"])
        assert a["required_q"] == len(set(e["dense_required_positions"] + e["refreshed_pending"]))
        assert a["dense_required_q"] == e["dense_required_q"]
    return a


@pytest.mark.parametrize("q", slots.OFFERS)
@pytest.mark.parametrize("prompt", [3, 44])
def test_full_request_including_first_last_and_terminal(q, prompt):
    k = slots.AllSlotKernel(CPUModel().eval())
    s = create(k, prompt=prompt, length=62)
    for _ in range(128):
        a = check(k, s, q)
        if a["phase"] == "done":
            return
    pytest.fail("request did not terminate")


@pytest.mark.parametrize("fused,dirty", [(True, True), (False, True), (False, False)])
def test_growth(fused, dirty):
    m = CPUModel().eval()
    m.fill = set()
    if not dirty:
        m.propose_at = dict.fromkeys(range(32, 64), 0)
    k = slots.AllSlotKernel(m, fuse_growth=fused)
    s = create(k)
    check(k, s, 8)
    a = check(k, s, 64)
    assert (a["action"], a["physical_q"]) == (
        ("fused_growth", 64) if fused else ("split_growth", 32 + int(dirty)))


def test_park_then_strict_edit_and_revoke():
    m = CPUModel().eval()
    m.fill, m.propose_at = {2}, dict.fromkeys(range(32, 64), 0)
    k = slots.AllSlotKernel(m)
    s = create(k)
    check(k, s, 64)
    a = check(k, s, 4)
    assert (a["action"], a["physical_q"], a["mandatory_q"]) == ("tail_dirty_drain", 32, 32)
    m.propose_at = {68: 0, 72: 2}
    check(k, s, 64)
    assert s.right_history.fresh == {8}
    assert check(k, s, 32)["mandatory_positions"][-2:] == [68, 72]
    assert check(k, s, 4)["physical_q"] == 4


@pytest.mark.parametrize("mode,n", [("zero", 0), ("right", 2), ("random", 2)])
def test_low_true_slack_and_post_limit(mode, n):
    m = CPUModel().eval()
    m.propose_at = {60: 0, 61: 0}
    k = slots.AllSlotKernel(m, prefix_slots={8: n}, prefix_mode=mode)
    s = create(k, prompt=44, max_post=0)
    check(k, s, 8)
    check(k, s, 4)  # project the wide initial admissions
    m.propose_at.clear()
    assert check(k, s, 8)["physical_q"] == 4+n  # two MASK, two ordinary old rows
    assert s.phase == "denoise"
    assert check(k, s, 4)["action"] == "commit_head"


def test_dirty_tail_verify_commit_promote_and_32_underfill():
    k = slots.AllSlotKernel(CPUModel().eval())
    s = create(k, prompt=44)
    check(k, s, 64)
    assert check(k, s, 4)["physical_q"] == 64  # full verification plus parked tail
    assert check(k, s, 16)["action"] == "commit_head"
    assert check(k, s, 8)["action"] == "promote_verify"
    s = create(k, prompt=44, max_post=1)
    check(k, s, 64)
    k.model.propose_at = {68: 2}
    assert check(k, s, 64)["physical_q"] == 52  # only editable head rows + tail
    assert check(k, s, 4)["physical_q"] == 33  # commit with dirty tail
    s = create(k, prompt=44, length=1)
    assert check(k, s, 64)["physical_q"] == 32
    assert check(k, s, 32)["physical_q"] == 20


def test_done_tail_on_high_offer_and_invalid_offer():
    m = CPUModel().eval()
    m.fill = {2}
    k = slots.AllSlotKernel(m)
    s = create(k, max_post=1)
    for q in (False, 4., 5):
        with pytest.raises(ValueError):
            quote(k, s, q)
    assert not s.events and not m.calls
    check(k, s, 64)
    m.propose_at = {72: 2}
    check(k, s, 64)
    assert s.right_done and check(k, s, 64)["physical_q"] == 33


@pytest.mark.parametrize("chosen", slots.OFFERS)
def test_five_prices_before_one_choice(chosen):
    m = CPUModel().eval()
    # Bootstrap writes one head token and five tail tokens.
    m.propose_at = dict.fromkeys([*range(33, 64), *range(69, 96)], 0)
    k = slots.AllSlotKernel(m)
    s = create(k)
    check(k, s, 64)
    before = freeze(s)
    prices = {q: quote(k, s, q) for q in reversed(slots.OFFERS)}
    assert freeze(s) == before
    assert {q: a["physical_q"] for q, a in prices.items()} == {4: 6, 8: 8, 16: 16, 32: 37, 64: 64}
    assert all(a["mandatory_positions"] == [32, 64, 65, 66, 67, 68] for a in prices.values())
    assert check(k, s, chosen) == prices[chosen]

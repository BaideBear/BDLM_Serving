"""Model-free contracts for the opt-in Q16 paid switchable profile."""
from types import SimpleNamespace
from unittest.mock import patch

import torch

from bdlm_serving.methods.t2t.serving_profiles import open_strict_llada
from experiments.t2t.serving_slots.llada_window_sample import switchable_paid_class
from experiments.t2t.serving_slots.llada_window_switch import TopAgreementWindowRequest
from test_llada_switched_head import planner, record_plan


PAID_STATE = dict(origin=32, end=131, resident=range(32, 44),
                  resolved=range(32, 40), masks=range(40, 64))


def paid_request():
    base = planner(switchable=True, top_gate="quarter", coverage="head",
                   accept=True, **PAID_STATE)
    request = object.__new__(switchable_paid_class())
    request.__dict__.update(base.__dict__)
    return request


def test_switchable_paid16_adds_only_waiting_head_masks_at_q16():
    request = paid_request()
    plan, sampled, phase, cap = request._switch_plan(16)
    assert cap == 16 and phase == "sample"
    assert plan.cold == tuple(range(44, 56))
    assert plan.mask == (40, 41, 42, 43)
    assert request._paid_window_mask == (40, 41, 42, 43)
    assert sampled == tuple(range(40, 56))
    event = request.step(16)
    assert event["paid_waiting_mask"] == (40, 41, 42, 43)
    assert set(event["sampled"]) == set(event["cold"]) | {40, 41, 42, 43}
    assert {40, 41, 42, 43} <= set(event["written_absolute"])
    for q in (4, 8, 64):
        request = paid_request()
        plan, sampled, phase, cap = request._switch_plan(q)
        assert request._paid_window_mask == ()
        assert "paid_waiting_mask" not in request.step(q)


def test_switchable_paid16_keeps_q64_repeat_steps_identical():
    def make(cls):
        base = planner(switchable=True, top_gate="quarter", coverage="head",
                       accept=True, **PAID_STATE)
        request = object.__new__(cls)
        request.__dict__.update(base.__dict__)
        return request

    old = make(TopAgreementWindowRequest)
    new = make(switchable_paid_class())
    for _ in range(4):
        if old.status != "ready" or new.status != "ready":
            break
        before = old.step(64)
        after = new.step(64)
        assert record_plan(before) == record_plan(after)
        assert before["dirty_after"] == after["dirty_after"]
        assert before["written_absolute"] == after["written_absolute"]
        assert torch.equal(old.x, new.x)
    assert old.status == new.status and old.tokens == new.tokens


def test_open_strict_llada_paid16_profile_uses_the_switchable_repeat_engine():
    class FakePaid:
        origin = 3

        def __init__(self, *args, **kwargs):
            self.args, self.kwargs = args, kwargs
            self.status = "ready"

    class FakeRepeat:
        origin = 3

        def __init__(self, *args, **kwargs):
            self.args, self.kwargs = args, kwargs
            self.status = "ready"

    with patch("experiments.t2t.serving_slots.llada_window_sample.switchable_paid_class",
               return_value=FakePaid):
        paid = open_strict_llada(
            SimpleNamespace(training=False), torch.tensor([[1, 2, 3]]), seed=0,
            mask_id=0, eos_ids=[9], max_new_tokens=8, coverage="sparse",
            top_gate="half", profile="quarter_window_repeat64_paid16")
    assert isinstance(paid._request, FakePaid)
    assert paid._request.kwargs["switchable"] is True
    assert paid._request.kwargs["coverage"] == "head"
    assert paid._request.kwargs["top_gate"] == "quarter"
    assert paid._request.kwargs["close_policy"] == "edit_once"
    assert paid._request.kwargs["warm_head_sample"] is True
    assert paid._request.kwargs["borrow_at_half"] is False

    with patch("experiments.t2t.serving_slots.llada_window_switch.TopAgreementWindowRequest",
               FakeRepeat):
        repeat = open_strict_llada(
            SimpleNamespace(training=False), torch.tensor([[1, 2, 3]]), seed=0,
            mask_id=0, eos_ids=[9], max_new_tokens=8, coverage="sparse",
            top_gate="half", profile="quarter_window_repeat64")
    assert isinstance(repeat._request, FakeRepeat)

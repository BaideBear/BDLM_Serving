from contextlib import contextmanager
from threading import current_thread
from types import SimpleNamespace as NS

import pytest
import torch

from experiments.t2t.serving_slots import gemma_native_argmax_request as adapter
from test_gemma_strict_request import inputs, model


B = adapter.B


def fixture(monkeypatch, *, block, dirty, age=1, hit=True, paid=False, cap=2*B):
    req = adapter.NativeArgmaxRequest.__new__(adapter.NativeArgmaxRequest)
    req._owner, req.origin, req.ordinal, req.tick = current_thread(), 3, 0, 0
    req.tail, req.rebuild, req.cursor = None, False, 0
    req.status, req.paid_verify, req.max_new_tokens = "ready", paid, cap
    req.eos, req.tokens, req.ids = {99}, [], torch.tensor([[0, 1, 2]])
    req.mask = torch.ones_like(req.ids, dtype=torch.bool)
    req.rows = NS(initialized=adapter.FULL, dirty=set(dirty), _native_stop=False,
                  age=age-1, max_steps=48)
    req.sampler = NS(age=age-1, steps=48, phase="sample")
    req.bank, req.lane = adapter.WaveKV(B), None
    req.bank.live = list(range(3, B+3))
    req.model = NS(model=NS(decoder=NS(layers=[None, None])))
    req.capture = NS(draft=None)
    calls = []

    @contextmanager
    def visible(r, enabled, exposed):
        assert r is req and enabled
        yield
        exposed.extend((0, 1))

    def apply(heads, pos, sampled, prefix, cold, offer, route):
        head, = heads
        assert head.ordinal == 0 and not prefix and not cold and offer == B
        assert sampled == set(pos) and route == "dense_single"
        assert list(pos) == list(range(3, B+3))
        calls.append(("D", len(pos)))
        head.sampler.age += 1
        head.rows.age += 1
        head.rows._native_stop = hit
        head.rows.dirty = {2, 3}
        req.capture.draft = block if hit else None
        return dict(parts=[dict(rows=list(range(B)), sampled=list(range(B)),
                    dirty_before=sorted(dirty), dirty_after=[2, 3])],
                    fullstop=dict(hit=hit), encoded=False, encoder_F=0, encoder_Q=0)

    def encode(n, *, first):
        assert n == B and not first
        calls.append(("E", req.ids[0, -B:].tolist()))

    monkeypatch.setattr(adapter, "representation", visible)
    req._apply, req._encode = apply, encode
    req._start_block = lambda: calls.append(("start", req.ordinal))
    return req, calls


def test_native_eos_keeps_hit_argmax_only(monkeypatch):
    block = (13, 99) + (67,) * (B-2)
    req, calls = fixture(monkeypatch, block=block, dirty={1, 5})
    event = req.step(B)
    assert calls == [("D", B)]
    assert event["dirty_before_paid"] == [1, 5] and event["unread_speculative_rows"] == [2, 3]
    assert event["native_draft"] == block
    assert req.result() == event["delivered"] == (13, 99)
    assert (event["generation_F"], event["physical_generation_Q"]) == (1, B)


def test_nonterminal_encodes_exact_delivery(monkeypatch):
    block = tuple(i % 21 for i in range(B))
    req, calls = fixture(monkeypatch, block=block, dirty={7})
    event = req.step(B)
    assert calls == [("D", B), ("E", list(block)), ("start", 1)]
    assert tuple(req.tokens) == event["delivered"] == block
    assert (event["encoder_F"], event["encoder_Q"], event["generation_F"]) == (1, B, 2)
    with pytest.raises(RuntimeError, match="ready"):
        req.result()


def test_cap_does_not_encode_and_sparse_offer_is_rejected(monkeypatch):
    req, calls = fixture(monkeypatch, block=(7,)*B, dirty=set(), cap=B)
    with pytest.raises(ValueError, match="Q256"):
        req.step(128)
    event = req.step(B)
    assert calls == [("D", B)] and not event["encoded"]
    assert req.result() == (7,)*B


def test_clock_limit_is_not_a_delivery(monkeypatch):
    req, calls = fixture(monkeypatch, block=(7,)*B, dirty=set(), age=48, hit=False)
    with pytest.raises(RuntimeError, match="Clock limit"):
        req.step(B)
    assert calls == [("D", B)] and req.status == "failed" and not req.tokens


def test_paid_control_does_not_publish_at_hit(monkeypatch):
    req, calls = fixture(monkeypatch, block=(7,)*B, dirty=set(), paid=True)
    event = req.step(B)
    assert event["native_draft"] == (7,)*B and event["delivered"] is None
    assert not req.tokens and req.rows.dirty == {2, 3} and calls == [("D", B)]


def test_capture_returns_actual_native_criterion_result():
    seen = []

    class Criterion:
        def __call__(self, draft, scores):
            seen.append((draft, scores))
            return torch.tensor([True])

        def reset(self):
            seen.append("reset")

    capture = adapter._CaptureStop(Criterion())
    draft, scores = torch.arange(B)[None], torch.ones((1, B, 2))
    assert capture(draft, scores).item() and capture.draft == tuple(range(B))
    assert seen[0][0] is draft and seen[0][1] is scores
    capture.reset()
    assert capture.draft is None and seen[-1] == "reset"


@pytest.mark.parametrize("paid", [False, True])
def test_cpu_real_projection_and_two_block_handoff(model, monkeypatch, paid):
    class StopOnSecondTurn:
        def __call__(self, draft, scores):
            self.age += 1
            return torch.tensor([self.age == 2])

        def reset(self):
            self.age = 0

    from bdlm_serving.runtime.diffusiongemma import observe
    monkeypatch.setattr(model.generation_config, "stability_threshold", 1)
    monkeypatch.setattr(model.generation_config, "confidence_threshold", .005)
    monkeypatch.setattr(model, "_prepare_diffusion_stopping_criteria", lambda cfg: StopOnSecondTurn())
    data = inputs()
    enc = []
    with observe(model) as physical, model.model.encoder.register_forward_pre_hook(
            lambda m, a, kw: enc.append(tuple(kw["input_ids"][0].tolist())), with_kwargs=True):
        req = adapter.NativeArgmaxRequest(model, data, seed=17000, max_new_tokens=2*B, paid_verify=paid)
        es = []
        while req.status == "ready":
            es.append(req.step(B))
    routes = ["dense_single", "dense_single"] + (["verify_paid"] if paid else [])
    assert [e["route"] for e in es] == routes * 2
    assert physical == [("encoder", 5)] + [("decoder", B)] * len(routes) + [
        ("encoder", B)] + [("decoder", B)] * len(routes)
    assert enc[1] == es[len(routes)-1]["delivered"] == req.result()[:B]
    assert req.result()[B:] == es[-1]["delivered"] and len(req.result()) == 2*B

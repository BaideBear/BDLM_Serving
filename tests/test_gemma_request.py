"""CPU contracts for serial, cooperative use of one Gemma model (not concurrency)."""
from contextlib import contextmanager

import pytest
import torch

from experiments.t2t.serving_slots import gemma_unified as u, gemma_wave_kv as kv
from test_gemma_unified import B, TIERS, VOCAB, FakeModel, _Sampler, plan


A = (32, 512, 64, 128, 256)
BUDGET_B = (256, 32, 512, 64)
NOT_LIVE = (ValueError, RuntimeError, StopIteration)


class NoiseSampler(_Sampler):
    def initialize_canvas(self, batch, device):
        return torch.randint(1, VOCAB - 1, (batch, B), device=device)


class Model(FakeModel):
    """Reuse the physical tape, but never derive behavior from the model-global sampler ordinal."""
    def __init__(self, registry, *, eos=False):
        super().__init__(steps=4)
        self.registry = registry
        self.force_eos = eos
        self.fail_sample = False
        self.attns = [layer.self_attn for layer in self.model.decoder.layers]
        self.configs = [a.config for a in self.attns]
        for i, a in enumerate(self.attns):
            a.layer_idx = i

    def _prepare_sampler(self, cfg):
        return NoiseSampler(self)

    def __call__(self, **kw):
        result = super().__call__(**kw)
        q = kw["decoder_input_ids"].float().view(1, 1, -1, 1)
        k = torch.nn.functional.pad(q, (0, 0, kw["past_key_values"].length, 0))
        for a in self.attns:
            # Actually dispatch through the model's currently installed config/bank.
            shift = self.registry[a.config._attn_implementation](a, q, k, k, None)
        result.logits = result.logits.roll(shift % VOCAB, -1)
        return result

    def _denoising_step(self, *, decoder_forward, current_canvas, sampler, **kw):
        sampler.calls += 1
        raw = decoder_forward().logits
        draw = torch.randint_like(current_canvas, high=VOCAB - 2)
        if self.fail_sample:
            raise ArithmeticError("injected after RNG draw")
        ids = (current_canvas + draw + raw.argmax(-1)) % (VOCAB - 2) + 1
        sampler.accepted_token_mask[:, :4] = True
        if self.force_eos:
            ids[:, 3] = VOCAB - 1
        return ids, ids.clone(), raw, torch.tensor([False])

    def work(self):
        return (len(self.encoder_calls), len(self.decoder_calls), len(self.samplers),
                sum(s.calls for s in self.samplers))

    @contextmanager
    def boundary(self, *, idle=False):
        rng = torch.get_rng_state()
        mode = torch.is_grad_enabled(), torch.is_inference_mode_enabled()
        before = self.work()
        try:
            yield
        finally:
            assert torch.equal(rng, torch.get_rng_state()), "caller RNG changed"
            assert mode == (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
            assert self.registry.keys() == {"sdpa"}, "attention alias outlived the call"
            assert all(a.config is c and c._attn_implementation == "sdpa"
                       for a, c in zip(self.attns, self.configs))
            assert not self.model.decoder.self_conditioning._forward_pre_hooks
            if idle:
                assert self.work() == before, "unexpected encoder, decoder, spawn or sample"


@pytest.fixture
def model_factory(monkeypatch):
    # Do not use test_gemma_unified.engine: it replaces wave_attention by a no-op.
    # A private registry exercises real WaveKV, config swaps and alias cleanup without HF.
    def sdpa(module, q, k, v, mask):
        assert mask is not None
        return int(mask[0, 0, 0].sum()) + int(v[0, 0, -1, 0])

    registry = {"sdpa": sdpa}
    monkeypatch.setattr(kv, "_attention_functions", lambda: registry)
    monkeypatch.setattr(u, "WaveKV", kv.WaveKV)
    monkeypatch.setattr(u, "wave_attention", kv.wave_attention)
    with torch.random.fork_rng(devices=[]):
        yield lambda **kw: Model(registry, **kw)


def inputs(length):
    ids = torch.arange(length)[None] + 1
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def reference(model, data, seed, offers, causal=True):
    with model.boundary():
        return u.generate(model, data, plan(2 * B), seed=seed, causal=causal,
                          next_budget=lambda t: offers[t.tick % len(offers)])


def start(model, data, seed, causal=True, quoted=True):
    with model.boundary():
        h = u.start_request(model, data, plan(2 * B), seed=seed, causal=causal, quote_costs=quoted)
    assert not h.done
    return h


def step(h, model, expected, i, quoted=True):
    with model.boundary(idle=True):
        tick = h.tick
        assert getattr(tick, "tick", tick) == i and not h.done
        if quoted:
            options = h.options
            assert set(options) == set(TIERS)
            assert options[expected[i]["offer"]].decoder_q == expected[i]["q"]
    before = len(model.decoder_calls)
    with model.boundary():
        event = h.advance(expected[i]["offer"])
    assert len(model.decoder_calls) == before + 1
    assert event == expected[i], f"trace diverged at tick {i}"
    assert h.done == (i + 1 == len(expected))


def finish(h, model, ref, offset=0, quoted=True):
    ids, trace = ref
    for i in range(offset, len(trace)):
        step(h, model, trace, i, quoted)
    with model.boundary(idle=True):
        assert h.done and h.result() == (ids, trace)
        with pytest.raises(NOT_LIVE):
            h.advance(32)


def test_reference_really_uses_seed(model_factory):
    data = inputs(3)
    ref = reference(model_factory(), data, 7, A)
    assert ref == reference(model_factory(), data, 7, A)
    assert ref[0] != reference(model_factory(), data, 8, A)[0]
    assert {e["offer"] for e in ref[1]} == set(TIERS)
    assert any(e["promoted"] for e in ref[1])


@pytest.mark.parametrize("eos,quoted", [(False, True), (True, False)])
def test_generate_parity_and_natural_endpoint(model_factory, eos, quoted):
    data = inputs(3)
    ref = reference(model_factory(eos=eos), data, 19, A)
    model = model_factory(eos=eos)
    h = start(model, data, 19, quoted=quoted)
    with model.boundary(idle=True), pytest.raises(NOT_LIVE):
        h.result()
    finish(h, model, ref, quoted=quoted)
    assert len(ref[0]) == (4 if eos else 2 * B)
    assert [n for n, _ in model.encoder_calls] == ([3] if eos else [3, B])


def test_interleaved_shared_model_matches_both_solo_runs(model_factory):
    a, b = inputs(3), inputs(5)
    ra = reference(model_factory(), a, 7, A)
    rb = reference(model_factory(), b, 31, BUDGET_B, causal=False)
    model = model_factory()
    ha = start(model, a, 7)
    step(ha, model, ra[1], 0)
    hb = start(model, b, 31, causal=False)
    for i in range(max(len(ra[1]), len(rb[1]))):
        torch.rand(7)  # Caller also owns an independent CPU stream between advances.
        if i < len(rb[1]):
            step(hb, model, rb[1], i)
        if 0 < i < len(ra[1]):
            step(ha, model, ra[1], i)
    with model.boundary(idle=True):
        assert ha.result() == ra and hb.result() == rb


def test_invalid_offers_are_retryable_without_work(model_factory):
    data = inputs(3)
    ref = reference(model_factory(), data, 11, A)
    model = model_factory()
    h = start(model, data, 11)
    for i in range(3):  # Initial, eligible to spawn, and two resident canvases.
        with model.boundary(idle=True):
            before = (h.tick, dict(h.options))
        for bad in (31, True, 32.0, "512"):
            with model.boundary(idle=True), pytest.raises((TypeError, ValueError)):
                h.advance(bad)
            with model.boundary(idle=True):
                assert not h.done and (h.tick, h.options) == before
        step(h, model, ref[1], i)
    finish(h, model, ref, offset=3)


def test_close_older_request_leaves_younger_one_intact(model_factory):
    a, b = inputs(3), inputs(5)
    ra = reference(model_factory(), a, 5, A)
    rb = reference(model_factory(), b, 29, BUDGET_B, causal=False)
    model = model_factory()
    old = start(model, a, 5)
    step(old, model, ra[1], 0)
    young = start(model, b, 29, causal=False)
    step(young, model, rb[1], 0)
    step(old, model, ra[1], 1)  # A live tail is discarded, not committed.
    with model.boundary(idle=True):
        old.close()
        with pytest.raises(NOT_LIVE):
            old.result()
        with pytest.raises(NOT_LIVE):
            old.advance(512)
    finish(young, model, rb, offset=1)


def test_sampling_exception_restores_caller_and_other_request(model_factory):
    data = inputs(3)
    ref = reference(model_factory(), data, 37, A)
    model = model_factory()
    survivor = start(model, data, 37)
    step(survivor, model, ref[1], 0)
    broken = start(model, inputs(5), 3, causal=False)
    model.fail_sample = True
    with model.boundary(), pytest.raises(ArithmeticError, match="after RNG draw"):
        broken.advance(32)
    model.fail_sample = False
    finish(survivor, model, ref, offset=1)

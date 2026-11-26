"""CPU quote checks against the production callback and real executor control flow."""
from types import SimpleNamespace as NS

import pytest
import torch

from experiments.t2t.serving_slots.gemma_quote import BLOCK as B, OFFERS, quote_options


@pytest.fixture
def engine():
    from experiments.t2t.serving_slots import gemma_unified
    return gemma_unified


def canvas(ordinal=0, *, dirty=(), unresolved=range(B), **overrides):
    mask = torch.ones((1, B), dtype=torch.bool)
    mask[0, list(unresolved)] = False
    fields = dict(ordinal=ordinal, dirty=set(dirty), sampler=NS(accepted_token_mask=mask),
                  promoted=False, verify=False, tentative=False, age=1, raw=object())
    return NS(**dict(fields, **overrides))


def menu(engine, *live, cap=3*B, steps=6, may_open=False, quotas=None, choose=None):
    return quote_options(live, cap=cap, steps=steps, may_open=may_open,
                         prefix_slots=quotas or {}, choose=choose or engine._choose)


@pytest.mark.parametrize("fields,phase,reason", [
    (dict(age=0, raw=None), "bootstrap", None),
    (dict(promoted=True, age=5, verify=True), "verify", "promotion"),
    (dict(age=5, verify=True), "verify", "terminal_clock"),
    (dict(verify=True), "verify", "sparse_stop"),
    (dict(unresolved=()), "verify", "empty_selection"),
])
def test_full_head_reasons_and_last_canvas(engine, fields, phase, reason):
    for cap, bound in [(B, 0), (2*B, B)]:
        q = menu(engine, canvas(**fields), cap=cap)[32]
        assert (q.phase, q.verify_reason, q.decoder_q, q.decoder_f) == (phase, reason, B, 1)
        assert (q.post_reencode_q_max, q.post_reencode_f_max) == (bound, int(bool(bound)))


def test_underfill_dirty_overlap_and_prefix_cardinality(engine, monkeypatch):
    # Row 250 is dirty AND unresolved; only rows 0 and 1 are clean prefix seats.
    head = canvas(dirty=[5, 200, 250], unresolved=[2, 250, 255])
    tail = canvas(1, dirty=range(7))
    quotas = {32: 3, 64: 19, 128: 100}
    calls = []
    def counted(*args, **kw):
        calls.append((args[1:4], kw))
        return engine._choose(*args, **kw)
    def no_rng(*a, **kw):
        pytest.fail("quote tried to draw prefix rows")

    with monkeypatch.context() as patch:
        patch.setattr(engine, "Random", no_rng)
        opts = menu(engine, head, tail, quotas=quotas, choose=counted)
        assert opts == menu(engine, head, tail, quotas=quotas)
    assert len(calls) == 1
    assert tuple(opts) == OFFERS
    for offer in OFFERS[:3]:
        for mode in ("right", "random"):
            rows, optional, _ = engine._choose(head, offer-7, quotas[offer], mode, seed=8, tick=3)
            q = opts[offer]
            assert (q.head_q, q.prefix_q, q.parked_dirty_q) == (len(rows), len(optional), 7)
            assert (q.decoder_q, q.mandatory_q, q.post_reencode_q_max) == (14, 10, 0)


def test_dirty_prefix_is_not_an_optional_seat(engine):
    h = canvas(dirty=[0, 1, 3, 8], unresolved=[8, 200])
    q = menu(engine, h, quotas={32: 30})[32]
    assert (q.mandatory_q, q.prefix_q, q.decoder_q) == (4, 5, 10)


def test_verifying_head_cannot_park_to_absorb_tail_debt(engine):
    q = menu(engine, canvas(verify=True), canvas(1, dirty=range(40)))[32]
    assert (q.verify_reason, q.parked_dirty_q, q.decoder_q) == ("sparse_stop", 40, 296)
    assert (q.decoder_f, q.post_reencode_f_max, q.post_reencode_q_max) == (1, 1, B)


@pytest.mark.parametrize("hd,td,width,phase", [
    (0, 32, 0, "tail_dirty_drain"), (0, 40, 0, "tail_dirty_drain"),
    (5, 40, 5, "sparse"), (40, 1, 40, "sparse"), (B, 3, B, "sparse"),
])
def test_debt_is_neither_truncated_nor_double_counted(engine, hd, td, width, phase):
    opts = menu(engine, canvas(dirty=range(hd)), canvas(1, dirty=range(td)))
    q = opts[32]
    assert (q.head_q, q.tail_sample_q, q.parked_dirty_q, q.phase) == (width, 0, td, phase)
    assert (q.decoder_q, q.post_reencode_q_max) == (width + td, B if width == B else 0)
    assert opts[256].decoder_q == B + td
    assert (opts[512].decoder_q, opts[512].parked_dirty_q) == (2*B, 0)


def test_open_and_tentative_or_last_step_tail(engine):
    h = canvas()
    assert menu(engine, h)[512].decoder_q == B
    opts = menu(engine, h, may_open=True)
    assert [q.offer for q in opts.values() if q.opens_tail] == [512]
    assert opts[512].decoder_q == 2*B
    for fields in (dict(tentative=True), dict(age=5)):
        q = menu(engine, h, canvas(1, dirty=range(3), **fields))[512]
        assert (q.opens_tail, q.tail_sample_q, q.parked_dirty_q, q.decoder_q) == (False, 0, 3, 259)
    with pytest.raises(ValueError):
        menu(engine, h, cap=B, may_open=True)


@pytest.mark.parametrize("eos", [0, 15])
def test_live_menu_vs_default_executor(engine, eos, monkeypatch):
    from test_gemma_unified import DATA, FakeModel, _FakeWaveKV, _fake_wave_attention, plan
    monkeypatch.setattr(engine, "WaveKV", _FakeWaveKV)
    monkeypatch.setattr(engine, "wave_attention", _fake_wave_attention)
    offers = (32, 512, 32, 64, 128, 256, 512)
    chosen = []
    def run(quoted):
        model = FakeModel(steps=6, accept_first=20, accept_ordinals={1})
        model.cfg.eos_token_id = eos
        def supply(state, options=None):
            offer = offers[state.tick % len(offers)]
            assert (options is not None) == quoted
            if quoted:
                assert state.tick == len(model.decoder_calls) == len(chosen)
                assert tuple(options) == OFFERS
                chosen.append(options[offer])
            return offer
        result = engine.generate(model, DATA, dict(plan(), prefix_slots={128: 32}, prefix_mode="random"),
                                 next_budget=supply, quote_costs=quoted)
        return result, model

    from experiments.t2t.serving_slots import gemma_quote
    with monkeypatch.context() as patch:
        patch.setattr(gemma_quote, "quote_options", lambda *a, **kw: pytest.fail("unrequested quote"))
        reference, _ = run(False)
    (ids, trace), model = run(True)
    assert (ids, trace) == reference
    assert set(e["offer"] for e in trace) == set(OFFERS)
    assert len(chosen) == len(trace) == len(model.decoder_calls)
    for q, e, physical in zip(chosen, trace, model.decoder_calls):
        assert q.decoder_q == e["q"] == physical.q
        assert (q.phase, q.verify_reason, q.mandatory_q) == (e["phase"], e["verify_reason"], e["mandatory_q"])
        assert q.prefix_q == len(e["prefix_selected"]) and q.parked_dirty_q == len(e["refresh_only"])
        assert len(e["encoded"]) * B <= q.post_reencode_q_max
    reencodes = sum(len(e["encoded"]) for e in trace)
    assert [n for n, _ in model.encoder_calls] == [DATA["input_ids"].shape[1]] + [B]*reencodes
    assert reencodes == (0 if eos == 0 else 2)

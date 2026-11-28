from types import SimpleNamespace as NS
import pytest
from experiments.t2t.serving_slots.gemma_response_executor import GemmaResponseRequest


def make_request(**changes):
    r = object.__new__(GemmaResponseRequest)
    r.status = 'ready'
    r.tail = None
    r.tick = 7
    r.admission_offer = 64
    r.response_offer = 64
    r.response_admission_tick = None
    r.response_schedule = []
    for k, v in changes.items(): setattr(r, k, v)
    return r


def test_schedule_keeps_small_admission_then_exact_2b():
    r = make_request()
    assert r.response_offer_for() == 64
    r.tail = NS(ordinal=1)
    assert r.response_offer_for() == 512


def test_retired_or_open_response_does_not_return_to_small_offer():
    r = make_request(response_admission_tick=3, response_admission_ordinal=1, tail=None)
    assert r.response_offer_for() == 512


@pytest.mark.parametrize('offer', [32, 128, 256])
def test_registered_small_admission_offers_only(offer):
    assert make_request(admission_offer=offer).response_offer_for() == offer


def test_rejects_non_registered_small_offer_and_explicit_override():
    with pytest.raises(ValueError):
        make_request(admission_offer=16)
    r = make_request()
    with pytest.raises(ValueError):
        r.step(512)


def test_step_records_schedule_and_rejects_cap_violation(monkeypatch):
    r = make_request(tail=None)
    calls = []
    def step(offer):
        calls.append(offer)
        assert offer == 64
        return dict(decoder_Q=64)
    monkeypatch.setattr('experiments.t2t.serving_slots.gemma_fixed_dual.AblationRequest.step', step)
    e = r.step()
    assert calls == [64] and e['response_schedule_offer'] == 64
    assert e['response_block_opened'] is False and r.response_schedule == [64]
    monkeypatch.setattr('experiments.t2t.serving_slots.gemma_fixed_dual.AblationRequest.step',
                        lambda self, offer: dict(decoder_Q=65))
    with pytest.raises(RuntimeError, match='step cap'):
        r.step()

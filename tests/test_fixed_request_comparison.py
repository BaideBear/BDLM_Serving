import pytest

from experiments.t2t.serving_slots.fixed_request_comparison import collect,render


def test_same_fixed_offer_does_not_pool_hardware_or_contaminate_efficiency():
    rows=collect()
    at={(r["model"],r["cohort"],r["budget"],r["method"]):r for r in rows}
    capped=at["gemma","frozen90",128,"IBS capped"]
    assert capped["token_efficiency"]==pytest.approx(30814/209763)
    assert capped["F"]==2055 and capped["peak_decoder_Q"]=="128"
    assert at["gemma","frozen90",128,"IBS nominal"]["F"]==2042
    assert at["llada21","frozen90",32,"Native"]["F"]!=at["llada21","frozen90",32,"Flow W1"]["F"]
    assert at["llada21","frozen90",64,"Flow W2"]["token_efficiency"] is None
    assert ("llada21","frozen90",64,"IBS capped") not in at
    assert "separately verified" in render(rows)

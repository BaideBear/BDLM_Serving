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


def test_active_window_is_its_own_sealed_arm_never_a_cherry_picked_composite():
    rows=collect()
    at={(r["model"],r["cohort"],r["budget"],r["method"]):r for r in rows}
    for q in (4,8,16):
        arm=[at["llada21",c,q,"Active window"] for c in ("original30","new60","frozen90")]
        for k in ("N","F","history_Q","correct","n"):
            assert arm[0][k]+arm[1][k]==arm[2][k]
        assert arm[2]["token_efficiency"]==pytest.approx(arm[2]["N"]/arm[2]["history_Q"])
    assert at["llada21","frozen90",4,"Active window"]["correct"]==72
    assert at["llada21","frozen90",8,"Active window"]["correct"]==72
    assert at["llada21","new60",16,"Active window"]["correct"]==49
    assert not any(r["method"]=="Active window" and (r["model"]!="llada21" or r["budget"]>16) for r in rows)
    repeat=at["llada21","frozen90",64,"Paid repeat"]
    assert repeat["correct"]==75 and repeat["token_efficiency"]==pytest.approx(43387/399051)
    assert len([r for r in rows if r["method"]=="Paid repeat"])==3
    paid=[r for r in rows if r["method"]=="Paid MASK"]
    assert {(r["budget"],r["model"]) for r in paid}=={(8,"llada21"),(16,"llada21")} and len(paid)==6
    assert at["llada21","frozen90",8,"Paid MASK"]["correct"]==70
    q16=at["llada21","frozen90",16,"Paid MASK"]
    assert q16["correct"]==75 and q16["macro_TPF"]==pytest.approx(3.35629259)
    assert q16["token_efficiency"]==pytest.approx(44943/175591)

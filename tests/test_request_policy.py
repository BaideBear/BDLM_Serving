import pytest

from bdlm_serving.methods.t2t.request_policy import SLOTS, bind_request
from bdlm_serving.methods.t2t.slot_trace import BudgetOffer, RoundInfo
from experiments.t2t.serving_slots.external_report import cost_gaps


@pytest.mark.parametrize("model,width,q,k,selector",[
    ("gemma",256,128,32,"random"),("llada21",32,16,1,"right")])
def test_profile_maps_only_registered_slots(model,width,q,k,selector):
    for pref in (False,True):
        for native in (False,True):
            slots=list(SLOTS[model])+[None]
            source=bind_request(model,q,lambda i:slots[i.tick],prefix=pref,native_full=native)
            for t,offered in enumerate(slots):
                v=source(RoundInfo(t,0,width,1,width-1))
                expected=q if offered is None else offered
                assert v== (BudgetOffer(q,k,selector) if pref and expected==q else
                            BudgetOffer(expected,native_full=native and expected==width))
                v.plan(width)


def test_invalid_offer_is_not_clamped_or_treated_as_a_missing_sample():
    info=RoundInfo(0,0,32,29,2)
    for bad in (True,8.0,0,7,64,"16"):
        with pytest.raises(ValueError):
            bind_request("llada21",8,lambda i:bad)(info)
    with pytest.raises(ValueError,match="different model"):
        bind_request("gemma",32,lambda i:16)(info)
    def fail(_):
        raise TimeoutError("scheduler")
    with pytest.raises(TimeoutError):
        bind_request("llada21",8,fail)(info)


def test_cost_spread_not_cancelled_by_pooled_budget():
    result=cost_gaps([100,100,100,100],[70,95,105,130])
    assert result["total_q_gap"]==0 and result["requests_within_5pct_Q"]==2
    assert result["request_q_gap_p10"]==-.3 and result["request_q_gap_p90"]==.3
    with pytest.raises(ValueError):
        cost_gaps([0],[1])

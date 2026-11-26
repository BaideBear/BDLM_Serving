import pytest

from bdlm_serving.methods.t2t.slot_trace import BudgetOffer, RoundInfo, request_source, resolve


@pytest.mark.parametrize("bad", [False, 0, 33, BudgetOffer(8, 8), BudgetOffer(8, -1),
                                 BudgetOffer(8, 2, "unknown"), "8"])
def test_invalid_budget_offers(bad):
    with pytest.raises(ValueError):
        resolve(lambda _: bad, RoundInfo(0, 32, 32, 5, 3), 8)


def test_one_owner_and_missing_offer_fallback():
    with pytest.raises(ValueError):
        request_source((8,), lambda _: 8, "hard_prefix_zero")
    with pytest.raises(ValueError):
        request_source((), lambda _: 8, "targeted")
    with pytest.raises(ValueError):
        request_source((), 8, "hard_prefix_zero")
    assert resolve(lambda _: None, RoundInfo(1, 0, 32, 3, 10), 4) == (4, 0, "hard_prefix_zero")


@pytest.mark.parametrize("field", ["0","4r0","4r4","4v1","3o-1","064","８","8u02","8.0"])
def test_invalid_encoded_offers(field):
    from bdlm_serving.methods.t2t.slot_trace import parse_offer
    with pytest.raises(ValueError):
        parse_offer(field,256)


def test_encoded_flat_and_mixed_share_source():
    from bdlm_serving.methods.t2t.slot_trace import slots_for
    slots = slots_for("slottrace_32r8_64_128u32_32o4",256,32,0,24)
    source = request_source(slots,None,"hard_prefix_slack_right")
    observed = [resolve(source,RoundInfo(i,0,256,20,3),32) for i in range(5)]
    assert [x[:2] for x in observed] == [(32,8),(64,0),(128,32),(32,4),(32,8)]
    assert [x[2] for x in observed[1:4]] == ["hard_prefix_zero","hard_prefix_slack_random","hard_prefix_slack_oldest"]


def test_native_is_explicit_and_full_width_only():
    from bdlm_serving.methods.t2t.slot_trace import parse_offer
    assert parse_offer("256n",256).plan(256) == (256,0,"native_full")
    assert parse_offer("256",256).plan(256) == (256,0,"hard_prefix_zero")
    for bad in ("128n","256nr2","256nn","0256n"):
        with pytest.raises(ValueError):
            parse_offer(bad,256)
    with pytest.raises(ValueError):
        BudgetOffer(256,1,native_full=True).plan(256)

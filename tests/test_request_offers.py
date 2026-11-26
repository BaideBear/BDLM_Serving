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

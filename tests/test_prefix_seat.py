import pytest

from bdlm_serving.methods.t2t.prefix_seat import choose_prefix_seat


def plan(rows=(4, 5, 6, 7), **changes):
    state = dict(unresolved={4, 5, 6, 7}, resident={1, 3, 4, 5, 7}, due=set(),
                 paid_prefix_end=4, offer=4)
    return choose_prefix_seat(rows, **(state | changes))


def test_warm_only_one_out_one_in_no_right_edge_or_frontier_change():
    assert plan() == (5, 3)
    assert plan(resident={1, 3, 4, 7}) is None
    assert plan(used=True) is None


def test_debt_and_paid_prefix_certificates_are_protected():
    assert plan(due={5}) is None
    assert plan(due={0}) is None
    assert plan(unresolved={2, 4, 5, 6, 7}) is None
    assert plan(paid_prefix_end=0) is None


def test_open_seats_or_single_selected_mask_are_not_exchange_opportunities():
    assert plan((4, 5, 7)) is None
    assert plan(unresolved={5, 9}) is None
    assert plan(resident={4, 5, 6, 7}) is None


@pytest.mark.parametrize("rows", [(4, 5, 5, 7), (4, 7, 5, 6), (4, 5, 6, 7, 8), (-1, 4, 5, 6)])
def test_invalid_plans_do_not_mask_accounting_errors(rows):
    with pytest.raises(ValueError):
        plan(rows)

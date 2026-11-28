import pytest

from tools.audit_measured_hole import counts


def record(tick, native, eligible):
    return dict(block=0, tick=tick, positions=[2, 4], proposals=[55, 66],
                probabilities=[.4, .8], native=native, eligible=eligible)


def test_full_tape_checks_actual_votes_even_if_witness_blocks_an_extra():
    a = record(1, [False, False], [])
    b = record(2, [False, False], [4])
    assert counts([a, b], set(), full_tape=True)["blocked_rows"] == 1
    with pytest.raises(ValueError):
        counts([b, a], set(), full_tape=True)


def test_sparse_positive_events_do_not_impute_predecessor():
    a = record(35, [True, False], [4])
    assert counts([a], set(), full_tape=False)["kept_rows"] == 1
    with pytest.raises(ValueError):
        counts([a], set(), full_tape=True)

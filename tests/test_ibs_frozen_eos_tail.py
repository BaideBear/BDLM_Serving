import pytest

from tools.ibs_frozen_eos_tail import after_certificate


def test_certificate_counts_only_following_paid_calls():
    # Block 1; EOS at 5, position 6 is debt strictly to its right.
    tr = [(1, 1, [4, 5], [5]), (1, 2, [5, 6], [6]),
          (1, 3, [6], []), (1, 4, [7], [])]
    c = after_certificate(tr, eos_pos=5, b=4, q=2)
    assert (c["following_F"], c["following_Q"], c["first_debt_Q"]) == (2, 2, 1)
    assert c["following_clean_F"] == c["with_room_after_one_payment"] == 1


def test_zero_gain_if_stock_already_stops_on_certificate():
    c = after_certificate([(0, 4, [0, 1, 2, 3], [])], 2, 4, 4)
    assert c["certificates"] == 1 and not c["following_F"]


@pytest.mark.parametrize("row", [(0, 3, [1, 2], []), (0, 3, [2], [1])])
def test_reject_reread_or_rewrite_inside_retired_prefix(row):
    with pytest.raises(ValueError, match="retired"):
        after_certificate([(0, 2, [0, 1], []), row], 1, 4, 2)

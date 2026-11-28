import pytest

from bdlm_serving.methods.t2t.frontier_dose import progress_quota


@pytest.mark.parametrize("previous,current,dose,expected", [
    (None, 8, 2, 2), (0, 0, 16, 16), (0, 1, 16, 0),
    (13, 13, 4, 4), (13, 12, 4, 4), (3, 4, 0, 0), (None, 0, 0, 0),
])
def test_gate(previous, current, dose, expected):
    assert progress_quota(previous, current, dose) == expected


@pytest.mark.parametrize("args", [(-1, 0, 4), (1, -1, 2), (None, 0, -1), (True, 1, 2)])
def test_reject_invalid(args):
    with pytest.raises(ValueError):
        progress_quota(*args)

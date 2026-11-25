import pytest

from bdlm_serving.methods.t2t.frontier import select_hard
from bdlm_serving.methods.t2t.protocol import Arm
from bdlm_serving.methods.t2t.shadow import screened_swap


def test_slack_and_trigger():
    resolved = [True, True, False, False, False, True, True, True]
    args = (range(8), resolved, {1}, [-1, 3, -1, -1, -1, 1, 0, 0], [3]*8, 4)
    assert select_hard(*args, 4, 2, "hard_slack", 0) == ([1, 2, 3, 4], [])
    assert select_hard(*args, 4, 2, "hard_trigger", 0) == ([1, 2, 3, 4], [])
    assert select_hard(*args, 4, 2, "hard_trigger", 0, previous_strict={1}) == ([0, 1, 2, 3], [0])
    rows, seats = select_hard(*args, 6, 4, "hard_slack", 0)
    assert len(seats) == 2 and {1, 2, 3, 4} <= set(rows)


@pytest.mark.parametrize("passes", [True, False])
def test_screen_followed_by_exact_trial(passes):
    calls = []

    def probe(rows, pool, stage):
        calls.append((rows, pool, stage))
        return ({0: .7, 4: .8} if stage == "oracle_union" else {4: .6} if passes else {})

    base = [1, 2, 3]
    assert screened_swap(base, [True, True, False, False, True], {1}, range(5), probe) == (
        ([1, 2, 4], [4]) if passes else (base, []))
    assert calls == [([0, 1, 2, 3, 4], [0, 4], "oracle_union"), ([1, 2, 4], [4], "oracle_trial")]


def test_no_oracle_when_one_mask_and_only_fresh():
    def bad(*_):
        raise AssertionError("no spare seat")
    assert screened_swap([0, 2], [True, True, False], {0}, range(3), bad) == ([0, 2], [])
    with pytest.raises(ValueError):
        Arm.parse("16:0:14:hard_oracle1", 32)
    with pytest.raises(ValueError):
        Arm.parse("16:0:15:hard_oracle1", 256)


def test_readonly_cache_does_not_write_live_keys_or_values():
    import torch
    from bdlm_serving.methods.t2t.llada_sparse import _SelectedCache
    pos = torch.tensor([1, 3])
    k, v = torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4)
    orig = (k.clone(), v.clone())
    new = (torch.randn(1, 2, 2, 4), torch.randn(1, 2, 2, 4))
    cache = _SelectedCache(((k, v),), pos, 0, 5, shadow=True)
    out = cache.update(*new, 0)
    for current, old, update, result in zip((k, v), orig, new, out):
        assert torch.equal(current, old)
        assert torch.equal(result, old.index_copy(-2, pos, update))

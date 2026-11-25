import pytest

from bdlm_serving.methods.t2t.context_gate import CUTOFF, GATES, select
from bdlm_serving.methods.t2t.protocol import Arm


def go(mask, fresh=(), top=30, score=1., mode="hard_context_gate"):
    return select(range(32), [i not in mask for i in range(32)], fresh, [-1]*32, set(), 9,
                  8, mode, 91, 1, lambda rows, _: [score if i == top else -99.0 for i in rows])


@pytest.mark.parametrize("mode", GATES)
def test_physical_slack_placement(mode):
    q, rs, e = go({0, 1}, fresh={2}, mode=mode)
    assert e["reason"] == "swap_old" and e["victim"] == 7 and len(rs) == 1 and len(q) == 8
    assert set(e["base"]) - set(q) == {7} and set(q) - set(e["base"]) == set(rs)
    assert set(q) >= {0, 1, 2}
    if not mode.endswith("_random"):
        assert rs == [30]
    assert go({0, 1}, top=4, mode=mode)[1:] == ([], dict(
        base=list(range(8)), fresh=[], masks=[0, 1], old_pool=list(range(2, 32)),
        rank=4, score=1., victim=None, entrant=None, reason="already_selected"))
    assert Arm.parse("8:0:7:" + mode, 32).budget == 8
    with pytest.raises(ValueError):
        Arm.parse("4:0:3:" + mode, 32)


def test_floor_and_exact_threshold():
    m = set(range(10))
    for mode in GATES:
        q, rs, e = go(m, fresh=set(range(10, 17)), mode=mode)
        assert e["reason"] == "protected_mask" and not rs and 0 in q
        _, rs, e = go(m, fresh=set(range(10, 20)), mode=mode)
        assert e["reason"] == "overflow" and not rs
        q, rs, e = go(m, score=CUTOFF, mode=mode)
        if "slack" in mode:
            assert not rs and 7 in q and e["reason"] == "no_slack"
        else:
            assert rs and 7 not in q and 0 in q and e["reason"] == "swap_mask"
            assert go(m, score=CUTOFF-.0001, mode=mode)[2]["reason"] == "below_threshold"

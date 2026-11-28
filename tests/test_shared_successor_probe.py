import pytest

from bdlm_serving.methods.t2t import unified_rows as rows
from tools.shared_successor_probe import instrument


@pytest.mark.parametrize("b", (32,256))
def test_first_2b_offer_admits_only_allowed_successor(b):
    start=7
    def fresh(o):
        return rows.Block(o,b,frozenset(range(o,o+b)),frozenset(),frozenset())
    h,n=fresh(start),fresh(start+b)
    original=rows.plan_rows
    for arm, count in (("head_only",b),("allow_successor",2*b)):
        with instrument(arm) as bill:
            out=rows.plan_rows(2*b,b,h,n,params=rows.Params())
            assert out.rows==tuple(range(start,start+count))
            assert bill["tail_Q"]==count-b and bill["head_Q"]==b
            assert bill["tail_sampled_Q"]==count-b and bill["ordinary_Q"]==count
        assert original is rows.plan_rows


def test_never_drop_debt_or_allow_variable_offers():
    h=rows.Block(17,4,{18},{17},{17,18,19,20})
    n=rows.Block(21,4,{23,24},{22},{21,22})
    with instrument("head_only"):
        with pytest.raises(ValueError,match="established"):
            rows.plan_rows(8,4,h,n,opened=True,params=rows.Params())
        with pytest.raises(ValueError,match="2B"):
            rows.plan_rows(4,4,h,None,params=rows.Params())
    with instrument("allow_successor") as bill:
        out=rows.plan_rows(8,4,h,n,opened=True,params=rows.Params())
        assert set(out.dirty)=={17,22} and bill["tail_dirty_Q"]==1

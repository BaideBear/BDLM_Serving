import pytest
from experiments.t2t.serving_slots.coverage_span import select
from experiments.t2t.serving_slots.coverage_span_report import TIERS, decide


def test_debt_has_priority_and_may_be_parked():
    blocks=[(0,8,{3,4,5})]
    init=set(range(8))|{35,36}
    plan,sample=select(2,blocks,init,{35,36})
    assert plan.rows==(35,36) and not sample
    plan,sample=select(4,blocks,init,{35,36})
    assert plan.rows==(3,4,35,36) and sample==(3,4)


def test_prefix_quota_and_spare_capacity_are_different():
    blocks=[(0,16,set(range(8,16))),(16,24,set(range(16,24)))]
    init=set(range(8))
    plan,_=select(8,blocks,init,set())
    assert plan.rows==tuple(range(7,15))
    plan,_=select(24,blocks,init,set())
    assert plan.rows==tuple(range(24))
    for q in TIERS:
        plan,s=select(q,blocks,init,{4,5})
        assert {4,5} <= set(plan.rows) and set(s)<=set(plan.rows)
        assert len(plan.rows)==len(set(plan.rows))<=q


def test_pilot_needs_high_tier_gain_and_does_not_mix_arms():
    def row(tpf=1, score=9):
        return dict(macro_TPF=tpf, correct=score, new_failures=0)
    data={q:{"cohorts":{"all":dict(n=9,live=row(),span=row(2 if q<32 else 1),covered=row())}} for q in TIERS}
    assert decide(data)["expand"] is None
    data[32]["cohorts"]["all"]["span"]=row(2)
    assert decide(data)["expand"]=="span"
    data[16]["cohorts"]["all"]["span"]=row(2,7)
    assert decide(data)["expand"] is None
    with pytest.raises(ValueError):
        decide({32:data[32]})

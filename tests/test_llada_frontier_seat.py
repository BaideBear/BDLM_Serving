from dataclasses import replace

from bdlm_serving.methods.t2t.budget_family import Plan
from experiments.t2t.serving_slots.llada_frontier_seat import exchange


def test_exchange_preserves_width_dirty_sampler_arity_and_contiguous_build():
    p=Plan((4,9,10,11),(4,),(9,10,11),(),(),False)
    x,s,change=exchange(p,p.cold,{8,7})
    assert change==(11,7)
    assert x==Plan((4,7,9,10),(4,),(9,10),(7,),(),False)
    assert s==(7,9,10)


def test_exchange_never_trades_the_only_cold_row_or_a_debt_row():
    p=Plan((4,9),(4,),(9,),(),(),False)
    assert exchange(p,p.cold,{7})==(p,p.cold,None)
    p=replace(p,rows=(4,9,10),cold=(9,10))
    for warm in ({4},{11},set()):
        assert exchange(p,p.cold,warm)==(p,p.cold,None)

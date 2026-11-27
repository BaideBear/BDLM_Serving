from dataclasses import replace

from bdlm_serving.methods.t2t.budget_family import Plan
from experiments.t2t.serving_slots.llada_frontier_seat import exchange, interior_exchange
from experiments.t2t.serving_slots.llada_selection_geometry import policy


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


def test_interior_exchange_only_replaces_a_measured_mask():
    p=Plan((2,4,6,8),(2,),(),(6,8),(4,),False)
    x,s,change=interior_exchange(p,p.rows,{7,9})
    assert x==Plan((2,4,6,7),(2,),(),(6,),(4,),False)
    assert s==(2,4,6) and change==(8,7)
    p=replace(p,mask=(6,))
    assert interior_exchange(p,p.rows,{7})==(p,p.rows,None)


def test_ibs_candidate_keeps_settled_suffix_rows_then_clips_for_debt():
    masked={2,4,5,6}
    assert policy(2,8,masked,{7},4)==((2,3,4,7),(2,3,4,7))
    assert policy(2,8,masked,{7},2)==((2,3,7),(2,7))

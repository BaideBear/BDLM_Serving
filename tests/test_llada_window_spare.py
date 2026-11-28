from copy import deepcopy

import pytest

from bdlm_serving.methods.t2t.budget_family import Plan
from experiments.t2t.serving_slots.llada_window_spare import analyze,spare


def test_only_resident_clean_interior_can_use_unspent_width():
    p=Plan((0,3,4,6),(4,),(6,),(3,),(0,),False)
    resident={0,1,2,3,4,5,7,9,32,33}
    new,extra=spare(p,(6,),{3,6,7},resident,32,8)
    assert extra==(5,9) and new.rows==(0,3,4,5,6,9)
    assert (new.dirty,new.cold,new.mask,new.prefix)==(p.dirty,p.cold,p.mask,p.prefix)
    assert spare(p,(6,),{3,6,7},resident,32,5)[1]==(5,)
    assert spare(p,(6,),{3,6,7},resident,32,4)==(p,())
    assert spare(p,(),{3},resident,32,8)==(p,())


def test_unqualified_exposure_is_rejected():
    x=dict(rows=[1],sampled=[1],mask=[1],cold=[],dirty=[],prefix=[],tick=0,pre_state=dict(head=0,last_end=10),
           phase="sample",visible_kv=10,sample_clock=1,q=1,layer_physical_q=[1]*20)
    y=dict(deepcopy(x),spare_read=[4],rows=[1,4],q=2,layer_physical_q=[2]*20)
    a=dict(trace=[x]);b=dict(trace=[y])
    assert analyze(a,b,8)["extra_Q"]==1
    with pytest.raises(ValueError,match="resident read"):
        analyze(a,dict(trace=[dict(y,sampled=[1,4])]),8)
    with pytest.raises(ValueError,match="first read"):
        analyze(a,dict(trace=[dict(y,visible_kv=12)]),8)
    with pytest.raises(ValueError,match="before extra"):
        analyze(a,dict(trace=[dict(x,sample_clock=2)]),8)

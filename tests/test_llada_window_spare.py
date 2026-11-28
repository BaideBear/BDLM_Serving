from copy import deepcopy

import pytest

from bdlm_serving.methods.t2t.budget_family import Plan
from experiments.t2t.serving_slots.llada_window_spare import analyze,prefix_spare,spare,waiting


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


def test_single_extra_waiting_mask_without_displacing_paid_work():
    p=Plan((2,5,8),(2,),(8,),(5,),(),False)
    masks={2,3,4,5,8,9}; mem=set(range(8))
    new,s,extra=waiting(p,(5,8),masks,mem,32,8)
    assert extra==(3,) and new.mask==(3,5) and s==(3,5,8)
    assert new.rows==(2,3,5,8) and new.dirty==p.dirty and new.cold==p.cold
    assert waiting(p,(5,8),masks,mem,32,3)==(p,(5,8),())
    assert waiting(p,(5,8),{2,5,8},mem,32,8)==(p,(5,8),())
    warm=Plan((2,5),(2,),(),(5,),(),False)
    assert waiting(warm,(5,),masks,mem,32,8)==(warm,(5,),())


def test_extra_mask_must_be_independently_measured():
    x=dict(rows=[10],sampled=[10],mask=[],cold=[10],dirty=[],prefix=[],pre_state=dict(head=0,last_end=10),
           phase="sample",visible_kv=11,sample_clock=1,q=1,layer_physical_q=[1]*20)
    y=dict(x,rows=[3,10],sampled=[3,10],mask=[3],spare_waiting=[3],proposals=[[99],[99]],admitted=[3],
           q=2,layer_physical_q=[2]*20)
    info=analyze(dict(trace=[x]),dict(trace=[y]),8,factor="waiting",mask_id=99)
    assert info["extra_admits"]==info["extra_Q"]==1
    with pytest.raises(ValueError,match="unresolved"):
        analyze(dict(trace=[x]),dict(trace=[dict(y,proposals=[[13],[99]])]),8,factor="waiting",mask_id=99)


def test_prefix_addition_is_read_only_and_uses_nearest_two_resident_rows():
    p=Plan((2,5),(2,),(),(5,),(),False)
    p2,extra=prefix_spare(p,(5,),[4,3,1,0],8)
    assert extra==(4,3) and p2.rows==(2,3,4,5) and p2.mask==p.mask
    assert prefix_spare(p,(5,),[4,3,1,0],3)[1]==(4,)
    assert prefix_spare(p,(),[4],8)==(p,())
    start=dict(rows=[0,1,2],sampled=[0,1,2],mask=[],cold=[0,1,2],dirty=[],prefix=[],
        pre_state=dict(head=0,origin=0,last_end=0,end=32),phase="sample",visible_kv=3,
        sample_clock=1,q=3,layer_physical_q=[3]*20,admitted=[0,1,2],revoked=[])
    old=dict(start,rows=[3],sampled=[3],cold=[],mask=[3],q=1,layer_physical_q=[1]*20,
        pre_state=dict(head=0,origin=0,last_end=8,end=32),visible_kv=8,
        sample_clock=2,admitted=[],revoked=[])
    new=dict(old,rows=[1,2,3],q=3,layer_physical_q=[3]*20,spare_prefix=[2,1])
    assert analyze({"trace":[start,old]},{"trace":[start,new]},8,factor="prefix")["extra_Q"]==2
    with pytest.raises(ValueError,match="right edge"):
        analyze({"trace":[start,old]},{"trace":[start,dict(new,rows=[0,1,3],spare_prefix=[1,0])]},8,factor="prefix")

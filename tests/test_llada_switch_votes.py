import pytest
from experiments.t2t.serving_slots.llada_switch_votes import audit


def event(extra=(),head=0):
    return dict(rows=[head,head+1],sampled=[head,head+1],
                proposals=[(0,1,.95,True,True),(0,2,.75,True,False)],admitted=[head,*extra])


def test_two_consecutive_paid_q64_observations_only():
    first,extra=audit(64,0,True,event(),0,{})
    assert not extra and first=={1:(0,2)}
    assert audit(64,0,True,event([1]),0,first)[1]=={1}
    for q,warm in ((32,True),(8,True),(64,False)):
        votes,extra=audit(q,0,warm,event(),0,first)
        assert not votes and not extra
        with pytest.raises(ValueError,match="wrong admission"):
            audit(64,0,True,event([1]),0,votes)
    empty=dict(rows=[1],sampled=[],admitted=[])
    assert audit(64,0,True,empty,0,first)==({},set())
    assert not audit(64,32,True,event(head=32),0,first)[1]

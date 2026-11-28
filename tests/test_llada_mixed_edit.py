from copy import deepcopy

import pytest

from bdlm_serving.methods.t2t.budget_family import Plan
from experiments.t2t.serving_slots.llada_mixed_edit import extra,inspect


def test_adds_projected_decoded_only_never_prefix_dirty_or_waiting_mask():
    p=Plan((0,1,2,3,4,5),(1,),(5,),(2,),(0,),False)
    assert extra(p,(2,5),{2,5,6},set(range(5)))==(3,4)
    assert not extra(p,(),{2,5},set(range(5)))


def test_first_new_write_requires_identical_common_proposals():
    st=dict(origin=0,head=0,last_end=4,end=32)
    def event(rows,sampled,cold,admitted,old):
        return dict(rows=rows,sampled=sampled,cold=cold,admitted=admitted,proposals=old,
                    pre_state=st,dirty=[],prefix=[],mask=[],layer_physical_q=[len(rows)]*20,
                    q=len(rows),sample_clock=1,phase="sample",visible_kv=5,revoked=[],
                    edited=[],written_absolute=admitted)
    start=event([0,1,2],[0,1,2],[0,1,2],[0,2],[[9],[9],[9]])
    a=event([2,4],[4],[4],[],[[9]])
    b=dict(a,sampled=[2,4],proposals=[[3],[9]],mixed_edit=[2],edited=[2],written_absolute=[2])
    first,v=inspect(dict(trace=[start,a]),dict(trace=[start,b]),9)
    assert first["tick"]==1 and v==dict(calls=1,seats=1,T2T=1,revoked=0)
    bad=deepcopy(b);bad["proposals"][1]=[3]
    with pytest.raises(ValueError,match="causal"):
        inspect(dict(trace=[start,a]),dict(trace=[start,bad]),9)

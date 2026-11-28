from copy import deepcopy

import pytest
import torch

from experiments.t2t.serving_slots.llada_frontier_accept import audit, check_first, frontier_class


def event(pos, confidence, admitted, *, old=None, drafts=None, last_end=32):
    old=old or [0]*len(pos)
    drafts=drafts or [11]*len(pos)
    return dict(sampled=pos,rows=pos,q=len(pos),dirty_before=[],dirty_after=admitted,
                proposals=[(o,d,c,c>.5,c>.9) for o,d,c in zip(old,drafts,confidence)],
                pre_state=dict(head=0,origin=2,end=64,last_end=last_end,stopping=False),
                admitted=admitted,edited=[],revoked=[],layer_physical_q=[len(pos)]*20,
                phase="sample",sample_clock=1,tail_opened=False)


@pytest.mark.parametrize("q",[32,64])
def test_covered_frontier_stops_at_hole_or_weak_token_and_never_enters_tail(q):
    pos=[2,3,5,6,32,33]
    a=event(pos,[.55,.6,.95,.7,.7,.8],[5,33])
    b={**a,"admitted":[2,3,5,33],"dirty_after":[2,3,5,33]}
    result=check_first({"trace":[a]},{"trace":[b]},0,q)
    assert result==({"tick":0,"rows":[2,3],"width":6},2,1)
    b=deepcopy(a); b["proposals"][0]=(0,11,.5,False,False)
    assert audit([b],0,q,True)==[set()]
    b["admitted"].append(32)
    with pytest.raises(ValueError,match="explained"):
        audit([b],0,q,True)


def test_actual_selection_preserves_incumbent_and_never_admits_after_missing_frontier():
    class Prior:
        def _additional_admissions(self,idx,old,draft,conf,accepted):
            return idx[None,:]==10
    req=frontier_class(Prior)();req._left=3;req.head=0;req.mask_id=0
    idx=torch.tensor([3,4,6,10,32]); old=torch.tensor([[0,0,0,0,0]])
    draft=torch.tensor([[12,16,12,12,12]]);conf=torch.tensor([[.6,.51,.84,.52,.8]])
    assert req._additional_admissions(idx,old,draft,conf,old.bool()).tolist()==[[True,True,False,True,False]]
    req._left=2
    assert req._additional_admissions(idx,old,draft,conf,old.bool()).tolist()==[[False,False,False,True,False]]
    req._left=None
    assert req._additional_admissions(idx,old,draft,conf,old.bool()).tolist()==[[False,False,False,True,False]]


def test_cold_head_cannot_extend_despite_a_contiguous_qualified_run():
    e=event([2,3],[.6,.91],[3],last_end=10)
    assert audit([e],0,32,True)==[set()]
    e["admitted"]=[2,3]
    with pytest.raises(ValueError,match="explained"):
        audit([e],0,32,True)


def test_solo_rule_disables_the_new_write_on_first_or_existing_successor_projection():
    for pos,last_end in (([2,3,32],32),([2,3],40)):
        old=event(pos,[.6]+[.95]*(len(pos)-1),pos[1:],last_end=last_end)
        excluded=[]
        assert audit([old],0,64,True,solo=True,excluded=excluded)==[set()] and excluded==[1]
        with pytest.raises(ValueError,match="explained"):
            audit([old],0,64,True)
    one=event([2,3],[.6,.95],[2,3])
    assert audit([one],0,64,True,solo=True)==[{2}]


def test_switched_frontier_permission_does_not_survive_a_downshift():
    from types import SimpleNamespace
    from experiments.t2t.serving_slots.llada_switch_votes import audit as audit_votes

    class Request:
        def _step(self,q):
            return self._left
    req=frontier_class(Request,only_offer=32)()
    req.head=req.origin=0;req.end=32;req.stopping=False;req.mask_id=0
    req.bank=SimpleNamespace(last_end=32)
    req.x=torch.tensor([[9,9,0,0]])
    assert [req._step(q) for q in (4,32,16,64,32)]==[None,2,None,None,2]
    e=event([2,3],[.6,.95],[2,3])
    added=set()
    assert audit_votes(32,0,True,e,0,{},frontier_left=2,frontier_added=added)==({},set())
    assert added=={2}
    with pytest.raises(ValueError,match="admission"):
        audit_votes(16,0,True,e,0,{})

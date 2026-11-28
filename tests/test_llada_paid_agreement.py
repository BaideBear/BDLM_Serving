"""Selection, proposal history and first-difference audits; no model inference."""
from copy import deepcopy

import pytest
import torch

from experiments.t2t.serving_slots.llada_agreement import confirmation, inspect_trace


def test_temporal_votes_are_per_head_address_and_not_extra_edits():
    p=torch.tensor([32,33,34,35,36,37,63,64])
    old=torch.tensor([[0,0,0,0,4,0,0,0]])
    draft=torch.tensor([[5,6,7,0,8,9,11,12]])
    c=torch.tensor([[.81,.72,.5,.75,.79,.8,.73,.89]])
    accept=torch.tensor([[False,False,False,False,False,True,False,False]])
    prior=torch.full((32,),-1)
    first,memo=confirmation(32,p,old,draft,c,accept,prior,0)
    assert not first.any()
    assert memo[0:6].tolist()==[5,6,-1,-1,-1,-1] and memo[-1].item()==11
    draft[0,1]=13
    extra,_=confirmation(32,p,old,draft,c,accept,memo,0)
    assert p[extra[0]].tolist()==[32,63]


def event(head=32,warm=True):
    return dict(pre_state=dict(head=head,end=128,last_end=head+(32 if warm else 10)),
                sampled=[head+2,head+3,head+32],
                proposals=[(0,8,.81,True,False),(0,7,.95,True,True),(0,9,.71,True,False)],
                admitted=[head+3,head+32])


def test_ledger_disallows_drain_cold_or_different_head_as_first_vote():
    e=event()
    drain=dict(pre_state=e["pre_state"],sampled=[],admitted=[])
    assert inspect_trace([e,drain,e],0,True)[0]==[[],[],[]]
    assert inspect_trace([event(warm=False),e],0,True)[0]==[[],[]]
    assert inspect_trace([e,event(head=64)],0,True)[0]==[[],[]]
    altered=deepcopy(e)
    altered["admitted"].append(34)
    assert inspect_trace([e,altered],0,True)[0]==[[],[34]]
    with pytest.raises(ValueError,match="do not explain"):
        inspect_trace([e,drain,altered],0,True)
    with pytest.raises(ValueError,match="do not explain"):
        inspect_trace([e,altered],0,False)


def test_retained_decoded_or_unchanged_mask_cannot_supply_temporal_vote():
    e=event()
    other=deepcopy(e)
    other["proposals"][0]=(2,8,.81,True,False)
    assert inspect_trace([other,e],0,True)[0]==[[],[]]
    other["proposals"][0]=(0,0,.81,True,False)
    assert inspect_trace([other,e],0,True)[0]==[[],[]]

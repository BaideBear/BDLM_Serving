from types import SimpleNamespace as NS

import pytest

from experiments.t2t.serving_slots.gemma_waiting_seat import alignment, arm_class, gates, replacement


def test_age_and_identity():
    selected=set(range(64))
    stamps=[1]*256
    assert replacement(selected,{63},{27},stamps) is None
    stamps[80]=stamps[81]=0
    assert replacement(selected,{62,63},{27},stamps)==(61,80)
    assert replacement(selected,selected-{0},set(),stamps) is None
    assert replacement(selected,set(),set(range(1,256)),stamps) is None
    with pytest.raises(ValueError):
        replacement(selected,{64},set(),stamps)


@pytest.mark.parametrize("enabled,warm,route,expected",[
    (False,True,"sparse",False),(True,True,"sparse",True),(True,False,"sparse",False),
    (True,True,"drain",False),(True,True,"verify",False),(True,True,"verify_paid",False)])
def test_one_parent_call(enabled,warm,route,expected):
    calls=[]
    class Parent:
        def _apply(self,states,pos,sample,prefix,cold,offer,route):
            calls.append((pos,sample))
            return {"positions":list(pos),"parts":[dict(sampled=sorted(sample),prefix=list(prefix))]}
    tape=[]
    req=arm_class(Parent,enabled,tape)()
    req.origin=req.tick=0
    req.rebuild,req.tail=False,None
    stamps=[1]*256
    stamps[99]=0
    h=NS(ordinal=0,rows=NS(age=9,_stamps=stamps,dirty={1},accepted={0},
        initialized=set(range(256 if warm else 255))),sampler=NS(sc=set(range(256))),can_sample=True)
    orig=tuple(range(64))
    sample=set(orig) if route=="sparse" else set()
    req._apply([h],orig,sample,(0,) if sample else (),(),64,route)
    assert len(calls)==len(tape)==1
    assert set(calls[0][0])==(set(range(63))|{99} if expected else set(orig))
    assert calls[0][1]==(set(calls[0][0]) if sample else set())


def test_gate_cannot_hide_losses_or_added_work():
    def row(i,f,on=True):
        return dict(index=i,first=True,off=dict(F=10,correct=True),on=dict(F=f,correct=on))
    assert all(gates([row(0,9),row(10,11),row(20,10)]).values())
    assert not gates([row(0,8,False),row(10,10),row(20,10)])["no_losses"]
    assert not gates([row(0,9),row(10,13),row(20,10)])["total_F_not_higher"]
    assert not gates([row(0,10),row(10,10),row(20,10)])["one_F_saving"]


def test_without_exchange_all_outcomes_match():
    notes=[[{"swap":None}],[{"swap":None}]]
    ref=dict(tape=["same"],output_ids=[3])
    assert alignment(ref,ref,notes) is None
    with pytest.raises(ValueError,match="Unexplained"):
        alignment(ref,dict(tape=["same"],output_ids=[9]),notes)


def test_first_exchange_requires_same_state_and_first_physical_difference():
    notes=[[{"swap":None},{"swap":{"state":"abc"},"tick":1}] for _ in range(2)]
    off=dict(tape=["start","ref"])
    assert alignment(off,dict(tape=["start","alt"]),notes)["tick"]==1
    with pytest.raises(ValueError,match="First different"):
        alignment(off,dict(tape=["different","alt"]),notes)
    with pytest.raises(ValueError,match="First different"):
        alignment(off,off,notes)
    notes[1][1]["swap"]["state"]="xyz"
    with pytest.raises(ValueError,match="First different"):
        alignment(off,dict(tape=["start","alt"]),notes)

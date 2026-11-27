from types import SimpleNamespace as NS

import pytest

from experiments.t2t.serving_slots.gemma_joint_verify import B, FULL, validate_plan, validate_tick


@pytest.mark.parametrize("hd,td", [({1,9},{0,2}),(set(),{7}),({3},set()),(FULL,FULL)])
def test_one_full_head_plus_exact_tail(hd,td):
    base=13
    h=NS(ordinal=0,can_sample=False,sampler=NS(phase="sample"),rows=NS(initialized=FULL,dirty=hd))
    t=NS(ordinal=1,rows=NS(dirty=td))
    req=NS(origin=base,ordinal=0,joint_verify=True,sample_full_rebuild=True,fuse_first_verify=True,
           top_gate="half",tail=t,rebuild=False,cursor=0,bank=NS(causal=True))
    pos=tuple(range(base,base+B))+tuple(base+B+i for i in sorted(td))
    validate_plan(req,[h,t],pos,set(),2*B)
    with pytest.raises(ValueError):
        validate_plan(req,[h,t],pos[:-1],set(),2*B)
    with pytest.raises(ValueError):
        validate_plan(req,[h,t],pos,{pos[-1]},2*B)
    req.bank.causal=False
    with pytest.raises(ValueError):
        validate_plan(req,[h,t],pos,set(),2*B)


def test_tail_clock_and_debt_are_not_optional():
    h=dict(phase="verify",rows=list(range(B)),age_before=4)
    t=dict(phase="drain",rows=[9],dirty_before=[9],age_before=3,age_after=3,dirty_after=[])
    e=dict(ordinal=0,parts=[h,t],offer=2*B,resident_at_close=1,head_mask_before=0,
           sampled_absolute=[],dirty_after_absolute=[],written_absolute=[])
    seen=set(range(B)); due={7,B+9}; pos=sorted(seen|due)
    req=NS(joint_verify=True,sample_full_rebuild=True)
    validate_tick(e,0,[],due,seen,pos,req)
    for k,v in (("age_after",4),("dirty_after",[9]),("phase","sample")):
        with pytest.raises(ValueError):
            validate_tick({**e,"parts":[h,{**t,k:v}]},0,[],due,seen,pos,req)


@pytest.mark.parametrize("on,resident,hd,td,q,expected", [
    (True, True, [], [], 512, "verify"),
    (False, True, [2], [4], 512, "drain"),
    (True, True, [2], [4], 512, "verify_joint_paid"),
    (True, True, [], [4], 512, "verify_joint_paid"),
    (True, True, [2], [], 512, "verify_joint_paid"),
    (True, False, [2], [], 512, "verify_paid"),
    (True, True, [2], [4], 256, "drain"),
])
def test_dispatch_only_at_real_joint_opportunity(on,resident,hd,td,q,expected):
    from test_gemma_ablation_plans import request
    r = request("00")
    r.joint_verify, r.fuse_first_verify, r.sample_full_rebuild = on, True, True
    r.top_gate = "half"
    r.rows.initialized = r.rows.accepted = FULL
    r.rows.dirty = frozenset(hd)
    if resident:
        r.tail = r._spawn()
        r.tail.rows.dirty = frozenset(td)
    e = r._advance(q)
    assert e["route"] == expected and not e["sample"]
    if expected == "verify_joint_paid":
        assert e["positions"] == tuple(range(r.origin,r.origin+B)) + tuple(r.origin+B+j for j in td)

"""Planner/replay only. No checkpoint or model execution."""
from copy import deepcopy

import pytest

from experiments.t2t.serving_slots.gemma_cold_pacing import first_batch, make_auditor
from experiments.t2t.serving_slots.gemma_cold_pacing_audit import B, replay


def tape(q, pace, dirty=True):
    seen, debt, age, out = set(), set(), 0, []
    def add(rows, sample, route, written):
        nonlocal debt, age
        p = dict(phase="verify" if route.startswith("verify") else "sample" if sample else "drain",
                 ordinal=0,rows=sorted(rows),sampled=sorted(sample),prefix=[],admitted=[],revoked=[],
                 input_changed=sorted(written),age_before=age,dirty_before=sorted(debt))
        out.append(dict(parts=[p],tick=len(out),ordinal=0,offer=q,opened=None,
                        head_initialized_before=len(seen),head_mask_before=B if len(seen)<B else 0,
                        q=len(rows),decoder_Q=len(rows),route=route))
        debt = (debt-rows)|written
        age += bool(sample)
        p.update(age_after=age,dirty_after=sorted(debt))
        seen.update(rows)
    while len(seen)<B:
        first = set(sorted(set(range(B))-seen)[:q-len(debt)])
        rows = debt|first
        if first and pace:
            rows,first,_ = first_batch(rows,first,first,debt,q)
        add(set(rows),first,"sparse" if first else "drain",first if dirty else set())
    if debt:
        add(debt,set(),"drain",set())
    for k in range(0,B,q):
        add(set(range(k,k+q)),set(),"verify",set())
    return out


@pytest.mark.parametrize("q",(32,64,128))
@pytest.mark.parametrize("dirty",(False,True))
def test_first_only_progress(q,dirty):
    for pace in (False,True):
        trace = tape(q,pace,dirty)
        s,blocks = replay(trace,q,pace)
        chunk = q//2 if pace else q
        assert blocks[0]["first_width"] == chunk and blocks[0]["init_clock"] == B//chunk
        assert s["cold_Q"] == B+s["cold_paid_Q"] and s["verify_Q"] == B
        assert s["overlap_F"] == ((B//chunk)-1 if pace and dirty else 0)
        assert s["cold_drain_F"] == ((B//q)-1 if dirty and not pace else 0)


def test_mixed_or_missing_payment():
    with pytest.raises(ValueError,match="mandatory"):
        first_batch((10,11),{10,11},(10,11),{3},32)
    with pytest.raises(ValueError,match="Mixed"):
        first_batch((3,10),{3,10},(10,),{3},32)


def test_corrupted_clock_and_debt():
    orig = tape(32,True)
    for key,value in (("age_after",3),("dirty_after",[])):
        t = deepcopy(orig)
        t[1]["parts"][0][key] = value
        with pytest.raises(ValueError,match="second clock"):
            replay(t,32,True)


def test_auditor_is_private():
    from experiments.t2t.serving_slots import gemma_fixed_dual_eval as shared
    from experiments.t2t.serving_slots.gemma_fixed_dual import AblationRequest
    from experiments.t2t.serving_slots.gemma_cold_pacing import ColdPacingRequest
    local = make_auditor()
    assert shared.AblationRequest is AblationRequest
    assert local.__wrapped__.__globals__["AblationRequest"] is ColdPacingRequest


@pytest.mark.parametrize("pace,route,cold", ((True,"sparse",()),(True,"verify_paid",()),
                                            (False,"sparse",(10,11))))
def test_unaffected_branches_delegate_unchanged(monkeypatch,pace,route,cold):
    from experiments.t2t.serving_slots.gemma_cold_pacing import AblationRequest, ColdPacingRequest
    req = object.__new__(ColdPacingRequest)
    req.pace = pace
    args = ([object()],(10,11),{10,11} if route=="sparse" else set(),(),cold,32,route)
    calls = []
    monkeypatch.setattr(AblationRequest,"_apply",lambda self,*a:calls.append(a))
    req._apply(*args)
    assert len(calls)==1 and all(a is b for a,b in zip(args,calls[0],strict=True))

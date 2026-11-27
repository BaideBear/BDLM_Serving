"""Unit tests on receipts, not CPU inference."""
from copy import deepcopy

import pytest

from experiments.t2t.serving_slots.llada_admission_factor import decisions, paired


def arm(cutoff):
    e=dict(tick=0,block=0,q=5,rows=[29,30,31,32,33],sampled=[29,30,31,32,33],
           proposals=[(0,1,.8,True,False),(0,2,.7,True,False),(4,0,.8,True,False),
                      (0,0,.3,False,False),(0,1,.2,False,False)],
           dirty_before=[],edited=[],revoked=[31],retired_blocks=[])
    e["admitted"]=[29] if cutoff=="high" else [29,30]
    e["dirty_after"]=e["written_absolute"]=e["admitted"]+[31]
    return dict(trace=[e],row_visits=dict(m2t=len(e["admitted"])),F=1,Q=5,
                correct=False,failure=None,output_ids=[])


def test_independent_fallback_by_physical_block_including_mask_to_self():
    high,low=arm("high"),arm("low")
    event,counts=paired(high,low,0)
    assert event==dict(tick=0,Q=5,high=1,low=2,more=1)
    assert counts["high"]==dict(natural=0,fallback=1,fallback_groups=2,edit=0,revoke=1)
    assert counts["low"]==dict(natural=2,fallback=0,fallback_groups=1,edit=0,revoke=1)


def test_wrong_first_rows_or_update_fails():
    high,low=arm("high"),arm("low")
    low["trace"][0]["rows"]=[28,30,31,32,33]
    with pytest.raises(ValueError):
        paired(high,low,0)
    e=deepcopy(high["trace"][0])
    e["admitted"]=[30]
    with pytest.raises(ValueError):
        decisions(e,0,"high")


def test_relax_only_a_previously_initialized_head_not_a_joint_tail():
    # The second block has a 0.7 candidate that a global 0.5 rule would admit.
    e=dict(block=3,sampled=[98,99,128,129],
           proposals=[(0,3,.8,True,False),(0,4,.7,True,False),
                      (0,5,.95,True,True),(0,6,.7,True,False)],
           admitted=[98,99,128],edited=[],revoked=[])
    r=decisions(e,0,"warm_head",True)
    assert (r["natural"],r["fallback"])==(3,0)
    with pytest.raises(ValueError):
        decisions(e,0,"warm_head",False)
    e["admitted"]=[98,128]
    r=decisions(e,0,"warm_head",False)
    assert (r["natural"],r["fallback"])==(1,1)


def test_a_cold_head_cannot_claim_the_warm_threshold():
    high,low=arm("high"),arm("low")
    low["trace"][0]["admission_head_warm_before"]=True
    with pytest.raises(ValueError,match="pre-forward warm"):
        paired(high,low,0,"warm_head")

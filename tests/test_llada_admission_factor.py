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

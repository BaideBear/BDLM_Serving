"""Synthetic billing records only; no model loading."""
from copy import deepcopy
import pytest

from experiments.t2t.serving_slots.llada_close_factor import proof


def receipt():
    step=dict(phase="sample",tick=0,block=0,rows=[30,31],q=2,sampled=[30,31],
              dirty_before=[],dirty_after=[31],written_absolute=[31],retired_blocks=[])
    last=dict(phase="drain",tick=1,block=0,head_mask_at_plan=0,dirty_before=[31],
              rows=[31],q=1,sampled=[],dirty_after=[],written_absolute=[],retired_blocks=[0],
              cold=[],mask=[],admitted=[])
    rows={}
    for arm,phase in [("unchanged","drain"),("full_write","close_full_write"),("edit_once","close_edit_once")]:
        event=deepcopy(last)
        if arm!="unchanged":
            event.update(phase=phase,q=2,rows=[30,31])
        if arm=="edit_once":
            event.update(sampled=[30,31],dirty_after=[30],written_absolute=[30],retired_blocks=[])
        rows[arm]=dict(trace=[deepcopy(step),event],failure=None)
    return rows


def test_full_terminal_rows_exclude_prompt_and_can_edit_only_in_trial():
    marks=proof(receipt(),30)
    assert not marks["unchanged"] and not marks["full_write"][0]["sampled"]
    assert marks["edit_once"][0]["written_absolute"]==[30]


@pytest.mark.parametrize("error",["paid_width","lost_write","same_block_again","stale_start","writes_in_control"])
def test_no_free_work_or_unpaired_first_change(error):
    r=receipt()
    e=r["edit_once"]["trace"][1]
    if error=="paid_width":
        e["q"]=33
    elif error=="lost_write":
        e["dirty_after"]=[]
    elif error=="same_block_again":
        r["edit_once"]["trace"].append(e)
    elif error=="stale_start":
        r["edit_once"]["trace"][0]["dirty_before"]=[30]
    else:
        r["full_write"]["trace"][1]["sampled"]=[30]
    with pytest.raises(ValueError):
        proof(r,30)

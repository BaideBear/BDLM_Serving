from copy import deepcopy

import pytest

from experiments.t2t.serving_slots.llada_reference import PARITY, verify_row


def test_baseline_parity_ignores_time_but_not_status_work_or_positions():
    old = dict.fromkeys(PARITY,0)
    old.update(policy="optimus_ibs_c4",status="generation_failed",output_ids=[],
               calls=[dict(rows=4,physical_rows=4,batch=1,kv_rows=50)],
               events=[dict(phase="denoise",q_tokens=4,absolute_positions=[2,3,4,5])])
    new = deepcopy(old)
    new["e2e_shared_gpu_wall_ms"] = 1000
    new["calls"][0]["batch_call_id"] = 255
    verify_row(old,new)
    for key, value in (("status","success"),("generation_forwards",1),("output_ids",[1])):
        with pytest.raises(ValueError):
            verify_row(old,{**new,key:value})
    new["events"][0]["absolute_positions"] = [3,4,5,6]
    with pytest.raises(ValueError):
        verify_row(old,new)

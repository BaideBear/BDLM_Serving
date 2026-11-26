from copy import deepcopy

import pytest

from experiments.t2t.serving_slots.llada_reference import PARITY, verify_row
from tools.prefix_cohorts import normalize_request


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


@pytest.mark.parametrize("mode,chunk,width", [("ibs",4,32),("obs",32,64)])
def test_progress_limit_counts_as_wrong_and_charges_real_work(mode,chunk,width):
    case=dict(index=0,dataset="gsm8k",task_id="x",seed=1707)
    failure=dict(type="optimus_progress_limit",delivered_output_tokens=0,cached_prefix=2,
                 step_limit=2*width+4,steps_in_block=2*width+4)
    q=chunk+1
    raw=dict(case,model="llada21",policy=f"optimus_{mode}_c{chunk}",phase="formal",generation_cap=2048,
             status="generation_failed",request_failure=failure,output_text="",output_tokens=0,output_ids=[],
             eos=False,truncated=False,max_q=q,q_histogram={str(q):1},generation_forwards=1,
             generation_query_rows=q,generation_physical_padded_rows=q,prefill_rows=0,
             calls=[dict(rows=q,physical_rows=q,batch=1)],extra=dict(mode=mode,fixed_chunk=chunk,
                 window_switches=0,request_failure=failure),events=[dict(phase="optimus",fixed_chunk=chunk,
                 mode=mode,q_tokens=q,physical_forwards=1,active_width=width,absolute_positions=list(range(q)))])
    row=normalize_request(raw,case,"baseline")
    assert (row["failed"],row["correct"],row["N"],row["F"],row["physical_Q"],row["nominal_q"])==(1,0,0,1,q,chunk)
    with pytest.raises(ValueError):
        normalize_request(dict(raw,status="timeout"),case,"baseline")


def test_packed_w2_terminal_rows_count_even_when_position_ids_repeat(monkeypatch):
    case=dict(index=1,dataset="gsm8k",task_id="test",seed=17)
    raw=dict(case,model="llada21",policy="flow_w2",phase="formal",generation_cap=2048,
             status="success",request_failure=None,output_text="ok",output_tokens=1,max_q=64,
             generation_forwards=1,generation_query_rows=64,generation_physical_padded_rows=64,
             prefill_rows=0,q_histogram={"64":1},calls=[dict(rows=64,physical_rows=64,batch=1)],
             events=[dict(phase="flowblock_packed",physical_forwards=1,q_tokens=64,
                 max_physical_q=64,physical_layer_attention_calls=20,query_rows_over_layers=1280,
                 absolute_positions=list(range(32))+[31]*32)])
    monkeypatch.setattr("tools.prefix_cohorts.score",lambda *a:{"correct":True})
    assert normalize_request(raw,case,"baseline")["physical_Q"]==64
    with pytest.raises(ValueError):
        normalize_request(dict(raw,generation_physical_padded_rows=32),case,"baseline")

import pytest

from experiments.t2t.serving_slots.llada_gate_cohorts import validate


def test_check_layer_width_and_read_only_work():
    task=dict(index=0,dataset="example",task_id="first",seed=7)
    spec=dict(arm="test",q=8,cross_open="head",borrow=True,coverage="sparse")
    tick=dict(stage="selected",actual_q=2,offer=8,physical_q=2,absolute_positions=[4,36],
              layer_physical_q=[2]*20,head_q=1,tail_q=1,F=1,tail_read_only=[36])
    row=dict(task,**spec,projection_modules=20,encoder_Q=0,device_uuid="a",correct=True,failure=None,
             samples_sha256="x",input_sha256="p",law=[1,8],trace=[tick],N=1,F=1,Q=2,
             head_Q=1,tail_Q=1,tail_read_only_Q=1,head_full_F=0,full64_F=0,decoder_Q=2)
    assert validate(row,spec,task,"x")["tail_read_only_Q"]==1
    with pytest.raises(ValueError,match="workload"):
        validate(row|{"tail_read_only_Q":0},spec,task,"x")
    with pytest.raises(ValueError,match="decoder tape"):
        validate(row|{"trace":[tick|{"layer_physical_q":[2]*19+[8]}]},spec,task,"x")

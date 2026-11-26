import pytest

from tools.run_dual_t2t import audit_dual64, saturation_summary


def sample():
    p=list(range(32,96))
    events=[dict(stage="selected",phase="denoise",F=1,actual_q=64,effective_q=64,physical_q=64,
                 absolute_positions=p,layer_physical_q=[64]*2,active_width=64,
                 saturated=True,full_two_block=True)]
    return dict(dual=dict(nominal_q=64,pair_blocks=2,gen_length=2048,mode="rolling_pair"),
                dual_saturation=saturation_summary(events),events=events,generation_query_rows=64,
                generation_physical_padded_rows=64,generation_forwards=1,max_q=64)


def test_true_dual_identity_layers_and_blocks_checked():
    row=sample()
    audit_dual64(row)
    for key,value in (("absolute_positions",list(range(32))*2),
                      ("absolute_positions",list(range(1,65))),
                      ("layer_physical_q",[32,32])):
        row=sample()
        row["events"][0][key]=value
        with pytest.raises(ValueError):
            audit_dual64(row)
    row=sample()
    row["dual"]["pair_blocks"]=1
    with pytest.raises(ValueError):
        audit_dual64(row)

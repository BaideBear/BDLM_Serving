import pytest

from bdlm_serving.methods.t2t.offer_gate import choose_native
from bdlm_serving.methods.t2t.slot_trace import RoundInfo
from experiments.t2t.serving_slots.switch_report import first_changed_offer


@pytest.mark.parametrize("n,tail",[(0,True),(64,True),(65,False),(256,False)])
def test_gate_complements_never_promote_low(n,tail):
    info=RoundInfo(5,73,256,n,177)
    assert choose_native(info,256,128,64,"tail").native_full==tail
    assert choose_native(info,256,128,64,"early").native_full!=tail
    for mode in ("all","tail","early"):
        low=choose_native(info,128,128,64,mode)
        assert not low.native_full and low.q==128


def test_first_offer_requires_common_preceding_observation_and_state():
    fields=dict(model="gemma",index=1,task_id="test",dataset="test",seed=1,prefill_rows=5)
    event=dict(block=5,clock=5,available_fresh=1,slot_tick=0)
    base=dict(fields,events=[{"stage":"bootstrap"},dict(event,slot_offer=256)])
    row=dict(fields,events=[{"stage":"bootstrap"},dict(event,slot_offer=128)])
    assert first_changed_offer(base,row)==1
    row["events"][1]["clock"]=6
    with pytest.raises(ValueError,match="pre-step"):
        first_changed_offer(base,row)
    row["events"][0]["extra"]=3
    with pytest.raises(ValueError,match="before"):
        first_changed_offer(base,row)

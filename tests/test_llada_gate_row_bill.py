from copy import deepcopy

import pytest

from tools.llada_gate_row_bill import IDLE, bill, unpack_idle


def fixture():
    def event(t, rows, before, cold, prefix, edits, after):
        return dict(tick=t, q=len(rows), rows=rows, sampled=rows, offer=64,
                    layer_physical_q=[len(rows)]*20, prefix_kv_length=32,
                    dirty_before=before, dirty=before, cold=cold, mask=[], prefix=prefix,
                    edited=edits, admitted=[], revoked=[], written_absolute=edits,
                    dirty_after=after, tail_q=0)
    return dict(F=2, Q=3, decoder_Q=3, encoder_Q=0, correct=True, N=1, failure=None,
                trace=[event(0, [32,33], [], [33], [32], [32], [32]),
                       event(1, [32], [32], [], [], [], [])])


def test_distinguish_refresh_and_edit():
    result=bill(fixture())
    assert (result["dirty_Q"], result["cold_Q"], result["prefix_Q"], result["prefix_T2T"],
            result["dirty_T2T"], result["history_Q"], result["edits"]) == (1,1,1,1,0,3,1)


@pytest.mark.parametrize("field,value", [
    ("dirty_before", []), ("prefix", [32]), ("layer_physical_q", []), ("prefix_kv_length", 0)])
def test_invalid_ledger_fails_closed(field,value):
    arm=deepcopy(fixture())
    arm["trace"][1][field]=value
    with pytest.raises(ValueError):
        bill(arm)


def test_q64_unused_is_not_the_number_of_pending_masks():
    trace=[]
    for i,(rows,dirty,admitted,opened) in enumerate([
        ([63],[],[], False), ([63,64,65],[],[64], True),
        ([63,64,65],[64],[], True), ([63,65],[],[65], True),
        ([63,65],[65],[], True),
    ]):
        trace.append(dict(tick=i,q=len(rows),offer=64,rows=rows,sampled=rows,
                          layer_physical_q=[len(rows)]*20,prefix_kv_length=63,
                          dirty_before=dirty,dirty=dirty,cold=[],mask=[],prefix=[],edited=[],
                          admitted=admitted,revoked=[],written_absolute=admitted,dirty_after=admitted,
                          tail_q=len(rows)-1,tail_opened=opened,head_mask_at_plan=1,
                          pre_state=dict(head=32,origin=63,end=66,last_end=[63,64,66,66,66][i],
                                         stopping=False)))
    q=sum(e["q"] for e in trace)
    arm=dict(F=5,Q=q,decoder_Q=q,encoder_Q=0,N=3,correct=False,failure=None,trace=trace)
    actual=unpack_idle(arm)
    assert [actual[k] for k in IDLE] == [305,0,2,0,0,1,1]
    assert actual["open_F"] == 4 and actual["open_with_idle_F"] == 2
    arm["trace"][-1]["head_mask_at_plan"]=0
    with pytest.raises(ValueError,match="MASK state"):
        unpack_idle(arm)

from copy import deepcopy

import pytest

from tools.llada_gate_row_bill import bill


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

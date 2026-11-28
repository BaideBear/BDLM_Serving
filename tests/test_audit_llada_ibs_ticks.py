from copy import deepcopy

import pytest

from tools.audit_llada_ibs_ticks import inspect


def trace():
    events=[]; calls=[]
    def step(p,masked,dirty,writes,edit=()):
        events.append(dict(absolute_positions=p,q_tokens=len(p),phase="optimus",mode="ibs",block=0,
                           physical_cap=4,offer_tick=len(events),selected_local=p,selected_masked=masked,
                           visible_kv=max(p)+1,dirty_before=dirty,written_absolute=writes,
                           edited_positions=edit,edited=len(edit),accepted=len(set(writes)-set(edit))))
        calls.append(dict(batch=1,rows=len(p),physical_rows=len(p),kv_rows=max(p)+1))
    step([0,1,2],[False,True,True],[],[1])
    step([1,2,3],[False,True,True],[1],[1,2,3],[1])
    step([1,2,3],[False,False,False],[1,2,3],[3],[3])
    step([3],[False],[3],[])
    return dict(prompt_ids=[1],events=events,calls=calls,generation_forwards=4,generation_query_rows=10)


def test_paid_warm_and_dirty_edits_are_real_ledger_counts():
    c=inspect(trace(),4)
    assert (c["cold_Q"],c["warm_Q"],c["cold_admits"],c["warm_admits"])==(3,1,2,1)
    assert (c["edits_with_cold"],c["edits_without_masks"],c["all_dirty_edits"])==(1,1,1)
    assert c["mixed_mask_F"]==c["mixed_mask_warm_admits"]==1
    assert c["all_dirty_F"]==2 and not c["cap_full_debt_F"]


def test_refuse_hidden_write_or_fabricated_edit():
    r=trace(); bad=deepcopy(r)
    bad["events"][1]["dirty_before"]=[]
    with pytest.raises(ValueError,match="write"):
        inspect(bad,4)
    r["events"][1]["edited_positions"]=[2]
    with pytest.raises(ValueError,match="write"):
        inspect(r,4)


def test_failed_case_keeps_every_paid_forward_and_terminal_debt():
    r=trace()
    r["events"].pop(); r["calls"].pop()
    r.update(generation_forwards=3,generation_query_rows=9,status="generation_failed")
    c=inspect(r,4)
    assert (c["F"],c["Q"],c["generation_failed"],c["outstanding_on_failure"])==(3,9,1,1)
    with pytest.raises(ValueError,match="unreconciled"):
        inspect(dict(r,status="success"),4)
    with pytest.raises(ValueError,match="physical forward"):
        inspect(dict(r,calls=[]),4)

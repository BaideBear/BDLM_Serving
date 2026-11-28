"""Strict handoff exposure is a census, not hypothetical forward savings."""
import importlib.util
from pathlib import Path
import pytest


TOOLS = Path(__file__).resolve().parents[1]/"tools"
CONTROL = TOOLS.parent/"experiments/t2t/serving_slots/controls/llada-flow-active-w2"


def load(name, folder, monkeypatch):
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(name, folder/(name+".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def example(edit=False):
    events = []
    def tick(block, pos, w, e, ready, mask, dirty):
        events.append(dict(block=block,phase="optimus",absolute_positions=pos,written_absolute=w,
                           edited_positions=e,cached_prefix_after=ready,selected_masked=mask,
                           dirty_before=dirty))
    for lo in (0,8,16,24):
        tick(0,list(range(lo,lo+8)),[31] if lo==24 else [],[],min(lo+8,31),
             [False]*7+[lo==24],[])
    if edit:
        tick(0,[31],[31],[31],31,[False],[31])
    tick(0,[31],[],[],32,[False],[31])
    tick(1,list(range(32,40)),[32],[],0,[True]*8,[])
    return dict(prompt_ids=[123]*31,generation_forwards=len(events),events=events)


def test_first_closing_slack_is_not_a_free_forward(monkeypatch):
    m = load("ibs_close_overlap",TOOLS,monkeypatch)
    c = m.count(example(), "llada21", 8)
    assert c["geometry_seats"] == c["positions_also_used_by_next_first_call"] == 7
    assert c["entire_next_first_call_covered"] == 0
    assert c["immediate_clean_promotion"] == c["geometry_blocks"] == 1
    changed = m.count(example(edit=True), "llada21", 8)
    assert changed["geometry_blocks"] == changed["head_edited_before_promotion"] == 1
    assert changed["promoted_no_later_edit"] == 0


def test_missing_debt_raises(monkeypatch):
    m = load("ibs_close_overlap",TOOLS,monkeypatch)
    r=example()
    r["events"][4]["dirty_before"]=[]
    with pytest.raises(ValueError,match="prior debt"):
        m.count(r,"llada21",8)


def test_sealed_head_and_eos_discard_are_different(monkeypatch):
    m = load("audit_payment",CONTROL,monkeypatch)
    B, MASK, EOS = m.B, m.MASK, m.EOS
    cold, final = [MASK]*B, [EOS]+[123]*(B-1)
    def slot(a,b,fin):
        return dict(ids=a,new=b,prompt=[False]*B,pe0=0,pe=0,fin0=False,fin=fin,frontier=1.)
    def tick(slots):
        return dict(commit=0,right=len(slots),head=0,open=len(slots),slots=slots,
                    written_KV=B*len(slots),physical_width=B*len(slots))
    r=dict(cap=2048,arm="active_w2",prompt_ids=[],output_ids=[EOS],eos=True,
           ticks=[tick([slot(cold,final,False)]),tick([slot(final,final,True),slot(cold,final,False)])])
    c=m.account(r)
    assert c["writes_M2T"]==2*B and c["paid_M2T"]==c["discarded_unpaid_M2T"]==B
    assert c["sealed_slots_with_unpaid_writes"]==c["forwards_with_seal_and_surviving_other_slot_write"]==0
    assert c["forwards_with_seal_and_discarded_other_slot_write"]==1

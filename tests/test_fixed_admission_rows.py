import pytest
from tools.fixed_admission_rows import inspect_tick, MASK


def tick(rows, ids, dirty=(), written=None, new=(), q=4, edits=()):
    changes = written or {}
    return dict(event=dict(rows=rows, dirty=sorted(set(dirty)&set(rows)), dirty_before=list(dirty),
                           dirty_after=list(set(dirty)-set(rows) | changes.keys()), admission_rows=list(new),
                           sampled=rows, prefix=[], written_absolute=list(changes), edited=list(edits), physical_q=len(rows),
                           decoder_Q=len(rows), history_Q=len(rows), layer_physical_q=[len(rows)]*20,
                           encoder_F=0, decoder_F=1, generation_F=1, offer=q,
                           admission_start=min(new) if new else None, over_offer_Q=max(0,len(rows)-q),
                           response_mode="full_block", nominal_offer=q, parent_plan_rows=rows[:q]),
                ids=ids, before=min(new) if new else 0, written=changes)


def test_disjoint_budget_and_crosscut_edit():
    c,d=inspect_tick(tick([1,2,3],[15,MASK,17],dirty=[1],written={1:145,2:120},edits=[1]),4,{1},1,2049)
    assert (c["writeback"],c["other_mask"],c["clean_decoded"],c["new_block"])==(1,1,1,0)
    assert c["t2t"]==c["t2t_on_writeback"]==1 and d=={1,2}


def test_optional_prefix_is_only_a_subset_of_clean_decoded():
    t=tick([4,5,6],[115,116,117],written={6:559},edits=[6])
    t["event"]["prefix"]=[6]
    c,_=inspect_tick(t,4,set(),0,2048)
    assert c["clean_decoded"]==3 and c["sampled_optional_prefix"]==c["t2t_on_optional_prefix"]==1


def test_initialization_is_not_regular_mask_or_an_arbitrary_overflow():
    rows=list(range(32,64))
    c,_=inspect_tick(tick(rows,[MASK]*32,new=rows),4,set(),3,2051)
    assert c["new_block"]==32 and c["other_mask"]==0 and c["over_offer_Q"]==28
    with pytest.raises(ValueError,match="new block"):
        inspect_tick(tick(rows[:8],[MASK]*8,new=rows[:8]),4,set(),3,2051)
    with pytest.raises(ValueError,match="overflow"):
        inspect_tick(tick(rows,[MASK]*32),4,set(),3,2051)
    with pytest.raises(ValueError,match="outstanding"):
        inspect_tick(tick([26],[MASK],dirty=[12]),4,{12},3,2051)

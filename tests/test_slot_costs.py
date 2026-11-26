import pytest

from experiments.t2t.serving_slots.slot_costs import events, percentile, selected_types


def test_reencode_charged_and_free_prefill_not_duplicated():
    row = dict(prefill_rows=11, physical_events=[["encoder",11],["decoder",2],["encoder",256]],
               events=[dict(stage="selected", q=2, positions=[0,4])])
    assert [(stage,q) for stage,q,*_ in events(row, "gemma")] == [("selected",2),("reencode",256)]
    row["events"].append(dict(stage="verify", q=2, positions=[0,4]))
    with pytest.raises(ValueError, match="Extra decoder"):
        list(events(row, "gemma"))


def test_only_fresh_debt_can_overflow_on_selected_round():
    plan = dict(quota=0, added_rows=[], evicted_rows=[], fresh_rows=[1,2,3], zero_mask_rows=[],
                zero_backfill_rows=[], selected_rows=[1,2,3], budget=2, overflow=1)
    event = dict(prefix_plan=plan, mandatory_fresh=3)
    assert selected_types(event,[1,2,3],3,2) == (3,0,0)
    plan.update(fresh_rows=[1,2], zero_mask_rows=[3])
    event["mandatory_fresh"] = 2
    with pytest.raises(ValueError):
        selected_types(event,[1,2,3],3,2)


def test_step_weighted_tail_percentile():
    assert percentile({4:94,32:6},95,100) == 32
    assert percentile({4:95,32:5},95,100) == 4

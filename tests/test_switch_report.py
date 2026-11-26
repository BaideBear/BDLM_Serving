import copy
import pytest

from experiments.t2t.serving_slots.switch_report import audit, gemma_stages


def fixture():
    def selected(tick, offer, fresh, masks):
        rows = sorted(fresh + masks)
        p = dict(fresh_rows=fresh, zero_mask_rows=masks, zero_backfill_rows=[], added_rows=[],
                 evicted_rows=[], block_start=32, budget=offer, quota=0, prefix_end=2,
                 eligible_prefix_rows=[], actual_width=len(rows),
                 selected_rows=rows, overflow=max(0,len(rows)-offer))
        return dict(stage="selected", phase="denoise", slot_offer=offer, slot_tick=tick,
                    mandatory_fresh=len(fresh), physical_q=len(rows), layer_physical_q=[len(rows)]*2,
                    absolute_positions=[32+i for i in rows], selected_masked_before=len(masks), prefix_plan=p)
    events = [dict(phase="prefill", stage="prefill", physical_q=32),
              dict(phase="denoise", stage="bootstrap", physical_q=32),
              selected(0, 4, [2], [3,4,5]), selected(1, 2, [3,4,5], []),
              dict(phase="commit", stage="commit", physical_q=32)]
    return dict(model="llada21", events=events, generation_forwards=4, output_tokens=8,
                generation_physical_padded_rows=71, generation_query_rows=71, q_by_phase=dict(selected=7))


def test_work_counts_mandatory_overflow_and_realized_switch():
    c = audit(fixture(), (4,2))
    assert (c["down"],c["up"],c["switched_requests"]) == (1,0,1)
    assert (c["mandatory"],c["masks"],c["overflow_rows"],c["charged_other"]) == (4,3,1,64)


@pytest.mark.parametrize("mutation", ["tick","q","mandatory","commit","overlap"])
def test_refuse_corrupted_work(mutation):
    row = copy.deepcopy(fixture())
    e = row["events"][2]
    if mutation == "tick":
        e["slot_tick"] = 3
    elif mutation == "mandatory":
        e["mandatory_fresh"] += 1
    elif mutation == "q":
        e["layer_physical_q"][-1] -= 1
    elif mutation == "commit":
        row["events"][-1]["slot_offer"] = 2
    else:
        e["prefix_plan"]["zero_backfill_rows"].append(3)
    with pytest.raises(ValueError):
        audit(row, (4,2))


def test_switch_audit_requires_registered_prefix_dose_and_boundary():
    row = fixture()
    event = row["events"][2]
    p = event["prefix_plan"]
    event["slot_prefix_mode"] = "hard_prefix_slack_right"
    p["quota"] = 1
    assert audit(row, ("4r1",2))["prefix"] == 0
    with pytest.raises(ValueError):
        audit(row, (4,2))
    p["zero_mask_rows"].remove(5)
    p["added_rows"] = p["eligible_prefix_rows"] = [5]
    event["selected_masked_before"] -= 1
    with pytest.raises(ValueError):
        audit(row, ("4r1",2))


def test_native_offer_is_billed_as_offer_not_forced_verification():
    event = dict(stage="native_offer",slot_offer=256,slot_tick=0,q=256,positions=list(range(256)),
                 mandatory_fresh=9,fresh_selected=9,available_fresh=9,overflow=0)
    row = dict(model="gemma",prefill_rows=3,events=[dict(stage="bootstrap",q=256),event],
               physical_events=[["encoder",3],["decoder",256],["decoder",256]],
               generation_forwards=2,generation_query_rows=512,generation_physical_padded_rows=512,
               q_by_phase=dict(bootstrap=256,native_offer=256),output_tokens=60)
    result = audit(row,("256n",))
    assert result["offered_rows"] == result["native_rows"] == result["charged_other"] == 256
    live=dict(row,physical_events=list(map(tuple,row["physical_events"])))
    assert audit(live,("256n",))==result
    with pytest.raises(ValueError):
        audit(row,(256,))


def test_gemma_can_finish_on_native_offer_or_verification_but_not_selection():
    def event(stage,block,finish=False):
        return dict(stage=stage,block=block,q=128 if stage=="selected" else 256,
                    admitted=2 if stage=="bootstrap" else 0,edited=int(stage=="selected"),
                    hard_strict_edits=int(stage=="selected"),revoked=0,resolved=2,
                    stop_candidate=finish or stage=="selected",finished=finish,terminal_clock=False)
    events=[event("bootstrap",4),event("selected",4),event("verify",4,True),
            event("bootstrap",260),event("native_offer",260,True)]
    counts=dict(bootstrap=2,selected=1,verify=1,reencode=1,native_offer=1)
    row=dict(events=events,physical_events=[("encoder",4),("decoder",256),("decoder",128),
             ("decoder",256),("encoder",256),("decoder",256),("decoder",256)],
             forwards_by_phase=counts,q_by_phase={k:v*(128 if k=="selected" else 256)
                                                 for k,v in counts.items()},t2t_edits=1)
    c=gemma_stages(row)
    assert (c["blocks"],c["native_closes"],c["verify_closes"],c["reencode_Q"],
            c["strict_T2T"],c["M2T"],c["selected_stop_candidates"])==(2,1,1,256,1,4,1)
    for key,value in (("finished",True),("resolved",1),("hard_strict_edits",0)):
        bad=copy.deepcopy(row)
        bad["events"][1][key]=value
        with pytest.raises(ValueError):
            gemma_stages(bad)

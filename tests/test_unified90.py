from copy import deepcopy
from types import SimpleNamespace
import pytest

from experiments.t2t.serving_slots.unified90 import ARMS, paired, work_row


def record(arm, *, exposed=False):
    first = ARMS[arm][0][0]
    step = dict(tick=0, offer=first, q=2, head=0, phase="sample", mandatory_q=1, overflow=0,
                encoded=[], promoted=[], prefix_selected=[], refresh_only=[], strict_edits=[[]],
                selected=[11, 12], sampled=[11,12], visible=[11,12] + ([267] if exposed else []),
                ages_before=[1], dirty_before=[[0]])
    obs = dict(all_projected_widths_match=True, actual_decoder_q=[2], F=1, Q=2, reencodes=0,
               decoder_layers=30, projection_modules=85, prefill=11)
    task = dict(index=0, dataset="demo", task_id="demo/0", seed=4)
    tok = SimpleNamespace(all_special_ids=[0,3], decode=lambda ids, **kw: "answer")
    cfg = SimpleNamespace(eos_token_id=[3], pad_token_id=0)
    return work_row(task, arm, [7,3], [step], obs, tok, cfg, 2048, "input")


def test_numerator_counts_delivered_tokens_not_edits_or_q():
    row = record("q512")
    assert (row["output_tokens"], row["tpf"], row["token_efficiency"]) == (1, 1., .5)
    assert row["first_opportunity"] is None


def test_spawn_readback_rejects_changed_unexposed_output_and_unpaid_projection():
    from experiments.t2t.serving_slots.unified90 import spawn_witness
    a=record("q512")
    a.update(first_earlier_spawn=None,dual_ticks=0,quote_verified_decoder_calls=1)
    b=deepcopy(a)
    assert spawn_witness(a,b) is None
    b["output_ids"]=[8,3]
    with pytest.raises(ValueError,match="No earlier admission"):
        spawn_witness(a,b)
    b["generation_physical_padded_rows"]+=1
    with pytest.raises(ValueError,match="cost/quote"):
        spawn_witness(a,b)


def test_identical_unexposed_request():
    on, off = record("replay"), record("replay_open")
    paired(on, off)
    off["output_ids"] = [12,3]
    with pytest.raises(ValueError, match="unexposed"):
        paired(on, off)


def test_after_first_visibility_divergence_is_not_filtered():
    on, off = record("replay", exposed=True), record("replay_open", exposed=True)
    assert on["first_opportunity"] == 0
    off["output_ids"] = [10,3]
    paired(on, off)
    bad = deepcopy(off)
    bad["first_geometry_sha256"] = "wrong"
    with pytest.raises(ValueError, match="geometry"):
        paired(on, bad)


def test_different_offers_are_not_a_mask_comparison():
    with pytest.raises(ValueError, match="mask-only"):
        paired(record("q512"), record("replay_open"))


def test_first_rank_change_may_not_remove_a_mask_or_dirty_row():
    from experiments.t2t.serving_slots.unified90 import check_rank_change
    a,b=record("q64_k16_random"),record("q64_k16_oldest")
    x=dict(optional=[5],selected=[5,10,12],pre=[[10]],phase="sparse",edits=[[]],
           head=0,ages=[2],prefix_end=9)
    y=dict(x,optional=[4],selected=[4,10,12])
    assert check_rank_change(a,b,[x,x],[x,y])==1
    assert check_rank_change(a,b,[x],[x]) is None
    with pytest.raises(ValueError,match="width"):
        check_rank_change(a,b,[x],[dict(y,selected=[4,10,11])])
    with pytest.raises(ValueError,match="prefix witness"):
        check_rank_change(a,b,[x],[dict(y,optional=[3])])
    with pytest.raises(ValueError,match="No rank intervention"):
        check_rank_change(a,dict(b,generation_forwards=7),[x],[x])
    with pytest.raises(ValueError,match="inputs differ"):
        check_rank_change(a,dict(b,input_sha256="other"),[x],[x])


def test_archived_rank_pair_checks_unchanged_work_and_intervention_price():
    from experiments.t2t.serving_slots.fixed_slots import rank_evidence
    a,b=record("q64_k16_random"),record("q64_k16_oldest")
    for x in (a,b):
        x["first_rank_change"]=None
    assert not rank_evidence(a,b,16)
    with pytest.raises(ValueError,match="diverged"):
        rank_evidence(a,dict(b,output_ids=[98]),16)
    for x in (a,b):
        x["first_rank_change"]=0
        x["trace"][0].update(phase="sparse",prefix=1)
    assert rank_evidence(a,b,16)
    b["trace"][0]["q"]+=1
    with pytest.raises(ValueError,match="matched"):
        rank_evidence(a,b,16)


def test_early_open_uses_new_acceptances_not_cumulative_or_correctness():
    from experiments.t2t.serving_slots.unified90 import check_early_spawn
    rows=list(range(7,263))
    def tick(k,adm):
        return dict(tick=k,head=0,q=256,offer=512,dirty_before=[[]],ages_before=[k],
                    selected=rows,sampled=rows,admitted=[adm],revoked=[[]])
    a=tick(0,list(range(96)))
    b=tick(1,list(range(96,176)))  # 80 of the remaining 160 became accepted.
    c=tick(2,[])
    both=list(range(7,519))
    d=dict(c,q=512,selected=both,sampled=both,dirty_before=[[],[]],ages_before=[2,0])
    assert check_early_spawn([a,b,c],[a,b,d],[0],[9]) == dict(tick=2,head=0,newly_accepted=80,previously_unaccepted=160)
    assert check_early_spawn([a,b,c],[a,b,c],[0],[0]) is None
    with pytest.raises(ValueError,match="thresholds"):
        x=dict(b,admitted=[list(range(96,96+79))])
        check_early_spawn([a,x,c],[a,x,d],[0],[9])
    with pytest.raises(ValueError,match="matched head"):
        check_early_spawn([a,b,c],[a,b,dict(d,dirty_before=[[2],[]])],[0],[9])
    with pytest.raises(ValueError,match="Unchanged admission"):
        check_early_spawn([a,b,c],[a,b,c],[0],[9])


def test_exchange_matches_earliest_physical_change_and_never_spends_dirty_rows():
    from experiments.t2t.serving_slots.unified90 import check_exchange
    common=dict(offer=128,tick=0,phase="sparse",head=0,visible=list(range(100,356)),
                dirty_before=[[0]],ages_before=[1],mandatory_q=1,prefix_end=10,q=5)
    a=dict(common,selected=[100,110,111,112,113],prefix_selected=[])
    b=dict(common,selected=[100,109,110,111,112],prefix_selected=[109],displaced_mask=[113])
    assert check_exchange([a],[b],[0],[1])==(0,1)
    with pytest.raises(ValueError,match="exchange intervention"):
        check_exchange([a],[dict(b,displaced_mask=[100])],[0],[1])
    with pytest.raises(ValueError,match="one-for-one"):
        check_exchange([a],[dict(b,q=6)],[0],[1])
    with pytest.raises(ValueError,match="without exchanging"):
        check_exchange([a],[dict(a,displaced_mask=[])],[0],[1])


def test_archive_exchange_requires_measured_width_and_counts():
    from experiments.t2t.serving_slots.fixed_slots import exchange_evidence
    base=record("q512")
    t=dict(base["trace"][0], offer=128, q=8, mandatory_q=1, prefix=1, phase="sparse")
    a=dict(base,trace=[t],oldest_reproduced=True,quote_verified_decoder_calls=1,
           exchange_by_tick=[0],total_exchanged_rows=0,first_exchange_tick=0)
    b=dict(a,trace=[dict(t,prefix=2)],exchange_by_tick=[1],total_exchanged_rows=1)
    assert exchange_evidence(a,b)
    with pytest.raises(ValueError,match="equal-width"):
        exchange_evidence(a,dict(b, trace=[dict(t, q=9,prefix=2)]))
    with pytest.raises(ValueError,match="count mismatch"):
        exchange_evidence(a,dict(b,total_exchanged_rows=2))
    with pytest.raises(ValueError,match="paid unresolved"):
        exchange_evidence(a,dict(b,trace=[dict(t,prefix=3)], exchange_by_tick=[2], total_exchanged_rows=2))

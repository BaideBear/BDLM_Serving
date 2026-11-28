from dataclasses import replace

from bdlm_serving.methods.t2t.budget_family import Plan
from experiments.t2t.serving_slots.llada_frontier_seat import exchange, interior_exchange
from experiments.t2t.serving_slots.llada_selection_geometry import policy
from experiments.t2t.serving_slots.llada_window_geometry import window
from experiments.t2t.serving_slots.llada_successor_seats import extra_rows


def test_exchange_preserves_width_dirty_sampler_arity_and_contiguous_build():
    p=Plan((4,9,10,11),(4,),(9,10,11),(),(),False)
    x,s,change=exchange(p,p.cold,{8,7})
    assert change==(11,7)
    assert x==Plan((4,7,9,10),(4,),(9,10),(7,),(),False)
    assert s==(7,9,10)


def test_exchange_never_trades_the_only_cold_row_or_a_debt_row():
    p=Plan((4,9),(4,),(9,),(),(),False)
    assert exchange(p,p.cold,{7})==(p,p.cold,None)
    p=replace(p,rows=(4,9,10),cold=(9,10))
    for warm in ({4},{11},set()):
        assert exchange(p,p.cold,warm)==(p,p.cold,None)


def test_interior_exchange_only_replaces_a_measured_mask():
    p=Plan((2,4,6,8),(2,),(),(6,8),(4,),False)
    x,s,change=interior_exchange(p,p.rows,{7,9})
    assert x==Plan((2,4,6,7),(2,),(),(6,),(4,),False)
    assert s==(2,4,6) and change==(8,7)
    p=replace(p,mask=(6,))
    assert interior_exchange(p,p.rows,{7})==(p,p.rows,None)


def test_ibs_candidate_keeps_settled_suffix_rows_then_clips_for_debt():
    masked={2,4,5,6}
    assert policy(2,8,masked,{7},4)==((2,3,4,7),(2,3,4,7))
    assert policy(2,8,masked,{7},2)==((2,3,7),(2,7))


def test_window_preserves_pinned_rows_and_width_but_retries_nearby_mask():
    p=Plan((2,5,10,11),(5,),(10,11),(),(2,),False)
    new,sampled,delta=window(p,(10,11),{4,7,8,9,10,11},set(range(10)),12)
    assert new==Plan((2,4,5,6),(5,),(),(4,),(2,),False)
    assert sampled==new.rows and delta["interior"]==(6,)


def test_window_does_not_advance_a_drain_or_change_an_identical_cold_plan():
    p=Plan((2,3),(2,3),(),(),(),True)
    assert window(p,(),{7,8},set(range(7)),9)==(p,(),None)
    p=Plan((4,5),(),(4,5),(),(),False)
    assert window(p,p.cold,set(range(4,9)),set(range(4)),9)==(p,p.cold,None)


def test_window_keeps_contiguous_new_keys_if_cold_and_warm_mix():
    p=Plan((4,9,10,11),(4,),(9,10,11),(),(),False)
    mask=set(range(5,12))
    new,sampled,_=window(p,p.cold,mask,set(range(9)),12)
    assert new.rows==(4,5,6,7) and sampled==new.rows
    p=Plan((4,8,9,10),(4,),(8,9,10),(),(),False)
    new,sampled,_=window(p,p.cold,{7,8,9,10},set(range(8)),11)
    assert new.rows==(4,7,8,9) and new.cold==sampled==(8,9)


def test_successor_only_revisits_an_already_built_unfinished_tail():
    p=Plan((2,31,32,34),(32,),(),(34,),(2,),False)
    masks={31,34}
    assert extra_rows(p,p.rows,32,36,masks,set(range(36)))==(33,35)
    assert not extra_rows(p,p.rows,32,36,{31},set(range(36)))
    assert not extra_rows(p,p.rows,32,36,masks,set(range(35)))
    assert not extra_rows(p,(),32,36,masks,set(range(36)))
    assert not extra_rows(replace(p,cold=(34,)),(34,),32,36,masks,set(range(36)))


def test_window_extension_requires_registered_geometry_and_same_gpu():
    import pytest
    from experiments.t2t.serving_slots.llada_frontier_seat import qualify_extension
    params=dict(protocol="llada-fixed-width-window-v1",device="GPU-test",q=16,
                checkpoint="ckpt",samples="s",scorer="c",geometry_sha256=
                "258c9a8be716798a0728e2dc0adbd1b1bc159d83b3a16eca13db63774040fde1")
    old=dict(params,order=list(range(30)),planner=
             "5a986abee4a9461c07c4811014d978ece7fa6575a48fbe26c015e88d9fbf1254")
    new=dict(params,order=list(range(90)),planner="new")
    assert qualify_extension("window",old,new)=="window"
    for k in ("geometry_sha256","device","q","samples"):
        with pytest.raises(ValueError):
            qualify_extension("window",old,dict(new,**{k:"changed"}))


def test_paid_extension_requires_same_q_device_checkpoint_samples_scorer():
    import pytest
    from experiments.t2t.serving_slots.llada_frontier_seat import qualify_extension
    params=dict(protocol="llada-window-paid-mask-v1",device="GPU-test",q=16,
                checkpoint="ckpt",samples="s",scorer="c",
                geometry_sha256="g",sampling_sha256="p")
    old=dict(params,order=list(range(30)))
    new=dict(params,order=list(range(90)))
    assert qualify_extension("paid",old,new)=="paid"
    for k in ("protocol","q","device","checkpoint","samples","scorer",
              "geometry_sha256","sampling_sha256"):
        with pytest.raises(ValueError):
            qualify_extension("paid",old,dict(new,**{k:"changed"}))
    with pytest.raises(ValueError):
        qualify_extension("paid",dict(old,order=list(range(1,31))),new)


def test_paid_warm_mask_does_not_sample_prefix_dirty_or_insert_new_queries():
    from experiments.t2t.serving_slots.llada_window_sample import extra_mask
    p=Plan((2,3,4,5,6),(3,),(6,),(4,5),(2,),False)
    assert extra_mask(p,p.cold)==(4,5)
    assert extra_mask(replace(p,drain_only=True),p.cold)==()
    assert extra_mask(p,p.rows)==()
    assert extra_mask(replace(p,cold=()),())==()

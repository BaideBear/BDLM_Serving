from dataclasses import replace

import pytest

from bdlm_serving.methods.t2t.budget_family import Plan
from experiments.t2t.serving_slots.llada_contested import analyze, exchange


def test_only_saturated_clean_prefix_competes_with_nearest_waiting_mask():
    p = Plan((0,2,3,4), (2,), (), (4,), (0,3), False)
    masks, resident = {4,5,6,7}, set(range(7))
    a, s, pair = exchange(p,p.rows,masks,resident,4)
    assert s == a.rows == (2,3,4,5) and pair == (0,5)
    assert a.prefix == (3,) and a.mask == (4,5) and a.dirty == (2,)
    assert exchange(p,p.rows,masks,resident,8)[2] is None
    assert exchange(p,p.rows,masks,{0,2,3,4},4)[2] is None
    assert exchange(p,(4,),masks,resident,4)[2] is None
    assert exchange(replace(p,cold=(4,)),p.rows,masks,resident,4)[2] is None


def test_receipt_reconstructs_prefix_and_cannot_skip_nearer_waiting_mask():
    st=dict(origin=0,head=0,last_end=8,end=32)
    def ev(rows,**extra):
        return dict(pre_state=st,rows=rows,sampled=rows,q=len(rows),
                    layer_physical_q=[len(rows)]*20,sample_clock=1,
                    visible_kv=8,phase="sample",cold=[],dirty=[],revoked=[],**extra)
    init=ev([0,1],prefix=[],mask=[0,1],admitted=[0,1],proposals=[[99],[99]])
    ctrl=ev([1,2],prefix=[1],mask=[2],admitted=[],proposals=[[27],[99]])
    def trial(take):
        return ev([2,take],prefix=[],mask=[2,take],admitted=[take],
                  proposals=[[99],[99]],contest=(1,take))
    def audit(take):
        return analyze(dict(trace=[init,ctrl]),dict(trace=[init,trial(take)]),2,99)
    assert audit(3)["mask_admits"] == 1 and audit(3)["extra_Q"] == 0
    with pytest.raises(ValueError,match="same-width"):
        audit(4)


def test_saturated_interior_swap_keeps_prefix_dirty_and_nearest_interior():
    plan = Plan((1,2,3,4,5), (4,), (), (2,), (1,), False)
    masks, resident = {2,6,7}, set(range(7))
    new,sampled,pair = exchange(plan,plan.rows,masks,resident,5,interior=True)
    assert pair == (5,6) and sampled == new.rows == (1,2,3,4,6)
    assert new.dirty == (4,) and new.prefix == (1,) and new.mask == (2,6)
    assert exchange(plan, plan.rows, masks, resident, 8, interior=True)[2] is None
    assert exchange(replace(plan,dirty=(3,4,5)),plan.rows,masks,resident,5,interior=True)[2] is None


def test_interior_receipt_replays_actual_frontier_and_keeps_supplemental_width_zero():
    st = dict(origin=0,head=0,last_end=8,end=32)
    def event(rows,prev,admitted,masks,**tags):
        return dict(rows=rows,sampled=rows,proposals=[[p] for p in prev],admitted=admitted,mask=masks,
                    pre_state=st,q=len(rows),layer_physical_q=[len(rows)]*20,visible_kv=8,
                    sample_clock=1,phase="sample",cold=[],dirty=[],prefix=[],revoked=[],**tags)
    init=event([0,2,3],[99]*3,[0,2,3],[0,2,3])
    ctrl=event([1,2,3],[99,77,88],[],[1])
    swap=event([1,2,4],[99,77,99],[4],[1,4],contest=(3,4))
    def audit(candidate):
        return analyze({"trace":[init,ctrl]},{"trace":[init,candidate]},3,99,interior=True)
    assert audit(swap)["swapped_seats"] == 1 and audit(swap)["extra_Q"] == 0
    with pytest.raises(ValueError, match="same-width"):
        audit(event([1,3,4],[99,88,99],[],[1,4],contest=(2,4)))

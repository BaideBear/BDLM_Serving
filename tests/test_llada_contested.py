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

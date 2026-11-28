from types import SimpleNamespace

import pytest

from experiments.t2t.serving_slots.llada_window_fallback import compare_first, decision, first_waiting


def event(cold, warm, wc, cc, *, waiting=True):
    return dict(sampled=[cold,warm],cold=[cold],paid_waiting_mask=[warm] if waiting else [],
                proposals=[(99,1,cc,cc>.5,cc>.9),(99,2,wc,wc>.5,wc>.9)])


def test_only_fallback_changes_not_high_confidence_admissions():
    e=event(5,4,.70,.55)
    assert decision(e,99,False)==[4] and decision(e,99,True)==[5]
    for wc,cc,expected in ((.94,.60,[4]),(.96,.95,[4,5]),(.53,.55,[5])):
        e=event(5,4,wc,cc)
        assert decision(e,99,False)==decision(e,99,True)==expected
    e=event(5,4,.70,.55,waiting=False)
    assert decision(e,99,True)==decision(e,99,False)==[4]


def test_head_frontier_must_really_be_in_the_paid_sampler():
    p=SimpleNamespace(cold=(5,))
    class Paid:
        def _cross_plan(self,q):
            self.extra_window_mask=(2,4)
            return p,(2,4,5),"sample",8
        def _inputs(self,b):
            return self.masks,(),None
    req=first_waiting(Paid)()
    req.masks={2,4,5}
    assert req._cross_plan(8)[1]==(2,5)
    req.masks={1,2,4,5}
    assert req._cross_plan(8)[1]==(5,)


def test_paid_frontier_rejects_skipping_true_leftmost_waiting_row():
    common=dict(pre_state=dict(origin=2,head=0,end=32),rows=[2,4,5],cold=[5],q=3,mask=[2,4],
                dirty=[],prefix=[],sample_clock=1,phase="sample",visible_kv=6,layer_physical_q=[3]*20,
                revoked=[])
    def tick(seats):
        candidates=seats+[5]
        return dict(common,paid_waiting_mask=seats,sampled=candidates,
                    proposals=[(99,1,.95,True,True)]*len(candidates),admitted=candidates)
    a={"trace":[tick([2,4])]}
    _,s=compare_first(a,{"trace":[tick([2])]},99)
    assert s==dict(omitted=1,retained=1,accepted=1)
    with pytest.raises(ValueError,match="wrong paid-frontier"):
        compare_first(a,{"trace":[tick([4])]},99)

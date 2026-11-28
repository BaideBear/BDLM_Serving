import unittest
from types import SimpleNamespace as NS

from bdlm_serving.methods.t2t.budget_family import Law, choose
from experiments.t2t.serving_slots.gemma_eos_prefix_dose import arm_class, classify, first_change


class Base:
    def __init__(self, mask):
        self.fixed_q, self.early, self.tick, self.top_gate = 64, True, 0, "off"
        self.fuse_first_verify, self.ablation, self.prefix_order, self.reuse_paid_masks = True,"00","right",False
        self.origin, self.tail, self.rebuild, self.law = 10,None,False,Law()
        self.rows=NS(accepted=set(range(256))-mask,initialized=set(range(256)),dirty={0})

    def _apply(self, states, pos, sample, prefix, cold, offer, route):
        return dict(positions=list(pos), decoder_Q=len(pos),
                    parts=[dict(dirty_before=[0],prefix=[x-10 for x in prefix],rows=[x-10 for x in pos])])


def exercise(mask):
    notes=[]
    for d in (8,4):
        tape=[]
        cls=arm_class(Base,d,tape)
        obj=cls(mask)
        h=NS(ordinal=0,rows=obj.rows,sampler=NS(age=4))
        plan=choose(64,256,{0},mask,list(reversed(range(1,min(mask)))),law=Law(1,d))
        pos=tuple(10+x for x in plan.rows)
        obj._apply([h],pos,set(pos),tuple(10+x for x in plan.prefix),set(),64,"sparse")
        notes.append(tape)
    return notes


class TestDose(unittest.TestCase):
    def test_mask_trade(self):
        a,b=exercise(set(range(64,256)))
        self.assertEqual((b[0]["more_prefix"],b[0]["fewer_masks"],b[0]["extra_width"]),(8,8,0))
        self.assertIsNotNone(first_change({"tape":["x"]},{"tape":["y"]},[a,b]))

    def test_slack_fill(self):
        _,b=exercise({250})
        self.assertEqual((b[0]["more_prefix"],b[0]["fewer_masks"],b[0]["extra_width"]),(8,0,8))

    def test_verify_dirty_once(self):
        self.assertEqual(classify([0,1],{1},{1},set(),"verify_paid"),
                         dict(dirty=1, mask=0,prefix=0,verify_clean=1))

    def test_unpaid(self):
        with self.assertRaises(ValueError):
            classify([0],{1},{0},set(),"drain")

    def test_double_class(self):
        with self.assertRaises(ValueError):
            classify([0],set(),{0},{0},"sparse")

    def test_unexplained(self):
        with self.assertRaises(ValueError):
            first_change({"tape":["x"]},{"tape":["y"]},[[{"more_prefix":0}]]*2)

import unittest
from types import SimpleNamespace as NS

from experiments.t2t.serving_slots.gemma_predicate_only import compare, first_opportunity, native_canvas


class TestPredicate(unittest.TestCase):
    def test_native_or_limit_only(self):
        c = native_canvas(NS)(rows=NS(_native_stop=False,age=8,max_steps=48,accepted=set(range(256))))
        self.assertTrue(c.can_sample)
        c.rows._native_stop = True
        self.assertFalse(c.can_sample)
        c.rows._native_stop, c.rows.age = False, 48
        self.assertFalse(c.can_sample)

    def test_qualified_opportunity(self):
        tick=dict(route="dense_single",accepted=256,hit=False,age=8,ordinal=1)
        a=dict(init=[{},{}], ticks=[{},{}])
        self.assertEqual(compare([tick,dict(tick,route="verify_paid")],[tick,dict(tick,age=9)],
                                 a,a,1,2),0)
        for change in (dict(age=48),dict(hit=True),dict(accepted=255),dict(route="verify")):
            self.assertIsNone(first_opportunity([dict(tick,**change)]))
        with self.assertRaises(ValueError):
            compare([],[],a,a,1,2)

    def test_match_actual_prebranch_rng_and_epoch(self):
        t=dict(route="dense_single",accepted=256,hit=False,age=8,ordinal=0)
        off=[t,dict(t,route="verify_paid")]
        on=[t,dict(t,age=9)]
        a=dict(init=["a"],ticks=[{"rng":"s"}])
        for b in (dict(init=["b"],ticks=a["ticks"]),dict(init=["a"],ticks=[{"rng":"t"}])):
            with self.assertRaises(ValueError):
                compare(off,on,a,b,1,2)
        with self.assertRaises(ValueError):
            compare(off,[t,dict(on[1],ordinal=1)],a,a,1,2)

    def test_unexposed_whole_row_not_score_alone(self):
        t=dict(route="dense_single",accepted=256,hit=True,age=8,ordinal=0)
        ticks=[t,dict(t,route="verify")]
        a=dict(init=[3],ticks=[{},{}])
        self.assertIsNone(compare(ticks,ticks,a,a,{"ids":[4]},{"ids":[4]}))
        with self.assertRaises(ValueError):
            compare(ticks,ticks,a,a,{"ids":[4],"correct":True},{"ids":[5],"correct":True})

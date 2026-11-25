import unittest
from copy import deepcopy

from bdlm_serving.methods.t2t.exchange import choose, position_keys
from bdlm_serving.methods.t2t.protocol import Arm
from tools.exchange_study import audit_pair


class ExchangeTest(unittest.TestCase):
    rates = {("old",):.2, ("mask",):.5}

    def test_relative_price_and_immune_first_mask(self):
        args = (list(range(6)),[False,True,False,False,True,True],{1},set(),[-1]*6,2,4,2,self.rates)
        self.assertEqual(choose(*args,1.),([0,1,2,3],[],[(4,3,False)]))
        self.assertEqual(choose(*args,4.),([0,1,4,5],[4,5],[(4,3,True),(5,2,True)]))

    def test_old_competes_with_old_at_unweighted_rate(self):
        args = (list(range(6)),[False,True,True,True,True,True],{1},{1},[-1,1,1,1,1,1],2)
        keys = position_keys(*args)
        rates = dict(self.rates)
        rates[keys[4]] = .95
        selected,reserved,offers = choose(*args,3,2,rates,4.)
        self.assertEqual(([0,1,4],[4],[(4,2,True)]),(selected,reserved,offers))
        self.assertEqual(choose(*args,3,2,self.rates,4.)[0],[0,1,2])

    def test_fresh_overflow_and_saturation(self):
        args = (list(range(6)),[True,True,True,False,True,True],{0,1,2},set(),[-1]*6,2)
        self.assertEqual(choose(*args,2,1,self.rates,4.),([0,1,2],[],[]))
        self.assertEqual(choose(*args,6,1,self.rates,4.),(list(range(6)),[],[]))

    def test_b32_only(self):
        self.assertEqual(Arm.parse("16:0:14:hard_exchange_equal",32).edit,2)
        with self.assertRaises(ValueError):
            Arm.parse("16:0:14:hard_exchange_equal",256)

    def test_first_physical_exchange_and_noop(self):
        event = dict(block=0,context_version=0,step_in_block=2,hard_clock=3,mandatory_fresh=1,
                     masked_before=4,actual_q=3,physical_q=3,layer_physical_q=[3,3],F=1,prefix_kv_length=0,
                     stage="selected",absolute_positions=[0,1,2],reserved_extra_edit_positions=[],
                     reserved_strict_edit_positions=[],selected_masked_before=2,accepted=1,hard_strict_edits=0)
        base = dict(events=[{"stage":"bootstrap"},event], index=0, model="llada21",seed=2, task_id="x",
                    dataset="toy",prefill_rows=0,status="success",output_ids=[1],output_tokens=1,
                    generation_forwards=2,generation_query_rows=35)
        self.assertEqual(audit_pair(base,base)["victim"],"none")
        new = deepcopy(base)
        new["events"][-1].update(absolute_positions=[0,1,3],reserved_extra_edit_positions=[3],
                                 selected_masked_before=1)
        self.assertEqual(audit_pair(new,base)["victim"],"mask")
        new["events"][-1]["reserved_extra_edit_positions"] = [1]
        with self.assertRaisesRegex(ValueError,"physical rows"):
            audit_pair(new,base)
        new = deepcopy(base)
        new["output_ids"] = [9]
        with self.assertRaisesRegex(ValueError,"No-swap"):
            audit_pair(new,base)

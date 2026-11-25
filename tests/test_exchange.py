import unittest
from copy import deepcopy

from bdlm_serving.methods.t2t.exchange import choose, position_keys, randomize_candidate
from bdlm_serving.methods.t2t.protocol import Arm
from tools.exchange_study import audit_pair, propensity_counts


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

    def test_randomize_candidates_only(self):
        valid = range(9)
        resolved = [False, True, False, False, True, True, True, True, True]
        ranked = choose(valid, resolved, {1}, set(), [-1]*9, 2, 4, 2, self.rates, 4.)
        observed = set()
        for seed in range(100):
            picked, reserved, doc = randomize_candidate(valid, resolved, {1}, *ranked, seed)
            self.assertEqual(doc["base"], [0, 1, 2, 3])
            self.assertEqual(doc["masked_victims"], [2, 3])
            self.assertEqual(doc["pool"], [4, 5, 6, 7, 8])
            self.assertEqual(set(picked), {0, 1} | set(reserved))
            self.assertEqual(len(picked), 4)
            observed.add(tuple(reserved))
        self.assertEqual(len(observed), 10)
        self.assertIn(tuple(ranked[1]), observed)  # collisions must be retained
        self.assertEqual(randomize_candidate(valid, resolved, {1}, [0, 1], [], [], 2),
                         ([0, 1], [], None))

    def test_propensity_weights_only_observed_rank_matches(self):
        from random import Random
        pool = [5,6,7]
        for seed in range(15):
            chosen, = Random(f"exchange-placebo:a:{seed}:2:7").sample(pool,1)
            event = dict(block=2,hard_clock=7,reserved_extra_edit_positions=[64+chosen],
                         reserved_strict_edit_positions=[64+chosen],placebo_exchange=dict(
                             pool=pool,base=[1,2],ranked=[1,6],executed=[1,chosen]))
            counts = propensity_counts(event,seed,"a")
            self.assertEqual(counts["weighted_hits"],3 if chosen==6 else 0)
            self.assertEqual(counts["uniform_hits"],1)
            event["reserved_extra_edit_positions"] = []
            with self.assertRaises(ValueError):
                propensity_counts(event,seed,"a")

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

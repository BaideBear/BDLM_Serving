import unittest

from bdlm_serving.methods.t2t.exchange import choose, position_keys
from bdlm_serving.methods.t2t.protocol import Arm


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

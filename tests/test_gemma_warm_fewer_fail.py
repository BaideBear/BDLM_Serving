import unittest
from experiments.t2t.serving_slots.gemma_warm_fewer_fail import advance, choose


class TestWarmSeat(unittest.TestCase):
    def test_counts_only_unsuccessful_samples(self):
        f=[0]*256
        advance(f,[0,1,2],{1,3})
        advance(f,[],set())
        advance(f,[2],set())
        self.assertEqual(f[:4],[1,0,2,0])

    def test_protection_and_strict_order(self):
        f=[3]*256
        rows=set(range(32))
        debt={31}
        acc={2,30,32}
        prefix={30}
        self.assertEqual(choose(rows,debt,acc,prefix,f),(True,None))
        f[32:35]=[0,1,1]
        self.assertEqual(choose(rows,debt,acc,prefix,f),(True,(29,33)))
        self.assertEqual(choose(rows,rows-{0},set(),set(),f),(False,None))

    def test_missing_debt_is_rejected(self):
        with self.assertRaises(ValueError):
            choose(set(range(32)),{33},set(),set(),[0]*256)

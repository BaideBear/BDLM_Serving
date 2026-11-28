import unittest

from bdlm_serving.methods.t2t.paid_agreement import EvidenceLedger
from tools.llada_ibs_eos_q16_agreement import debt, observe
from tools.llada_ibs_eos_q16_eval import independent_eligible


class PaidAgreement(unittest.TestCase):
    def case(self, apply):
        ledger, tape = EvidenceLedger(), []
        excluded = {99}

        def tick(k, *, block=1, positions=(35,36,37,38,39), native=(False,)*5):
            return observe(ledger,tape,excluded,apply,block=block,tick=k,positions=positions,
                           proposals=(14,99,15,16,17)[:len(positions)],
                           probabilities=(.6,.99,.5,.8,.9)[:len(positions)],native=native)
        self.assertEqual(tick(0), ())
        self.assertEqual(tick(1,native=(False,False,False,True,False)), (35,39) if apply else ())
        self.assertEqual(tick(2,positions=(),native=()), ())
        self.assertEqual(tick(3), ())
        self.assertEqual(tick(4,block=2), ())
        self.assertEqual([p for _,p in independent_eligible(tape,excluded)], [(),(35,39),(),()])
        self.assertEqual([x["tick"] for x in tape], [0,1,3,4])

    def test_noop(self):
        self.case(False)

    def test_apply(self):
        self.case(True)

    def test_suffix_debt_larger_than_q4_still_real(self):
        owed = set(range(40,52))
        self.assertEqual(debt(owed,start=32,ready=8,initialized=set(range(52))), sorted(owed))
        with self.assertRaises(ValueError):
            debt(owed|{38},start=32,ready=8,initialized=set(range(52)))


if __name__ == "__main__":
    unittest.main()

import unittest
from adapter import current_observe
from bdlm_serving.methods.t2t.paid_agreement import EvidenceLedger


class Dose(unittest.TestCase):
    def test_current_tie_exclusions_and_expiry(self):
        ledger=EvidenceLedger()
        tape=[]
        def step(t,probs,ids,native=(False,)*6,block=2):
            return current_observe(ledger,tape,{99},True,block=block,tick=t,
                positions=[11,12,13,14,15,16],probabilities=probs,proposals=ids,native=native)
        self.assertEqual(step(4,[.6,.4,.4,.5,.3,.2],[1,2,3,4,5,6]),())
        self.assertEqual(step(5,[.65,.85,.85,.5,.99,.96],[1,2,3,4,99,6],
                              (False,False,False,False,False,True)),(12,))
        self.assertEqual(tape[-1]["eligible"],(11,))
        self.assertEqual(step(7,[.9]*6,list(range(1,7))),())
        self.assertEqual(step(8,[.9]*6,list(range(1,7)),block=3),())

import unittest
from experiments.t2t.serving_slots.gemma_warm_accepted_pay import ANCHORS,FULL,align,eligible,gates


class TestPayment(unittest.TestCase):
    def test_only_resolved_debt(self):
        pos={1,2,3,4}
        self.assertEqual(eligible("sparse",FULL,pos,pos,{1,2},{1,3},{3}),{1})

    def test_leave_cold_close_and_pure_paid_alone(self):
        pos={1,2}
        for phase,init,sample,acc in (("sparse",{1},pos,{1}),("verify",FULL,set(),{1}),
                                     ("drain",FULL,set(),{1}),("sparse",FULL,pos,pos)):
            self.assertEqual(eligible(phase,init,pos,sample,{1},acc,set()),set())

    def test_reject_unpaid(self):
        with self.assertRaises(ValueError):
            eligible("sparse",FULL,{1,2},{1,2},{1,3},{1},set())

    def test_alignment_requires_identical_paid_logits(self):
        u = dict(skip=[7], logits="1"*64)
        prev = dict(first=None)
        off = [prev,dict(first=u,sampled=32,q=32)]
        on = [prev,dict(first=u,sampled=31,q=32)]
        x,y = dict(tape=["common","off"]),dict(tape=["common","on"])
        self.assertEqual(align(x,y,off,on),u)
        on[1]["first"] = dict(u,logits="2"*64)
        with self.assertRaises(ValueError):
            align(x,y,off,on)

    def test_unexposed_arms_require_entire_receipt_identity(self):
        tape=[dict(first=None)]
        x=dict(tape=["same"],F=1,N=9)
        self.assertIsNone(align(x,dict(x),tape,tape))
        with self.assertRaises(ValueError):
            align(x,dict(x,N=8),tape,tape)

    def test_gate_does_not_mix_criteria(self):
        x=dict(F=10,N=10,decoder_Q=32,correct=True)
        rows=[dict(index=i,first={"tick":1},off=x,on=dict(x,F=9)) for i in ANCHORS]
        self.assertTrue(all(gates(rows).values()))
        rows[-1]["on"]["correct"]=False
        self.assertFalse(gates(rows)["quality"])
        for r in rows[3:]:
            r["first"]=None
        self.assertFalse(gates(rows)["new_exposure"])

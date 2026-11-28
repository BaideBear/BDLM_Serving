import unittest
from experiments.t2t.serving_slots.gemma_cold_frontier_last import B,TIERS,audit,first_visits


class TestColdFrontierLast(unittest.TestCase):
    def test_initial_tiles(self):
        for q in TIERS:
            self.assertEqual(first_visits([],[],q,False),tuple(range(q)))
            self.assertEqual(first_visits([],[],q,True),tuple(range(q,2*q)))

    def test_paid_capacity(self):
        for q in TIERS:
            s=set(range(q,2*q))
            self.assertEqual(first_visits(s,s,q,True),())
            self.assertEqual(len(first_visits(s,{q},q,True)),q-1)

    def test_wrap(self):
        q=32
        self.assertEqual(first_visits(set(range(q,250)),set(),q,True),
                         (*range(26),*range(250,256)))

    def test_bad_debt(self):
        with self.assertRaises(ValueError):
            first_visits([], {0}, 64, True)

    def test_every_row_once(self):
        for q in TIERS:
            for enabled in (True,False):
                seen=set()
                for _ in range(B//q):
                    c=first_visits(seen,set(),q,enabled)
                    self.assertFalse(seen.intersection(c))
                    seen.update(c)
                self.assertEqual(len(seen),B)

    def test_bad_coverage(self):
        e=dict(tick=0,head=0,phase="warm",width=1,sample=1,age=0,dirty=0,writes=0)
        with self.assertRaises(ValueError):
            audit(dict(tape=["x"],F=1,decoder_F=1,encoder_F=0,encoder_Q=0),[e],32,True)

    def _one_head(self):
        q=64
        tape=[]
        age=0
        for start in (64,128,192,0):
            tape.append(dict(tick=len(tape),head=0,phase="cold",width=q,sample=q,age=age,
                             dirty=0,writes=q,prefix=0,cold=list(range(start,start+q))))
            age+=1
            tape.append(dict(tick=len(tape),head=0,phase="cold" if start else "closing_drain",
                             width=q,sample=0,age=age,dirty=q,writes=0,prefix=0,cold=[],first_tile_masks=13))
        tape.append(dict(tick=len(tape),head=0,phase="verify",width=q,sample=0,age=age,
                         dirty=0,writes=0,prefix=0,cold=[]))
        x=dict(tape=[None]*len(tape),F=len(tape),decoder_F=len(tape),decoder_Q=q*len(tape),
               encoder_F=0,encoder_Q=0)
        return x,tape

    def test_cold_to_close_and_full_payment(self):
        x,t=self._one_head()
        r=audit(x,t,64,True)
        self.assertEqual((r["cold_F"],r["cold_sample_F"],r["closing_drain_F"],r["warm_F"]), (7,4,1,0))
        self.assertEqual((r["cold_Q_first"],r["dirty_Q"],r["writes"],r["warm_entry_frontier_masks"]),
                         (B,B,B,13))
        t[1]["dirty"]-=1
        with self.assertRaises(ValueError):
            audit(x,t,64,True)

    def test_extra_clock_and_wrong_permutation(self):
        x,t=self._one_head()
        t[2]["age"]+=1
        with self.assertRaises(ValueError):
            audit(x,t,64,True)
        t[2]["age"]-=1
        t[0]["cold"],t[2]["cold"]=t[2]["cold"],t[0]["cold"]
        with self.assertRaises(ValueError):
            audit(x,t,64,True)

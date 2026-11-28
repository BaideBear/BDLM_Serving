import unittest
from types import SimpleNamespace

import torch
from bdlm_serving.methods.t2t.budget_family import Plan
from experiments.t2t.serving_slots.llada_response_executor import _Admission, admission_start


class Parent:
    def _switch_plan(self,q):
        return self.plan,self.sample,"sample",q


class Probe(_Admission,Parent):
    def __init__(self,start,end,rows,cold,dirty=()):
        self.origin=11
        self.end=end
        self.x=torch.full((1,end),99)
        self.mask_id=99
        self.bank=SimpleNamespace(initialized=set(range(start)),last_end=start)
        self._response_mode="full_block"
        self.dirty=set(dirty)
        self.stopping=False
        self.plan=Plan(tuple(rows),tuple(dirty),tuple(cold),(),(),False)
        self.sample=tuple(cold)


class ResponseTests(unittest.TestCase):
    def test_bootstrap_and_partially_resident_do_not_trigger(self):
        self.assertIsNone(admission_start(11,100,11,(11,12)))
        self.assertIsNone(admission_start(11,100,40,(40,41)))
        self.assertIsNone(admission_start(32,100,32,(32,33)))
        self.assertIsNone(admission_start(11,100,63,()))

    def test_promoted_head_real_extension_and_sample(self):
        r=Probe(32,111,range(32,48),range(32,48))
        p,s,phase,cap=r._switch_plan(16)
        self.assertEqual((p.rows,p.cold,s),(tuple(range(32,64)),)*3)
        self.assertEqual((phase,cap),("full_block_admission",32))

    def test_partial_last_and_existing_debt(self):
        r=Probe(64,69,[50,64,65],[64,65],[50])
        p,s,_,cap=r._switch_plan(4)
        self.assertEqual(p.rows,(50,64,65,66,67,68))
        self.assertEqual((s,cap),((64,65,66,67,68),6))
        self.assertEqual(p.dirty,(50,))

    def test_2b_preserves_old_seats_and_stays_in_budget(self):
        r=Probe(64,96,range(32,96),range(64,96),[34])
        r.sample=(39,*r.sample)
        p,s,phase,cap=r._switch_plan(64)
        self.assertEqual((p.rows,s,phase,cap),(r.plan.rows,r.sample,"full_block_admission",64))

    def test_no_fallback_on_illegal_build(self):
        r=Probe(32,96,[32,33],[32,33])
        r.x[0,45]=11
        with self.assertRaises(RuntimeError):r._switch_plan(16)
        with self.assertRaises(RuntimeError):admission_start(11,100,31,(32,))

    def test_strict_stays_strict(self):
        r=Probe(32,111,range(32,48),range(32,48))
        r._response_mode="strict"
        self.assertEqual(r._switch_plan(16),(r.plan,r.sample,"sample",16))

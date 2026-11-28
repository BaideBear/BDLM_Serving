import unittest
from collections import Counter
from copy import deepcopy

from experiments.t2t.serving_slots.gemma_first_tile_kv import FRONT, audit, build_only
from experiments.t2t.serving_slots.gemma_first_tile_kv_eval import matched
from experiments.t2t.serving_slots.gemma_first_tile_kv_report import bill


class TestFirstTile(unittest.TestCase):
    def test_once(self):
        cold=dict(seen=set(),sc=set(),accepted=set(),dirty=set(),age=0,positions=FRONT,sample=FRONT,route="sparse")
        self.assertTrue(build_only(True,**cold))
        self.assertFalse(build_only(False,**cold))
        self.assertFalse(build_only(True,**dict(cold,seen=FRONT)))
        for bad in (dict(age=1),dict(positions={0}),dict(sample=set()),dict(accepted={0}),dict(route="verify")):
            with self.assertRaises(ValueError):
                build_only(True,**dict(cold,**bad))

    def test_kv_only_not_accepted(self):
        self.assertFalse(build_only(True,FRONT,set(),set(),set(),0,{0},{0},"sparse"))
        with self.assertRaises(ValueError):
            build_only(True,FRONT,set(),{1},set(),0,{0},{0},"sparse")


def fixture(on):
    tape=[]
    age=sc=seen=dirty=0

    def add(route, phase, new=(), deferred=()):
        nonlocal age,sc,seen,dirty
        writes=len(new)
        tape.append(dict(route=route, phase=phase, first=not tape, tick=len(tape),
                         age=age, first_samples=list(new), kv_only_revisits=list(deferred),
                         q=32, head=0, seen=seen, sc=sc, sampled=len(new), writes=writes,
                         debt=dirty, first_front_admits=0, canvas="c", rng="r", first_logits="a"*64))
        sc+=len(new)
        age+=bool(new)
        dirty=writes

    for i in range(0,256,32):
        if dirty:
            add("drain","cold")
        skip=on and i==0
        add("first_tile_kv" if skip else "sparse","cold",[] if skip else range(i,i+32))
        seen+=32
    if on:
        add("drain","warm")
        add("sparse","warm",range(32),range(32))
    add("drain","close")
    for _ in range(8):
        add("verify","verify")
    counts=Counter(e["route"] for e in tape)
    x=dict(F=len(tape),decoder_F=len(tape),decoder_Q=32*len(tape),encoder_F=0,encoder_Q=0,
           N=1,correct=False,output_ids=[1],route_F=counts,route_Q={r:32*c for r,c in counts.items()},
           tape=[""]*len(tape))
    return x,tape


class TestLedger(unittest.TestCase):
    def test_real_revisit_required(self):
        x,t=fixture(True)
        d,_=bill(x,t,True)
        self.assertEqual((d["deferred_samples"],d["next_debt_Q"],d["first_writes"]),(32,0,0))
        next(e for e in t if e["kv_only_revisits"])["kv_only_revisits"].pop()
        with self.assertRaisesRegex(ValueError,"starvation"):
            audit(x,t,True)

    def test_clock_and_writeback(self):
        x,t=fixture(True)
        bad=deepcopy(t)
        bad[1]["age"]=1
        with self.assertRaisesRegex(ValueError,"clock"):
            audit(x,bad,True)
        next(e for e in t if e["debt"])["debt"]=0
        with self.assertRaisesRegex(ValueError,"Unpaid debt"):
            bill(x,t,True)

    def test_identical_initial_projection(self):
        x,t=fixture(False)
        _,u=fixture(True)
        self.assertEqual(bill(x,t,False)[0]["first_writes"],32)
        matched(t,u)
        u[0]["first_logits"]="b"*64
        with self.assertRaisesRegex(ValueError,"matched"):
            matched(t,u)

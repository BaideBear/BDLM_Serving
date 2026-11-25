import math
import random
import unittest

from bdlm_serving.methods.t2t.hazard import choose, features
from tools.hazard_gate import compare


class SharedHazardTest(unittest.TestCase):
    def test_offline_and_online_decisions_agree(self):
        rng = random.Random(4)
        for _ in range(300):
            valid = list(range(rng.randrange(6),32))
            masks = sorted(rng.sample(valid,rng.randrange(1,len(valid))))
            resolved = [i not in masks for i in range(32)]
            fresh = {i for i in valid if resolved[i] and rng.randrange(3)==0}
            old = [i for i in valid if resolved[i] and i not in fresh]
            strict = {i for i in fresh if rng.randrange(2)}
            writes = [rng.randint(-1,3) for _ in range(32)]
            w = [rng.uniform(-.8,.8) for _ in range(25)]
            scores = [(1/(1+math.exp(-sum(a*b for a,b in zip(
                features(i,masks,fresh,strict,writes,5), w)))),i) for i in old]
            scores.sort(key=lambda v:(-v[0],v[1]))
            for q, cap in ((4,1),(16,2)):
                for gate in (0,.05):
                    positions, reserve = choose(valid,resolved,fresh,writes,strict,5,q,cap,w,gate)
                    self.assertTrue(fresh <= set(positions))
                    self.assertEqual(len(positions),min(len(valid), max(q,len(fresh))))
                    for i in old:
                        r = compare(valid,masks,old,fresh,scores,{i},q,cap,gate)
                        self.assertEqual(r["incl_backfill"],int(i in positions))
                        self.assertEqual(r["hits"],int(i in reserve))

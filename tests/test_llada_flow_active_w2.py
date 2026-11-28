import unittest
from types import SimpleNamespace

import torch
from bdlm_serving.methods.flowblock.llada_active_w2 import ActiveW2, B, _ShortTail


class ActiveGeometryTest(unittest.TestCase):
    @staticmethod
    def call(obj, start=64, commit=0, right=1, leak=False):
        tail = start + B
        key = torch.arange(tail+2*B)[None,None]
        pos = torch.arange(start,start+2*B)[None]
        end = tail+(1+torch.arange(2*B)//B)[None,:,None]*B
        mask = (key<start) | ((key>=tail) & (key<end))
        if leak:
            mask[0,0,-1] = True
        ActiveW2._geometry(obj, *(torch.tensor([x]) for x in (start,commit,right)),
                           tail,key,mask,pos,256)

    def test_declared_slot_count_not_local_finished_count(self):
        obj = SimpleNamespace(compact=True,ticks=[])
        self.call(obj)
        self.assertEqual(obj.ticks[-1]["physical_width"],32)
        obj.ticks[-1]["slots"].append({"fin": False})
        self.call(obj,right=2)
        obj.ticks[-1]["slots"].extend([{"fin":False},{"fin":True}])
        self.call(obj,right=2)
        self.assertEqual(obj.ticks[-1]["physical_width"],64)

    def test_illegal_future_read_and_commit(self):
        obj = SimpleNamespace(compact=True,ticks=[])
        with self.assertRaisesRegex(ValueError,"dependency"):
            self.call(obj,leak=True)
        self.call(obj)
        obj.ticks[-1]["slots"].append({"fin":True})
        with self.assertRaisesRegex(ValueError,"unopened"):
            self.call(obj,start=128,commit=2,right=3)

    def test_tail_retains_key_coordinates_without_projecting_dead_slot(self):
        d = torch.full((1,2,1,1,128,1),99.)
        c = _ShortTail(d)
        k, v = torch.ones(1,1,32,1), torch.full((1,1,32,1),2.)
        for j,a in enumerate(c.update(k,v,0)):
            self.assertEqual(a.shape[-2],128)
            self.assertTrue((a[..., :64,:]==99).all())
            self.assertTrue((a[...,64:96,:]==(1+j)).all())
            self.assertTrue((a[...,96:,:]==0).all())
        with self.assertRaisesRegex(ValueError,"unopened"):
            c.update(k.repeat(1,1,2,1),v.repeat(1,1,2,1),0)

    def test_control_remains_wide_with_one_slot(self):
        obj = SimpleNamespace(compact=False,ticks=[])
        self.call(obj)
        self.assertEqual(obj.ticks[-1]["physical_width"],64)

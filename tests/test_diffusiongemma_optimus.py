import unittest

import numpy as np
import torch

from bdlm_serving.methods.optimus.upstream import load_controls
from bdlm_serving.runtime.diffusiongemma_optimus import B, CAP, CHUNKS, Resident, controller


class GemmaOptimusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.controls = load_controls()[0]

    def select(self, mode, c, prior, start=0):
        ctl, width = controller(self.controls,mode,c,start,None)
        dst = np.empty(width,dtype=np.int64)
        n,_ = ctl.fill_processing_buffers(dst,0,prior[:width])
        return dst[:n]

    def test_fixed_windows(self):
        for c in CHUNKS:
            for mode in ("ibs","obs"):
                old = np.zeros(2*B,dtype=np.uint8)
                self.assertEqual(self.select(mode,c,old).tolist(),list(range(c)))
                old[:B-2] = 1
                ctl, width = controller(self.controls,mode,c,0,None)
                ctl.update_from_mask(old[:width],mark_ready_if_all_left_non_mask=True)
                dst = np.empty(width,dtype=np.int64)
                n,_ = ctl.fill_processing_buffers(dst,0,old[:width])
                self.assertEqual(dst[:n].tolist(),list(range(B-2,B-2+(2 if mode=="ibs" else c))))
                self.assertEqual(controller(self.controls,mode,c,CAP-B,None)[1],B)

    def test_no_hard_q_clamp(self):
        for c in CHUNKS:
            old = np.zeros(2*B,dtype=np.uint8)
            old[1:1+c] = 1
            self.assertGreater(len(self.select("obs",c,old)),c)

    def test_resident_kv_and_encoder_replacement(self):
        b = Resident()
        def write(start, idx, encoder, payload):
            b.select(start,np.array(idx),torch.device("cpu"))
            q = torch.zeros(1,1,len(idx),1)
            src = torch.tensor(encoder+payload).view(1,1,-1,1)
            k,v = b.extend(0,q,src,src)
            self.assertEqual(k.flatten().tolist(),v.flatten().tolist())
            return k.flatten().tolist()
        self.assertEqual(write(0,[0,1,2],[99],[4,5,6]),[99,4,5,6])
        self.assertEqual(write(0,[0,2,3],[99],[7,8,9]),[99,7,5,8,9])
        self.assertEqual(write(2,[3],[99,70,50],[99]),[99,70,50,8,99])
        with self.assertRaisesRegex(RuntimeError,"uninitialized"):
            b.select(2,np.array([6]),torch.device("cpu"))

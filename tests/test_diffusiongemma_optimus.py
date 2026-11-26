import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from bdlm_serving.methods.optimus.upstream import load_controls
from bdlm_serving.runtime.diffusiongemma_optimus import B, CAP, CHUNKS, Resident, controller
from bdlm_serving.runtime.diffusiongemma_optimus import check_reproduction, historical_refs
from bdlm_serving.evaluation.frozen import sha
from tools.prefix_cohorts import optimus_chunk


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

    def test_reproduction_includes_trace_and_stable_source(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "00-obs_32.json"
            row = dict(events=[("encoder", 42), ("decoder", 31)], output_ids=[1, 2])
            path.write_text(json.dumps(row))
            ref = path, sha(path)
            check_reproduction(ref, row)
            with self.assertRaisesRegex(ValueError, "events"):
                check_reproduction(ref, dict(row, events=[]))
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "changed"):
                check_reproduction(ref, row)

    def test_no_missing_historical_tasks(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp) / "ibs"
            d.mkdir()
            (d / "run.json").write_text(json.dumps(dict(policies=["ibs_32"], hard_q_cap=False)))
            with self.assertRaisesRegex(ValueError, "settings"):
                historical_refs(d.parent, [("ibs", 32)], dict(hard_q_cap=True))
            with self.assertRaises(FileNotFoundError):
                historical_refs(d.parent, [("ibs", 32)], dict(hard_q_cap=False))

    def test_observed_max_is_not_always_the_canvas_width(self):
        row = dict(policy="obs_32", index=0, events=[["encoder", 17], ["decoder", 270]],
                   trace=[dict(q=270, cross=20)], max_q=270, max_decoder_q=270,
                   generation_forwards=1, generation_query_rows=270)
        self.assertEqual(optimus_chunk(row, B), 32)
        for mismatch in (dict(max_q=B), dict(policy="ibs_32"), dict(nominal_q=64),
                         dict(generation_query_rows=287), dict(trace=[dict(q=270,cross=45)])):
            with self.assertRaises(ValueError):
                optimus_chunk(dict(row, **mismatch), B)
        self.assertEqual(optimus_chunk(dict(row, policy="ibs_32", events=[["encoder",17],["decoder",33]],
                         trace=[dict(q=33,cross=0)], max_q=33, max_decoder_q=33, generation_query_rows=33), B), 32)

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

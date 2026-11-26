import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from bdlm_serving.methods.optimus.upstream import load_controls
from bdlm_serving.runtime.diffusiongemma_optimus import B, CAP, CHUNKS, Resident, controller
from bdlm_serving.runtime.diffusiongemma_optimus import check_reproduction, historical_refs
from bdlm_serving.runtime.optimus_limits import limited_rows, parse_caps, validate_trace
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

    def test_limit_only_when_binding_and_drain_before_new_admit(self):
        prior = np.array([1,0,1,1,1,0], dtype=np.uint8)
        dirty = np.array([0,0,1,1,1,0], dtype=bool)
        positions = np.arange(6)
        self.assertIs(limited_rows(positions,prior,dirty,6), positions)
        self.assertEqual(limited_rows(positions,prior,dirty,2).tolist(), [2,3])
        self.assertEqual(limited_rows(positions,prior,dirty,4).tolist(), [1,2,3,4])
        with self.assertRaisesRegex(ValueError,"replay plan"):
            limited_rows(positions[:3],prior,dirty,4)
        self.assertEqual(parse_caps("256,256,128,128",CHUNKS),(256,256,128,128))
        for invalid in ("256n","0","01","512","256,"):
            with self.assertRaises(ValueError):
                parse_caps(invalid,CHUNKS)

    def test_capped_traces_across_slide_with_dense_and_scattered_admissions(self):
        for mode in ("ibs","obs"):
            for slots in ((128,), (256,), (256,256,128,128),(128,128,256,256)):
                for seed in range(3):
                    rng = np.random.default_rng(seed)
                    trace, events, carry = [], [("encoder",13)], None
                    resolved = np.zeros(2*B, dtype=np.uint8)
                    dirty = np.zeros(2*B, dtype=bool)
                    for start in range(0,2*B,B):
                        # Use the same controller, cap and delayed-prefix update as generation.
                        ctl,width = controller(self.controls,mode,max(slots),CAP-2*B+start,carry)
                        buf=np.empty(width,dtype=np.int64)
                        for step in range(2*width+4):
                            part=slice(start,start+width)
                            prior=resolved[part].copy()
                            if ctl.cached_prefix_len >= B:
                                carry=prior[B:].copy()
                                break
                            tick=len(trace)
                            offer=slots[tick%len(slots)]
                            size,_=ctl.fill_processing_buffers(buf,0,prior,chunk_size_override=offer)
                            local=limited_rows(buf[:size].copy(),prior,dirty[part],offer)
                            if mode=="ibs" and slots==(256,):
                                self.assertEqual(buf[:size].tolist(),local.tolist())
                            dirty[start+local]=False
                            before=prior.copy()
                            before[dirty[part]]=0
                            masks=local[prior[local]==0]
                            admit=(masks if seed==0 or len(masks)<2 else
                                   np.sort(rng.choice(masks,size=(len(masks)+1)//2,replace=False)))
                            resolved[start+admit]=1
                            dirty[start+admit]=True
                            ctl.update_from_mask(before,mark_ready_if_all_left_non_mask=True,
                                                 dllm_mask_cur=resolved[part])
                            events.append(("decoder",len(local)))
                            trace.append(dict(tick=tick,offer=offer,q=len(local),block=start//B,
                                positions=(start+local).tolist(),admitted=(start+admit).tolist(),
                                accepted=len(admit),ready=ctl.cached_prefix_len,proposed_q=size,
                                cross=int((local>=B).sum()),visible_end=start+int(max(local))+1))
                            if ctl.cached_prefix_len >= B:
                                carry=before[B:].copy()
                                break
                        else:
                            self.fail("capped requests must progress")
                        if start==0:
                            events.append(("encoder",B))
                    q=sum(n for _,n in events[1:])
                    row=dict(policy=mode+"_cap_"+"_".join(map(str,slots)),events=events,trace=trace,
                        generation_forwards=len(events)-1,generation_query_rows=q,generation_physical_padded_rows=q,
                        max_q=max(n for _,n in events[1:]),max_decoder_q=max(s["q"] for s in trace))
                    validate_trace(row)
                    trace[0]["offer"]=0
                    with self.assertRaisesRegex(ValueError,"cap"):
                        validate_trace(row)

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

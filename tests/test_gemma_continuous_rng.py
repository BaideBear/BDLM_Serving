import unittest
from copy import deepcopy
from types import SimpleNamespace

from experiments.t2t.serving_slots.gemma_continuous_rng import sampler_class
from experiments.t2t.serving_slots.gemma_continuous_rng_report import audit, digest


class Gen:
    def __init__(self, *, device):
        self.device = device

    def manual_seed(self, seed):
        self.state = seed
        return self

    def get_state(self):
        return self.state


class TestRequestRng(unittest.TestCase):
    def run_stream(self, continuous):
        def draw(high, size, *, generator, device):
            self.assertEqual((high, size, device), (80, (1, 256), "A1"))
            value = generator.state
            generator.state += 3
            return value
        api = SimpleNamespace(Generator=Gen, randint=draw)
        module = SimpleNamespace(torch=api)

        class Base:
            def __init__(self, *, cap, vocab_size, seed, device):
                self.rng = module.torch.Generator(device=device).manual_seed(seed)
                self.ids = module.torch.randint(vocab_size, (1, 256), generator=self.rng, device=device)
        module.SelectedRowSampler = Base
        log = []
        cls = sampler_class(module, 5, continuous, log, lambda x: x)
        open_ = lambda s: cls(cap=256, vocab_size=80, seed=s, device="A1")
        first = open_(5)
        first.rng.state += 7
        second = open_(6)
        self.assertIs(first.rng, second.rng) if continuous else self.assertIsNot(first.rng, second.rng)
        with self.assertRaises(ValueError):
            open_(6)
        self.assertIs(module.torch, api)
        self.assertEqual([r["carried"] for r in log], [False, continuous])
        return [(r["before"], r["ids"], r["after"]) for r in log]

    def test_control_reseeds(self):
        self.assertEqual(self.run_stream(False), [(5, 5, 8), (6, 6, 9)])

    def test_request_continues_before_only_draw(self):
        self.assertEqual(self.run_stream(True), [(5, 5, 8), (15, 15, 18)])


class TestReceipt(unittest.TestCase):
    def setUp(self):
        self.reg = dict(pid=123)
        v=dict(ids=[41], encoder_F=0,decoder_Q=512,F=2,mask_layer_calls=60)
        ticks=[dict(ordinal=0,route="dense_single",before=1,end=2),
               dict(ordinal=0,route="verify_paid",before=2,end=2)]
        trace=dict(init=[dict(ordinal=0,carried=False,before=0,after=1,ids=9)],ticks=ticks)
        self.ref=dict(on=deepcopy(v),seed=0,index=0)
        self.row=dict(reference="h",registration=digest(self.reg),pid=123,seed=0,index=0,
                      off=deepcopy(v),on=deepcopy(v),audit=dict(off=trace,on=deepcopy(trace)))

    def test_no_exposure_is_exact_noop(self):
        self.assertEqual(audit(self.row,self.ref,"h",self.reg,0), (False,2))

    def test_unrecorded_reference_field_fails_replay(self):
        self.ref["on"]["layers"]=[]
        with self.assertRaisesRegex(ValueError,"replay"):
            audit(self.row,self.ref,"h",self.reg,0)

    def test_rng_advance_during_verify_rejected(self):
        for a in ("off","on"):
            self.row["audit"][a]["ticks"][-1]["end"]=3
        with self.assertRaisesRegex(ValueError,"RNG tape"):
            audit(self.row,self.ref,"h",self.reg,0)

"""Receipt-level tests only. No torch, CUDA context, weights or model forwards."""
from dataclasses import asdict
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch

from bdlm_serving.methods.t2t.unified_rows import Block, Params, plan_rows
from experiments.t2t.serving_slots import shared_gemma_worker as worker


TASK = dict(index=0, seed=17000, dataset="gsm8k", task_id="gsm8k/167",
            prompt="Six times seven?", reference="42")
PARAMS = Params(prefix_share=Fraction(1,4))
FAIL = "Gemma clock closed before complete sampler coverage"


def receipt(plan, *, route="ordinary"):
    pos = list(plan.rows)
    return dict(offer=32, physical_decoder_positions=pos, positions=pos, decoder_F=1,
                decoder_Q=len(pos), projection_streams=85, route=route,
                shared_plan=None if route != "ordinary" else {**asdict(plan), "drain_only": plan.drain_only})


class ScriptedRequest:
    def __init__(self, meter, failure=None, corrupt=False):
        self.origin, self.status, self.meter = 9, "ready", meter
        meter.started += 1
        self.encoder_receipts = [dict(prefill=True, F=1, Q=self.origin)]
        self.decoder_F = self.decoder_Q = self.cursor = 0
        self.last_attempt = self.tail = None
        self.failure = failure
        self.rows = NS(status="incomplete", age=48, max_steps=48, dirty=set(),
                       initialized=frozenset({3}))
        self.sampler = NS(age=48, sc={3: None})
        self.tokens = []
        view = Block(9,256,{12,13},{9},{9,10,11,13})
        plan = plan_rows(32,256,view,params=PARAMS)
        self.script = [receipt(plan)]
        if not failure:
            self.script.append(receipt(plan, route="verify"))
        if corrupt:
            self.script[0]["decoder_Q"] += 1

    def step(self, offer):
        if not self.script:
            self.status = "failed"
            self.meter.failed += 1
            raise self.failure
        e = self.script.pop(0)
        self.last_attempt = object()
        self.decoder_F += 1
        self.decoder_Q += len(e["positions"])
        if not self.script and not self.failure:
            self.tokens = [10,11,1]
            self.encoder_receipts.append(dict(prefill=False, F=1, Q=256))
            self.status = "done"
            self.meter.completed += 1
        return e

    def completed_bill(self, special):
        return dict(N=sum(t not in special for t in self.tokens),
                    F=self.decoder_F+1, decoder_Q=self.decoder_Q, encoder_Q=256)


class TestSharedGemmaWorker(unittest.TestCase):
    def setUp(self):
        device = NS(type="cuda")
        self.proc = MagicMock()
        self.proc.tokenizer.decode.return_value = r"Answer: \boxed{42}"
        self.handle = worker.Handle(NS(device=device, training=False),self.proc,device,frozenset({1}))

    def run_receipt(self, failure=None, corrupt=False):
        def factory(model, data, *, q, seed, max_new_tokens, params, meter):
            self.assertIs(model, self.handle.model)
            self.assertIs(params, PARAMS)
            self.assertEqual((q, seed, max_new_tokens), (32,17000,2048))
            return ScriptedRequest(meter, failure, corrupt)
        with patch.object(worker.adapter, "create_request", side_effect=factory):
            return worker.evaluate(self.handle, TASK, 32, PARAMS)

    def test_score_and_two_disjoint_Q_denominators(self):
        d = self.run_receipt()
        self.assertEqual((d["status"], d["correct"], d["N"], d["requests"]), ("complete",True,2,1))
        self.assertEqual((d["F"], d["decoder_F"], d["encoder_F"]), (3,2,1))
        self.assertEqual((d["encoder_Q"],d["prefill_Q"]), (256,9))
        self.assertEqual(d["Q"], d["roles"]["sample_Q"]+d["roles"]["verify_Q"])
        self.assertEqual((d["TE"],d["TPF"]), (2/d["Q"],2/3))
        self.assertEqual(d["roles"]["cold_Q"], 1)
        self.assertEqual(d["prompt_sha256"], sha256(TASK["prompt"].encode()).hexdigest())
        self.proc.apply_chat_template.assert_called_once_with(
            [{"role":"user","content":TASK["prompt"]}],tokenize=True,add_generation_prompt=True,
            return_dict=True,return_tensors="pt")
        self.proc.tokenizer.decode.assert_called_once_with([10,11,1], skip_special_tokens=True)

    def test_algorithm_failure_is_in_denominator_and_keeps_physical_work(self):
        d = self.run_receipt(RuntimeError(FAIL))
        self.assertEqual((d["status"],d["correct"],d["N"],d["requests"]),
                         ("valid_algorithm_failure",False,0,1))
        self.assertEqual((d["F"],d["encoder_F"],d["Q"]),(1,0,d["roles"]["sample_Q"]))
        self.assertGreater(d["Q"],0)
        self.proc.tokenizer.decode.assert_not_called()

    def test_no_unpaid_or_unfinished_attempt_can_be_classed_as_clock_failure(self):
        r = ScriptedRequest(worker.adapter.RequestMeter())
        r.status, r.tail = "failed", NS(rows=NS(dirty={4}))
        self.assertFalse(worker._coverage_expired(r, RuntimeError(FAIL), r.last_attempt))
        r.tail = None
        self.assertFalse(worker._coverage_expired(r, RuntimeError(FAIL), object()))
        r.rows.age = 47
        self.assertFalse(worker._coverage_expired(r, RuntimeError(FAIL), r.last_attempt))

    def test_bad_receipts_and_infrastructure_errors_propagate(self):
        with self.assertRaisesRegex(RuntimeError, "physical|measured"):
            self.run_receipt(corrupt=True)
        with self.assertRaisesRegex(RuntimeError, "unexpected CUDA"):
            self.run_receipt(RuntimeError("unexpected CUDA failure"))
        with patch.object(worker.scoring, "score", side_effect=RuntimeError("sandbox unavailable")):
            with self.assertRaisesRegex(RuntimeError, "sandbox unavailable"):
                self.run_receipt()

    def test_refuse_cpu_invalid_offer_and_unknown_dataset_before_preparing_prompt(self):
        with self.assertRaises(ValueError):
            worker.evaluate(self.handle,TASK,33,PARAMS)
        with self.assertRaises(ValueError):
            worker.evaluate(self.handle,dict(TASK,dataset="other"),32,PARAMS)
        self.handle.device.type = "cpu"
        with self.assertRaisesRegex(RuntimeError, "CPU"):
            worker.evaluate(self.handle,TASK,32,PARAMS)
        self.proc.apply_chat_template.assert_not_called()

    def test_loader_no_cuda_no_fallback(self):
        with patch.dict("sys.modules",torch=NS(cuda=NS(is_available=lambda:False))):
            with self.assertRaisesRegex(RuntimeError,"requires CUDA"):
                worker.load(Path("/not-used"))

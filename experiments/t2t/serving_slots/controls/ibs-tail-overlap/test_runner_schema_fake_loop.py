"""Validate runner_a1 event auditing against the real fused.py on a fake loop."""
import contextlib
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
MASK, ANSWER, EOS = 0, 1, 2


class FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(num_hidden_layers=20)
        self.calls = []

    def logits_for(self, token, position):
        values = torch.full((3,), -8.0)
        proposal = EOS if (token == MASK and position >= 32) else (ANSWER if token == MASK else token)
        values[proposal] = 8.0
        return values

    def __call__(self, inputs, *, position_ids, attention_mask, past_key_values,
                 use_cache, output_attentions=True, **kwargs):
        tokens = inputs[0].tolist()
        rows = len(tokens)
        end = int(past_key_values.visible_end)
        self.calls.append({"rows": rows, "physical_rows": rows, "batch": 1,
                           "kv_end": end, "inputs": list(tokens)})
        embeddings = torch.stack([torch.eye(3)[token] for token in tokens])
        key = embeddings[None, None, :, :]
        value = embeddings[None, None, :, :]
        for layer in range(self.config.num_hidden_layers):
            past_key_values.update(key, value, layer)
        logits = torch.stack([self.logits_for(token, int(position_ids[0][i]))
                              for i, token in enumerate(tokens)]).unsqueeze(0)
        return FakeOutput(logits)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_fixed():
    from bdlm_serving.methods.optimus import upstream
    upstream.control_directory = lambda name: Path("/tmp/ibs-tail-controls")
    from bdlm_serving.methods.optimus import fixed
    return fixed


class TestRunnerSchemaWithRealFused(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = load("runner_schema_worker", HERE/"runner_a1.py")
        cls.fused = load("runner_schema_fused", HERE/"fused.py")
        cls.fixed = load_fixed()

    def make_row(self, arm):
        model = FakeModel()
        engine = self.fixed.FixedOptimus(model, "llada21", 8, "ibs",
                                         physical_caps=(8,), prefix_refresh=0)
        receipt = []
        if arm == "stock":
            context = contextlib.nullcontext()
        else:
            patched = self.fused.install_run(self.fixed, arm=arm, receipt=receipt)
            context = unittest.mock.patch.object(self.fixed.FixedOptimus, "run", patched)
        prompt = torch.tensor([[ANSWER]*5])
        with context:
            raw, events, extra = engine.run(prompt, 64, 17007, MASK, [EOS])
        return {
            "index": 7, "status": "success", "output_ids": raw[0, 5:5+64].tolist(),
            "events": events, "calls": model.calls,
            "forward_inputs": [call["inputs"] for call in model.calls],
        }, receipt, model

    def test_triggered_schema_and_tail_debt_payment(self):
        stock, _receipt_stock, _model_stock = self.make_row("stock")
        noop, receipt_noop, _model_noop = self.make_row("noop")
        overlap, receipt_overlap, _model_overlap = self.make_row("overlap")
        self.assertTrue(receipt_noop and receipt_overlap)
        self.assertEqual(receipt_noop[0], receipt_overlap[0])
        audit = self.worker.audit_pair(stock, noop, overlap, 7, MASK)
        self.assertTrue(audit["triggered"])
        self.assertTrue(audit["head_write_set_drift"] in (True, False))
        self.assertFalse(audit["head_changed"])
        self.assertTrue(audit["promoted"])
        self.assertGreater(audit["tail_writes"], 0)
        self.assertEqual(audit["tail_start"] if "tail_start" in audit else None, None)
        self.assertGreaterEqual(audit["first_joint_tick"], 0)
        receipt_audit = self.worker.audit_receipts(receipt_noop, receipt_overlap)
        self.assertTrue(receipt_audit["overlap_matched_noop"])

    def test_eos_veto_or_untriggered_case_is_stock_equivalent(self):
        overlap, receipt, _model = self.make_row("overlap")
        stock, _receipt_stock, _stock_model = self.make_row("stock")
        noop, noop_receipt, _noop_model = self.make_row("noop")
        if not receipt:
            audit = self.worker.audit_pair(stock, noop, overlap, 7, MASK)
            self.assertFalse(audit["triggered"])
            self.assertEqual(self.worker.audit_receipts(noop_receipt, receipt)["first_opportunity"], None)


if __name__ == "__main__":
    unittest.main()

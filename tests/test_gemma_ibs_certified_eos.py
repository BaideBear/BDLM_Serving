from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from experiments.t2t.serving_slots.gemma_ibs_certified_eos import (
    B, CAP, OFFERS, Stop, _instrumented_loop, eligible,
)

OFFER = 32


class CertifiedEos(unittest.TestCase):
    def test_eligibility(self):
        self.assertIsNone(eligible(7, 7, [], OFFER))
        self.assertIsNone(eligible(B, 2, [], OFFER))
        self.assertIsNone(eligible(8, None, [], OFFER))
        self.assertIsNone(eligible(8, 2, [-1, 12], OFFER))
        self.assertIsNone(eligible(8, 2, [B], OFFER))
        self.assertIsNone(eligible(8, 2, [1, 12], OFFER))
        self.assertIsNone(eligible(8, 2, list(range(12, 12+OFFER+1)), OFFER))
        self.assertEqual(eligible(8, 2, list(range(12, 12+OFFER)), OFFER), "suffix_debt")
        self.assertEqual(eligible(8, 2, [], OFFER), "clean")

    def test_paid_only_projects_debt_and_returns_through_first_eos(self):
        for arm in ("strict", "control", "paid"):
            with self.subTest(arm=arm):
                stop = Stop(arm, OFFER)
                ids = np.arange(CAP)[None, :]
                resolved = np.zeros(CAP, bool)
                dirty = np.zeros(CAP, bool)
                start = 2*B
                resolved[start:start+8] = True
                dirty[start+11] = resolved[start+11] = True
                trace = [dict(tick=12, offer=OFFER)]
                self.assertIsNone(stop.after_update(ids, resolved, dirty, start, 8,
                                                    {start+2, start+5}, trace))
                rows = stop.rows(np.array([11, 12, 13]), resolved[start:start+B], dirty, start, OFFER)
                self.assertEqual(rows.tolist(), [11] if arm == "paid" else [11, 12, 13])
                if arm == "paid":
                    with self.assertRaises(RuntimeError):
                        stop.after_update(ids, resolved, dirty, start, 8, {start+2}, trace)
                    sparse = dict(tick=13, positions=[start+11], q=1, offer=OFFER, admitted=[], accepted=0)
                    trace.append(sparse)
                else:
                    trace.append(dict(tick=13, offer=OFFER))
                dirty[:] = False
                out = stop.after_update(ids, resolved, dirty, start, 8, {start+2}, trace)
                self.assertEqual(out, None if arm == "control" else list(range(start+3)))
                self.assertEqual(stop.trigger and stop.trigger["tick"],
                                 {"control": None, "strict": 13, "paid": 12}[arm])

    def test_offer_is_explicit_and_bound_to_real_payment(self):
        with self.assertRaises(RuntimeError):
            Stop("paid", 256)
        for offer in OFFERS:
            with self.subTest(offer=offer):
                stop = Stop("paid", offer)
                ids = np.arange(CAP)[None, :]
                dirty = np.zeros(CAP, bool)
                dirty[8:8+offer] = True
                resolved = np.zeros(CAP, bool)
                resolved[:8+offer] = True
                t = [dict(tick=0, offer=offer)]
                self.assertIsNone(stop.after_update(ids, resolved, dirty, 0, 8, {2}, t))
                picked = np.arange(8, 8+offer)
                self.assertTrue(np.array_equal(stop.rows(picked, resolved[:B], dirty, 0, offer), picked))
                with self.assertRaises(RuntimeError):
                    stop.rows(picked, resolved[:B], dirty, 0, 32 if offer != 32 else 64)

    def test_compiles_only_pinned_source(self):
        import os
        source = Path(os.environ.get("IBS_TEST_BASE",
            "/gs/work/gemma-ibs512-1127/bdlm_serving/runtime/diffusiongemma_optimus.py"))
        if not source.exists():
            source = Path(__file__).resolve().parents[1] / "bdlm_serving/runtime/diffusiongemma_optimus.py"
        class Module:
            __file__ = str(source)
        dummy = Module()
        original_globals = vars(dummy).copy()
        patched = _instrumented_loop(dummy)
        self.assertEqual(patched.__code__.co_varnames[:10],
                         ("bank", "model", "data", "controls", "unmask", "mode", "c", "qualify",
                          "caps", "_ibs_eos"))
        self.assertIn("stop", patched.__code__.co_varnames[10:])
        # exec inserts __builtins__ if not already present on the real module.
        self.assertEqual({k:v for k,v in vars(dummy).items() if k != "__builtins__"}, original_globals)


if __name__ == "__main__":
    unittest.main()

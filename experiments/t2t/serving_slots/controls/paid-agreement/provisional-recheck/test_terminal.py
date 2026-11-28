"""Exercise terminal debt sequencing without a model or GPU."""
import unittest

import numpy as np

from experiments.t2t.serving_slots.gemma_provisional_recheck import RecheckStop


class TerminalRepaymentTest(unittest.TestCase):
    def setUp(self):
        self.stop = RecheckStop(128)
        self.stop.trigger = dict(kind="suffix_debt", block=0, tick=9, eos_index=5,
                                 eos_id=106, debt=[8, 10])
        self.ids = np.zeros((1, 2048), dtype=int)
        self.ids[0, 5] = 106
        self.resolved = np.ones(2048, dtype=np.uint8)
        self.dirty = np.zeros(2048, dtype=bool)

    def step(self, tick, rows):
        trace = [dict(offer=128, tick=tick, q=len(rows), positions=rows, admitted=[], accepted=0)]
        return self.stop.after_update(self.ids, self.resolved, self.dirty, 0, 8, {106}, trace)

    def test_cannot_deliver_until_physical_suffix_repayment(self):
        self.dirty[8] = True
        self.assertIsNone(self.step(10, [8, 10]))
        self.assertIsNone(self.stop.payment)
        self.assertEqual(self.stop.rows(np.arange(8, 128), self.resolved, self.dirty, 0, 128).tolist(), [8])
        self.dirty[8] = False
        self.assertEqual(self.step(11, [8]), [0, 0, 0, 0, 0, 106])
        self.assertEqual(self.stop.payment, dict(tick=11, positions=[8], Q=1))

    def test_refuses_debt_outside_suffix_or_a_third_payment(self):
        self.dirty[4] = True
        with self.assertRaisesRegex(RuntimeError, "NO-GO"):
            self.step(10, [8, 10])
        self.dirty[4] = False
        self.dirty[8] = True
        self.assertIsNone(self.step(10, [8, 10]))
        with self.assertRaisesRegex(RuntimeError, "NO-GO"):
            self.step(11, [8])


if __name__ == "__main__":
    unittest.main()

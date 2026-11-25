import unittest

from bdlm_serving.runtime.diffusiongemma import account


class TestNativeLedger(unittest.TestCase):
    def test_two_canvases(self):
        tokens, m = account([("encoder", 7), ("decoder", 4), ("decoder", 4),
                             ("encoder", 4), ("decoder", 4)], [3, 4, 5, 6, 7, 100, 1, 0],
                            7, 4, 8, {1}, {100}, 0, 7 / 3)
        self.assertEqual(tokens, [3, 4, 5, 6, 7, 100, 1])
        self.assertEqual((m["output_tokens"], m["generation_forwards"], m["generation_query_rows"]),
                         (5, 4, 16))
        self.assertEqual(m["reencode_forwards"], 1)
        self.assertFalse(m["truncated"])

    def test_cap(self):
        _, m = account([("encoder", 8), ("decoder", 4)], [3] * 4, 8, 4, 4, {1}, set(), 0, 4)
        self.assertTrue(m["truncated"])
        self.assertEqual(m["reencode_query_rows"], 0)

    def test_inconsistent_trace(self):
        for events, tpf in [([("encoder", 8), ("decoder", 4)], 1),
                            ([("encoder", 8), ("decoder", 2)], 4),
                            ([("encoder", 8), ("encoder", 4), ("decoder", 4)], 4)]:
            with self.assertRaises(ValueError):
                account(events, [3] * 4, 8, 4, 4, {1}, set(), 0, tpf)

import unittest

from bdlm_serving.methods.t2t.protocol import Arm, matrix, verify_bridge, verify_work


class ProtocolTest(unittest.TestCase):
    def test_fresh_random_is_zero_edit_gemma_only(self):
        self.assertEqual(Arm.parse("64:4:60:fresh_random", 256).edit, 0)
        for form, block in (("64:4:56:fresh_random", 256), ("16:4:12:fresh_random", 32)):
            with self.assertRaises(ValueError):
                Arm.parse(form, block)

    def test_paired_controls(self):
        for model, block in [("gemma", 256), ("llada21", 32)]:
            arms = matrix(model)
            self.assertEqual(len(arms), len(set(arms)))
            for arm in arms:
                self.assertEqual(arm, Arm.parse(f"{arm.budget}:{arm.fresh}:{arm.mask}:{arm.selection}", block))
                if arm.selection == "targeted":
                    self.assertIn(Arm(arm.budget, arm.fresh, arm.mask, "random"), arms)
                    self.assertIn(Arm(arm.budget, arm.fresh, arm.budget-arm.fresh, "noedit"), arms)

    def test_fail_on_free_commit_or_bad_bridge(self):
        r = dict(output_tokens=12, generation_forwards=2, generation_query_rows=20,
                 generation_physical_padded_rows=20, max_q=16, tpf=6., token_efficiency=.6,
                 index=0, dataset="x", task_id="x/0", seed=123, output_ids=[17])
        verify_work(r)
        verify_bridge(r, r)
        for other in (dict(r, max_q=12), dict(r, output_ids=[18]), dict(r, generation_forwards=3)):
            with self.assertRaises(ValueError):
                verify_bridge(other, r)
        with self.assertRaises(ValueError):
            verify_work(dict(r, generation_query_rows=4))

    def test_record_retains_full_trace_and_charges_commit(self):
        from types import SimpleNamespace
        from bdlm_serving.methods.t2t.online import package
        tok = SimpleNamespace(decode=lambda seq, **kw: str(seq))
        record = package(tok, [7, 8, 1, 9], [("selected", 4), ("commit", 32)],
                         {1}, {1}, 44, events=[{"stage": "selected"}])
        verify_work(record)
        self.assertEqual((record["output_ids"], record["generation_query_rows"], record["max_q"]),
                         ([7, 8, 1], 36, 32))
        self.assertEqual(record["events"], [{"stage":"selected"}])

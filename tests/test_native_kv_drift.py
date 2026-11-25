"""CPU-only tests for the native DiffusionGemma per-position K/V drift diagnostic."""
import gzip
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from bdlm_serving.methods.t2t.frontier import _frontier_scores
from tools.native_kv_drift import (
    ATTENTION_ALIAS, EVENTS, CaptureStore, DriftState, aggregate_tasks,
    capture_attention, classify, enrichment, kv_drift_trace,
    offline_budget_coverage, relative_rms, shared_kv_layers, summary_row,
    summarize_task, top_set, verify_parity,
)


class RelativeRmsTest(unittest.TestCase):
    def test_zero(self):
        prev = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        self.assertEqual(relative_rms(prev.clone(), prev), [0.0, 0.0])

    def test_known_scaling(self):
        prev = torch.tensor([[[[1.0, 1.0], [2.0, 2.0]]]])
        for value in relative_rms(prev * 2, prev):
            self.assertAlmostEqual(value, 1.0, places=6)

    def test_shape_rejected(self):
        with self.assertRaises(ValueError):
            relative_rms(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 3, 2))


class TopSetTest(unittest.TestCase):
    def test_positive_cutoff_expands_ties(self):
        values = {0: 0.9, 1: 0.8, 2: 0.8, 3: 0.1, 4: 0.0}
        selected, k, cutoff, tie, expanded = top_set(values, range(5), fraction=0.4)
        self.assertEqual((k, cutoff, tie, expanded), (2, 0.8, 2, True))
        self.assertEqual(set(selected), {0, 1, 2})

    def test_zero_cutoff_does_not_expand(self):
        selected, k, cutoff, tie, expanded = top_set(
            {0: 0.0, 1: 0.0, 2: 0.0}, range(3), fraction=0.5)
        self.assertEqual((selected, k, cutoff, tie, expanded),
                         ([0, 1], 2, 0.0, 3, False))

    def test_floor_filters_near_zero(self):
        selected, k, cutoff, tie, expanded = top_set(
            {0: 1e-3, 1: 1e-9, 2: 0.0}, range(3), k=2, floor=1e-6)
        self.assertEqual((selected, k, cutoff, tie, expanded),
                         ([0], 2, 1e-3, 1, False))

    def test_floor_can_remove_all_positives(self):
        selected, k, cutoff, _, _ = top_set(
            {0: 1e-9, 1: 0.0}, [0, 1], k=1, floor=1e-6)
        self.assertEqual((selected, k, cutoff), ([], 1, None))

    def test_nominal_k_comes_from_full_pool(self):
        selected, k, _, _, _ = top_set(
            {0: 1.0, 1: 0.0, 2: 0.0}, range(3), k=2, floor=0.5)
        self.assertEqual((selected, k), ([0], 2))

    def test_tie_aware_false_keeps_exactly_k(self):
        selected, k, cutoff, tie, expanded = top_set(
            {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0}, range(4), k=2, tie_aware=False)
        self.assertEqual((selected, k, cutoff, tie, expanded),
                         ([0, 1], 2, 1.0, 3, False))

    def test_explicit_top_k(self):
        selected, k, cutoff, _, _ = top_set(
            {0: 0.1, 1: 0.3, 2: 0.2}, range(3), k=1)
        self.assertEqual((selected, k, cutoff), ([1], 1, 0.3))

    def test_empty(self):
        self.assertEqual(top_set({}, []), ([], 0, None, 0, False))

    def test_invalid_fraction(self):
        with self.assertRaises(ValueError):
            top_set({0: 1.0}, [0], fraction=0.0)


class ClassifyTest(unittest.TestCase):
    def test_events(self):
        prev_resolved = [True, True, True, False, True, False]
        prev_fresh = {1}
        before = [1, 2, 3, 0, 5, 0]
        after = [2, 9, 3, 4, 5, 0]
        accepted = [True, True, True, True, False, False]
        self.assertEqual(classify(prev_resolved, prev_fresh, before, after, accepted, 0),
                         "t2t_old")
        self.assertEqual(classify(prev_resolved, prev_fresh, before, after, accepted, 1),
                         "t2t_fresh")
        self.assertEqual(classify(prev_resolved, prev_fresh, before, after, accepted, 2),
                         "context")
        self.assertEqual(classify(prev_resolved, prev_fresh, before, after, accepted, 3),
                         "admit")
        self.assertEqual(classify(prev_resolved, prev_fresh, before, after, accepted, 4),
                         "revoke")
        self.assertEqual(classify(prev_resolved, prev_fresh, before, after, accepted, 5),
                         "mask_stay")


class EnrichmentTest(unittest.TestCase):
    def test_known(self):
        stats = enrichment([0, 1, 2, 3], [0, 1], [0, 2])
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["precision"], 0.5)
        self.assertEqual(stats["recall"], 0.5)
        self.assertEqual(stats["base_rate"], 0.5)
        self.assertEqual(stats["uniform_expected"], 1.0)
        self.assertEqual(stats["lift"], 1.0)
        self.assertEqual(stats["uniform_positional_hits"], 2)

    def test_perfect_selection_doubles_uniform(self):
        stats = enrichment([0, 1, 2, 3], [0, 1], [0, 1])
        self.assertEqual(stats["hits"], 2)
        self.assertEqual(stats["lift"], 2.0)

    def test_empty_top_has_no_expectation(self):
        stats = enrichment([0, 1, 2, 3], [], [0, 1])
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["base_rate"], 0.0)
        self.assertEqual(stats["uniform_expected"], 0.0)
        self.assertIsNone(stats["lift"])


class FrontierPredictionTest(unittest.TestCase):
    def test_previous_write_neighborhood(self):
        last_write = [-1] * 8
        last_write[4] = 5
        scores = _frontier_scores(list(range(8)), last_write, [5] * 8, 5)
        selected, k, _, _, _ = top_set(scores, range(8), k=2)
        self.assertEqual((set(selected), k), ({3, 5}, 2))


class SharedKvTest(unittest.TestCase):
    def test_detection(self):
        def attn(shared):
            return NS(v_proj=None if shared else object())
        layers = [NS(self_attn=attn(False)), NS(self_attn=attn(True)),
                  NS(self_attn=attn(False))]
        model = NS(model=NS(decoder=NS(layers=layers)))
        self.assertEqual(shared_kv_layers(model), [1])


class CaptureAttentionTest(unittest.TestCase):
    def test_passthrough_and_canvas_slice(self):
        impl = {"sdpa": lambda module, q, k, v, mask, **kw: (q, None)}
        attn = NS(layer_idx=0, config=NS(_attn_implementation="sdpa"))
        model = NS(model=NS(decoder=NS(layers=[NS(self_attn=attn)])))
        store = CaptureStore()
        store.active = True
        q = torch.arange(8, dtype=torch.float32).view(1, 1, 4, 2)
        k = torch.arange(16, dtype=torch.float32).view(1, 1, 8, 2)
        v = k + 100
        with capture_attention(model, {0}, 4, store, impl=impl):
            out, weights = impl[ATTENTION_ALIAS](attn, q, k, v, None)
        self.assertIsNone(weights)
        self.assertTrue(torch.equal(out, q))
        self.assertTrue(torch.equal(store.k[0], k[..., -4:, :]))
        self.assertTrue(torch.equal(store.v[0], v[..., -4:, :]))
        self.assertNotIn(ATTENTION_ALIAS, impl)
        self.assertEqual(attn.config._attn_implementation, "sdpa")

    def test_rejects_non_sdpa(self):
        impl = {"sdpa": lambda *a, **k: (a[1], None)}
        attn = NS(layer_idx=0, config=NS(_attn_implementation="eager"))
        model = NS(model=NS(decoder=NS(layers=[NS(self_attn=attn)])))
        with self.assertRaises(ValueError):
            with capture_attention(model, {0}, 4, CaptureStore(), impl=impl):
                pass


class DriftStateTest(unittest.TestCase):
    def test_two_steps_and_true_t2t(self):
        state = DriftState(width=4, layer=0, top_fraction=0.5)
        k1 = torch.ones(1, 1, 4, 2)
        v1 = torch.ones(1, 1, 4, 2)
        state.begin(10, 3, [0, 0, 0, 0])
        state.end([1, 0, 0, 0], [True, False, False, False], k1, v1, seen=1)
        self.assertEqual(state.records, [])

        k2, v2 = k1 * 1.1, v1 * 1.1
        state.begin(10, 2, [1, 0, 0, 0])
        state.end([1, 2, 0, 0], [True, True, False, False], k2, v2, seen=1)
        self.assertEqual(state.records, [])  # the only resolved row is still fresh

        k3, v3 = k2 * 1.1, v2 * 1.1
        state.begin(10, 1, [1, 2, 0, 0])
        state.end([3, 2, 0, 0], [True, True, False, False], k3, v3, seen=1)
        self.assertEqual(len(state.records), 1)
        record = state.records[0]
        self.assertEqual((record["pool_size"], record["top_count"]), (1, 1))
        self.assertEqual(record["hits"], 1)
        self.assertEqual(record["class_counts"]["t2t_old"], 1)
        self.assertEqual(record["class_counts"]["context"], 0)
        self.assertEqual(record["class_counts"]["revoke"], 0)
        self.assertEqual(record["lift"], 1.0)
        self.assertEqual(record["top_positions"][0][4], "t2t_old")
        self.assertEqual((record["step_id"], record["prev_step_id"]), (3, 2))
        self.assertEqual((record["raw_clock"], record["prev_raw_clock"]), (1, 2))
        self.assertEqual((state.last_write[0], state.last_write[1]), (3, 2))

    def test_new_canvas_resets(self):
        state = DriftState(width=4, layer=0, top_fraction=0.5)
        k, v = torch.ones(1, 1, 4, 2), torch.ones(1, 1, 4, 2)
        state.begin(10, 3, [0, 0, 0, 0])
        state.end([1, 0, 0, 0], [True, False, False, False], k, v, seen=1)
        state.begin(14, 3, [0, 0, 0, 0])
        self.assertIsNone(state.prev)
        self.assertFalse(any(state.h.resolved))
        self.assertEqual(state.step_id, 1)
        state.end([0, 0, 0, 0], [False, False, False, False], k, v, seen=1)
        self.assertEqual(state.records, [])

    def test_all_zero_drift_has_no_false_positives(self):
        state = DriftState(width=4, layer=0, top_fraction=0.5)
        k, v = torch.ones(1, 1, 4, 2), torch.ones(1, 1, 4, 2)
        state.begin(10, 3, [0, 0, 0, 0])
        state.end([1, 0, 0, 0], [True, False, False, False], k, v, seen=1)
        state.begin(10, 2, [1, 0, 0, 0])
        state.end([1, 2, 0, 0], [True, True, False, False], k, v, seen=1)
        state.begin(10, 1, [1, 2, 0, 0])
        state.end([1, 2, 0, 0], [True, True, False, False], k, v, seen=1)
        record = state.records[0]
        self.assertEqual(record["top_count"], 0)
        self.assertEqual(record["positives_available"], 0)
        self.assertEqual(record["uniform_expected"], 0.0)
        self.assertEqual(record["hits"], 0)
        self.assertIsNone(record["lift"])
        self.assertEqual(record["selected_count"], 1)

    def test_prediction_is_fixed_k_not_len_top(self):
        state = DriftState(width=4, layer=0, top_fraction=0.5)
        k1, v1 = torch.ones(1, 1, 4, 2), torch.ones(1, 1, 4, 2)
        state.begin(10, 3, [0, 0, 0, 0])
        state.end([1, 1, 1, 1], [True] * 4, k1, v1, seen=1)
        state.begin(10, 2, [1, 1, 1, 1])
        state.end([1, 1, 1, 1], [True] * 4, k1, v1, seen=1)
        k3, v3 = k1 * 1.1, v1 * 1.1
        state.begin(10, 1, [1, 1, 1, 1])
        state.end([1, 1, 1, 1], [True] * 4, k3, v3, seen=1)
        record = state.records[0]
        self.assertEqual(record["nominal_k"], 2)
        self.assertEqual(record["selected_count"], 2)
        self.assertFalse(record["selected_expanded"])
        self.assertEqual(len(record["selected_positions"]), 2)
        self.assertEqual(record["top_count"], 4)  # labels may expand, seats may not

    def test_seen_must_be_one(self):
        state = DriftState(width=2, layer=0, top_fraction=0.5)
        k, v = torch.ones(1, 1, 2, 1), torch.ones(1, 1, 2, 1)
        state.begin(0, 3, [0, 0])
        with self.assertRaises(RuntimeError):
            state.end([1, 0], [True, False], k, v, seen=0)

    def test_rejects_canvas_jump(self):
        state = DriftState(width=4, layer=0, top_fraction=0.5)
        state.begin(10, 3, [0, 0, 0, 0])
        state.end([0, 0, 0, 0], [False] * 4,
                  torch.ones(1, 1, 4, 1), torch.ones(1, 1, 4, 1), seen=1)
        with self.assertRaises(RuntimeError):
            state.begin(15, 3, [0, 0, 0, 0])


class SummaryTest(unittest.TestCase):
    def record(self):
        record = dict(
            block=10, clock=1, prev_clock=2, step_id=3, prev_step_id=2,
            phase="t-1_to_t_same_canvas", pool_kind="non_fresh_resolved",
            pool_size=4, nominal_k=2, top_k=2, top_count=2, positives_available=2,
            top_cutoff=0.5, top_tie_count=2, top_expanded=False,
            selected_count=2, selected_cutoff=0.1, selected_tie_count=1,
            selected_expanded=False, hits=1, precision=0.5, recall=0.5,
            base_rate=0.5, uniform_expected=1.0, uniform_positional_hits=1,
            lift=1.0, class_counts={e: 0 for e in EVENTS},
            class_hits={e: 0 for e in EVENTS}, mean_dk_top=0.4,
            mean_dv_top=0.3, mean_joint_top=0.4, mean_drift_top=0.4,
            mean_dk_pool=0.2, mean_dv_pool=0.1, mean_joint_pool=0.2,
            mean_drift_pool=0.2, mean_score_selected=0.3,
            mean_score_pool=0.1, top_positions=[], selected_positions=[],
        )
        record["class_counts"]["t2t_old"] = 1
        record["class_hits"]["t2t_old"] = 1
        return record

    def test_task_and_aggregate(self):
        record = self.record()
        summary = summarize_task([record])
        self.assertEqual(summary["hits_total"], 1)
        self.assertEqual(summary["micro_lift"], 1.0)
        self.assertEqual(summary["t2t_lift"], 2.0)
        self.assertEqual(summary["t2t_total"], 1)
        payload = dict(index=0, task_id="a", dataset="d", seed=1, layer=15,
                       layer_type="sliding_attention", shared_kv=False,
                       shared_kv_layers=[], summary=summary, steps=[record])
        aggregate = aggregate_tasks([payload], [dict(index=0)], 15)
        self.assertEqual(aggregate["hits_total"], 1)
        self.assertEqual(aggregate["micro_lift"], 1.0)
        self.assertEqual(aggregate["macro_lift_task"], 1.0)
        row = summary_row(payload)
        self.assertEqual(row["hits_total"], 1)
        self.assertEqual(row["t2t_total"], 1)


class ParityTest(unittest.TestCase):
    def test_exact_match_and_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out, ref = root / "out", root / "ref"
            out.mkdir()
            ref.mkdir()
            tasks = [dict(index=0, task_id="a", dataset="d", seed=1),
                     dict(index=1, task_id="b", dataset="d", seed=2)]
            for i in range(2):
                row = dict(index=i, task_id=tasks[i]["task_id"], dataset="d",
                           seed=i + 1, output_tokens=10, generation_forwards=2,
                           generation_query_rows=20,
                           events=[["encoder", 5], ["decoder", 10]],
                           output_ids=[1, 2, 3])
                (out / f"{i:02d}-native256.json").write_text(json.dumps(row))
                (ref / f"{i:02d}-native256.json").write_text(json.dumps(row))
            rows = verify_parity(out, ref, tasks)
            self.assertEqual([r["index"] for r in rows], [0, 1])
            bad = json.loads((out / "01-native256.json").read_text())
            bad["output_tokens"] = 11
            (out / "01-native256.json").write_text(json.dumps(bad))
            with self.assertRaises(AssertionError):
                verify_parity(out, ref, tasks)


class _FakeModel:
    def __init__(self, canvas=4):
        self.config = NS(canvas_length=canvas)
        attn = NS(layer_idx=0, layer_type="sliding_attention", v_proj=object(),
                  head_dim=2, k_proj=NS(out_features=2),
                  config=NS(_attn_implementation="sdpa"))
        self.model = NS(decoder=NS(layers=[NS(self_attn=attn)]))
        self.calls = 0

    def _denoising_step(self, **kw):
        from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
            ALL_ATTENTION_FUNCTIONS as impl,
        )
        attn = self.model.decoder.layers[0].self_attn
        scale = (1.0, 1.1, 1.21)[self.calls]
        self.calls += 1
        q = torch.full((1, 1, 4, 2), scale)
        k = torch.full((1, 1, 6, 2), scale)
        v = torch.full((1, 1, 6, 2), scale)
        impl[attn.config._attn_implementation](attn, q, k, v, None)
        after = torch.tensor([[[1, 0, 0, 0], [1, 2, 0, 0], [3, 2, 0, 0]][self.calls - 1]])
        accepted = torch.tensor([[[True, False, False, False],
                                  [True, True, False, False],
                                  [True, True, False, False]][self.calls - 1]])
        kw["sampler"].accepted_token_mask = accepted
        logits = torch.zeros(1, 4, 4)
        return after, after.clone(), logits, torch.zeros(1, dtype=torch.bool)


class TraceIntegrationTest(unittest.TestCase):
    def test_composition_writes_one_record(self):
        fake = types.ModuleType(
            "transformers.models.diffusion_gemma.modeling_diffusion_gemma")
        fake.ALL_ATTENTION_FUNCTIONS = {
            "sdpa": lambda module, q, k, v, mask, **kw: (q, None)}
        model = _FakeModel()
        task = dict(index=0, task_id="a", dataset="d", seed=1)
        with tempfile.TemporaryDirectory() as td, patch.dict(
                sys.modules, {fake.__name__: fake}):
            out = Path(td)
            with kv_drift_trace(out, layer=0, top_fraction=0.5)(model, task):
                for step, clock in enumerate((3, 2, 1), start=1):
                    kw = dict(
                        input_ids=torch.zeros(1, 10, dtype=torch.long),
                        cur_step=clock,
                        current_canvas=torch.tensor(
                            [[[0, 0, 0, 0], [1, 0, 0, 0], [1, 2, 0, 0]][step - 1]]),
                        sampler=NS(accepted_token_mask=torch.zeros(1, 4, dtype=torch.bool)),
                    )
                    model._denoising_step(**kw)
            trace = out / "00.trace.jsonl.gz"
            self.assertTrue(trace.exists())
            with gzip.open(trace, "rt") as stream:
                rows = [json.loads(line) for line in stream]
            self.assertEqual(len(rows), 1 + 3)
            payload = json.loads((out / "00-kvdrift-layer0.json").read_text())
            self.assertEqual(len(payload["steps"]), 1)
            self.assertEqual(payload["steps"][0]["step_id"], 3)
            self.assertEqual(payload["summary"]["hits_total"], 1)


class OfflineBudgetCoverageTest(unittest.TestCase):
    def _synthetic(self, root):
        folder = Path(root) / "archive"
        folder.mkdir()
        width = 32
        # Step 1 resolves 0..21; step 2 resolves 22..29; only 30,31 stay MASK.
        # The step-3 non-fresh pool is 0..21; the target old rows are 4,5.
        after1 = [1] * 22 + [0] * 10
        accepted1 = [True] * 22 + [False] * 10
        after2 = [1] * 30 + [0] * 2
        accepted2 = [True] * 30 + [False] * 2
        after3 = [1, 1, 1, 1, 2, 2] + [1] * 24 + [1, 1]
        accepted3 = [True] * 32
        rows = [
            dict(block=10, clock=3, before=[0] * 32, after=after1,
                 accepted=accepted1, draft=after1, certainty=[0.0] * 32),
            dict(block=10, clock=2, before=after1, after=after2,
                 accepted=accepted2, draft=after2, certainty=[0.0] * 32),
            dict(block=10, clock=1, before=after2, after=after3,
                 accepted=accepted3, draft=after3, certainty=[0.0] * 32),
        ]
        with gzip.open(folder / "00.trace.jsonl.gz", "wt") as stream:
            stream.write(json.dumps(dict(
                schema=1, index=0, task_id="a", dataset="d", seed=1)) + "\n")
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        diagnostic = dict(
            index=0, task_id="a", dataset="d", seed=1, layer=15, canvas=width,
            phase="t-1_to_t_same_canvas", pool_kind="non_fresh_resolved",
            steps=[dict(
                block=10, clock=1, prev_clock=2, raw_clock=1, prev_raw_clock=2,
                step_id=3, prev_step_id=2,
                phase="t-1_to_t_same_canvas", pool_kind="non_fresh_resolved",
                pool_size=22,
                top_positions=[[4, 0.5, 0.4, 0.5, "t2t_old"],
                               [5, 0.4, 0.5, 0.5, "t2t_old"]],
            )],
        )
        (folder / "00-kvdrift-layer15.json").write_text(json.dumps(diagnostic))
        (folder / "verified.json").write_text(
            json.dumps(dict(model="gemma", matched_requests=1)))
        return folder

    def test_coverage_and_dense_control(self):
        with tempfile.TemporaryDirectory() as td:
            folder = self._synthetic(td)
            result = offline_budget_coverage(
                folder, layer=15, budgets=(16, 32),
                policies=("hard_frontier", "hard_noedit"),
                out=folder / "out", require_30=False)
            by_key = {(row["budget"], row["policy"]): row
                      for row in result["aggregate"]}
            frontier = by_key[(16, "hard_frontier")]
            noedit = by_key[(16, "hard_noedit")]
            dense = by_key[(32, "hard_noedit")]
            self.assertEqual(frontier["reserved_hits"], 0)
            self.assertEqual(frontier["total_hits"], 0)
            self.assertEqual(frontier["reserved_count"], 2)
            self.assertEqual(frontier["seats_total"], 16)
            self.assertEqual(frontier["neutral_backfill_count"], 4)
            self.assertAlmostEqual(frontier["random_expectation_total"], 12 / 22)
            self.assertEqual(frontier["random_lift"], 0.0)
            self.assertEqual(frontier["mode"], "fixed_edit_seats")
            self.assertEqual(frontier["delta_total_hits_vs_noedit"], -2)
            self.assertEqual(frontier["delta_mask_lost_vs_noedit"], 0)
            self.assertEqual(noedit["total_hits"], 2)
            self.assertEqual(noedit["reserved_hits"], 0)
            self.assertEqual(noedit["neutral_backfill_count"], 6)
            self.assertAlmostEqual(noedit["random_expectation_total"], 12 / 22)
            self.assertAlmostEqual(noedit["random_lift"], 2 / (12 / 22))
            self.assertEqual(dense["total_hits"], 2)
            self.assertEqual(dense["target_coverage"], 1.0)
            self.assertTrue(dense["dense_control"])
            self.assertEqual(dense["mode"], "dense_full_selection")
            self.assertEqual(dense["random_expectation_total"], 2.0)
            self.assertEqual(dense["random_lift"], 1.0)
            self.assertIn("假设这些有非 fresh 的步均可选位",
                          result["metadata"]["assumption"])
            self.assertEqual(result["step_records"], 3)
            self.assertEqual(len(result["aggregate"]), 3)
            self.assertTrue((folder / "out" / "kvdrift-budget-coverage.csv").exists())
            self.assertTrue((folder / "out" / "kvdrift-budget-coverage.json").exists())
            self.assertFalse((folder / "out" / "kvdrift-budget-coverage.jsonl").exists())


if __name__ == "__main__":
    unittest.main()

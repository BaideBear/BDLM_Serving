"""Synthetic contracts for the CPU-only strict-budget prefix opportunity audit."""
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # pragma: no cover - import style depends on the checkout root
    from experiments.t2t.serving_slots import ibs_prefix_opportunity as ibs
except ModuleNotFoundError:  # pragma: no cover
    import ibs_prefix_opportunity as ibs


def task(index=0):
    return dict(index=index, dataset="gsm8k", task_id="gsm8k/1", seed=17000 + index)


def synthetic_row(q=8, final_width=6):
    """A minimal valid B32 trace with one seen/written prefix position."""
    prompt = [7] * 34
    events = [{
        "phase": "prefill", "q_tokens": 32, "physical_forwards": 1,
        "absolute_positions": list(range(32)),
    }]
    calls = [{"batch_call_id": 0, "rows": 32, "physical_rows": 32,
              "batch": 1, "kv_rows": 32}]
    steps = (
        (0, list(range(32, 40)), 0, 2, [], [34], 1, list(range(0, 32))),
        (1, list(range(34, 40)), 2, 3, [34], [35], 1, list(range(2, 32))),
        (2, list(range(35, 41)), 3, 4, [35], [36], 1, list(range(3, 32))),
    )
    for step, positions, before, after, dirty, writes, accepted, candidate in steps:
        width = final_width if step == 2 else len(positions)
        positions = positions[:width]
        events.append({
            "phase": "optimus", "block": 1, "step": step,
            "q_tokens": width, "physical_forwards": 1,
            "physical_cap": q, "fixed_chunk": q, "mode": "ibs",
            "active_width": 32, "offer_tick": step,
            "absolute_positions": positions,
            "selected_local": [value - 32 for value in positions],
            "candidate_local": candidate,
            "selected_masked": [False] * width,
            "dirty_before": dirty, "written_absolute": writes,
            "accepted": accepted, "edited": 0, "edited_positions": [],
            "cache_deferred_t2t_positions": [],
            "cached_prefix_before": before, "cached_prefix_after": after,
            "cross_boundary_rows": 0,
        })
        calls.append({"batch_call_id": step + 1, "rows": width,
                      "physical_rows": width, "batch": 1, "kv_rows": 64})
    widths = [event["q_tokens"] for event in events[1:]]
    return {
        "model": "llada21", "policy": "optcap_ibs_8", "phase": "formal",
        "generation_cap": 2048, "index": 0, "dataset": "gsm8k",
        "task_id": "gsm8k/1", "seed": 17000,
        "prompt_ids": prompt, "prefill_rows": 32,
        "events": events, "calls": calls,
        "generation_forwards": len(widths),
        "generation_query_rows": sum(widths),
        "generation_physical_padded_rows": sum(widths),
        "max_q": max(widths), "q_histogram": {str(w): widths.count(w)
                                              for w in widths},
        "logical_iterations": len(widths),
        "status": "success", "request_failure": None,
        "output_ids": [5], "output_text": "x", "output_tokens": 1,
        "eos": True, "truncated": False,
        "extra": {"mode": "ibs", "fixed_chunk": q, "window_switches": 0,
                  "physical_caps": [q], "commit": "c",
                  "source_hashes": {"sequence.py": "a"}},
    }


class PrefixOpportunityTest(unittest.TestCase):
    def test_valid_synthetic_trace(self):
        summary = ibs.validate_request(synthetic_row(), task(),
                                       dict(policy="optcap_ibs_8", q=8))
        self.assertEqual(summary["generation_q"], 20)
        self.assertEqual(summary["calls"], 4)

    def test_physical_call_mismatch_is_rejected(self):
        row = synthetic_row()
        row["calls"][1]["rows"] = 7
        with self.assertRaises(ValueError):
            ibs.validate_request(row, task(), dict(policy="optcap_ibs_8", q=8))

    def test_pool_requires_seen_written_and_clean_unselected(self):
        row = synthetic_row()
        records = ibs.tick_records(row, dict(policy="optcap_ibs_8", q=8))
        self.assertEqual([record.pool for record in records], [0, 0, 1])
        self.assertEqual(records[-1].added, 1)
        self.assertEqual(records[-1].remaining, 2)
        self.assertEqual(records[-1].quota, 2)

    def test_quota_and_remaining_seats_limit_injection(self):
        full = synthetic_row(final_width=8)
        records = ibs.tick_records(full, dict(policy="optcap_ibs_8", q=8))
        self.assertEqual(records[-1].pool, 1)
        self.assertEqual(records[-1].remaining, 0)
        self.assertEqual(records[-1].added, 0)

        row = synthetic_row(final_width=5)
        records = ibs.tick_records(row, dict(policy="optcap_ibs_8", q=8))
        self.assertEqual(records[-1].added, 1)

    def test_stage_classification(self):
        self.assertEqual(ibs.tick_stage(0, 5, 2), "bootstrap")
        self.assertEqual(ibs.tick_stage(2, 0, 10), "stall")
        self.assertEqual(ibs.tick_stage(3, 1, 32), "terminal")
        self.assertEqual(ibs.tick_stage(1, 2, 12), "denoise")

    def test_recognized_progress_limit_failure(self):
        failure = {"type": "optimus_progress_limit", "delivered_output_tokens": 0,
                   "steps_in_block": 68, "step_limit": 68, "cached_prefix": 3}
        row = {"status": "generation_failed", "request_failure": failure,
               "extra": {"request_failure": failure}, "output_tokens": 0,
               "output_ids": [], "output_text": "", "eos": False,
               "truncated": False,
               "events": [{"active_width": 32}]}
        ibs.validate_failure(row)
        row["request_failure"] = dict(failure, delivered_output_tokens=1)
        row["extra"]["request_failure"] = row["request_failure"]
        with self.assertRaises(ValueError):
            ibs.validate_failure(row)

    def test_aggregate_cohorts_and_prefix_conditioning(self):
        records = [
            ibs.TickRecord(0, "success", 1, 1, "denoise", 6, 8, 2, 4, 1, 2, 2, 2, 2),
            ibs.TickRecord(30, "success", 1, 1, "stall", 8, 8, 4, 4, 0, 3, 0, 0, 2),
            ibs.TickRecord(89, "success", 1, 0, "bootstrap", 8, 8, 0, 8, 1, 0, 0, 0, 2),
        ]
        status = {0: "success", 30: "success", 89: "success"}
        row = ibs.aggregate(records, status, range(0, 90), ("success",),
                            "all", "all")
        self.assertEqual(row["requests"], 3)
        self.assertEqual(row["requests_with_opportunity"], 1)
        self.assertEqual(row["slots_added"], 2)
        self.assertEqual(row["prev_physical_q"], 14)
        self.assertEqual(row["prev_forwards"], 2)
        nonempty = ibs.aggregate(records, status, range(0, 90), ("success",),
                                 "all", "nonempty")
        self.assertEqual(nonempty["ticks"], 2)
        empty = ibs.aggregate(records, status, range(0, 90), ("success",),
                              "all", "empty")
        self.assertEqual(empty["slots_added"], 0)

    def test_success_only_scope_reports_excluded_failures(self):
        records = [ibs.TickRecord(0, "success", 1, 1, "denoise", 8, 8, 0, 8,
                                  0, 0, 0, 0, 2)]
        status = {0: "success", 1: "generation_failed"}
        row = ibs.aggregate(records, status, range(0, 2), ("success",),
                            "all", "all")
        self.assertEqual(row["requests"], 1)
        self.assertEqual(row["excluded_generation_failed"], 1)

    def test_missing_marker_excludes_arm(self):
        raw = Path(tempfile.mkdtemp())
        folder = raw / "arm"
        folder.mkdir()
        policy = "optcap_ibs_16"
        (folder / "run.json").write_text(json.dumps({
            "model": "llada21", "suite": "frozen90", "samples_sha256": "sha",
            "block": 32, "batch_size": 1, "hard_q_cap": True,
            "cap_clock": "all_decoder_forwards_request_local",
            "cap_excludes": "prompt_prefill_only", "policies": [policy],
        }))
        for index in range(55):
            (folder / ("%02d-%s.json" % (index, policy))).write_text("{}")
        info = ibs.inspect_arm(raw, dict(policy=policy, q=16, folder="arm"),
                               "sha", {index: task(index) for index in range(90)})
        self.assertFalse(info["complete"])
        self.assertEqual(info["request_files_present"], 55)
        self.assertIn("Missing completion marker", info["reason"])

    def test_marker_with_missing_files_excludes_arm(self):
        raw = Path(tempfile.mkdtemp())
        folder = raw / "arm"
        folder.mkdir()
        policy = "optcap_ibs_8"
        (folder / "run.json").write_text(json.dumps({
            "model": "llada21", "suite": "frozen90", "samples_sha256": "sha",
            "block": 32, "batch_size": 1, "hard_q_cap": True,
            "cap_clock": "all_decoder_forwards_request_local",
            "cap_excludes": "prompt_prefill_only", "policies": [policy],
        }))
        (folder / ("complete-%s.json" % policy)).write_text(json.dumps(
            {"policy": policy, "requests": 90}))
        (folder / ("00-%s.json" % policy)).write_text("{}")
        info = ibs.inspect_arm(raw, dict(policy=policy, q=8, folder="arm"),
                               "sha", {index: task(index) for index in range(90)})
        self.assertFalse(info["complete"])
        self.assertIn("Missing request file", info["reason"])


if __name__ == "__main__":
    unittest.main()

"""Synthetic runner-format fixtures. No files, model, or third-party dependencies."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import t2t_gate_summary as summary


NEAR = "t2t_q4_f1_m1_near_miss"
RANDOM = "t2t_q4_f1_m1_gate_random"
RUN, OTHER = Path("/synthetic-run"), Path("/another-synthetic-run")


def event(stage, checked, block=1):
    q = len(checked)
    return dict(
        stage=stage, phase=summary.STAGE_PHASE[stage], block=block, q_budget=4,
        absolute_positions=list(checked), q_tokens=q, physical_q=q, actual_q=q,
        layer_physical_q=[q, q], F=1,
    )


def denoise(stage, checked, step, cats, *, block=1, strict=(), previous=(), edited=None,
            reserved=(), near=0, nonfresh=0, reserved_near=0, released=0, ages=None):
    e = event(stage, checked, block)
    strict, reserved = list(strict), list(reserved)
    e.update(step_in_block=step, edited=len(strict) if edited is None else edited,
             gate=dict(
                 available_near_miss_count=near, available_nonfresh_count=nonfresh,
                 reserved_positions=reserved, reserved_near_miss_count=reserved_near,
                 released_edit_slots_to_mask=released, previous_strict_edit_positions=list(previous),
                 strict_edit_positions=strict,
                 reserved_strict_edit_positions=[i for i in reserved if i in strict],
                 selected_category_counts=dict(zip(summary.CATEGORIES, cats)),
                 selected_evidence_age=ages if ages is not None else [None if step == 1 else 1] * len(checked),
             ))
    return e


def case(index, policy):
    dense = range(32, 64)
    events = [event("prefill", range(32), block=0),
              denoise("bootstrap", dense, 1, (2, 30, 0, 0, 0))]
    if index != 29:
        events.append(denoise("selected", [34, 36, 37, 38], 2, (0, 3, 1, 0, 0), released=2))
        if policy == NEAR:
            events.append(denoise("selected", [34, 37, 39, 40], 3, (0, 2, 1, 1, 0),
                                  strict=(34,), reserved=(34,), near=1, nonfresh=4,
                                  reserved_near=1, released=1, ages=[1, 1, 2, 2]))
        elif index == 0:
            events.append(denoise("selected", [37, 39, 40, 41], 3, (0, 1, 1, 0, 2),
                                  strict=(37,), reserved=(40, 41), near=2, nonfresh=5,
                                  ages=[1, 2, 2, 2]))
    previous = events[-1]["gate"]["strict_edit_positions"]
    events.extend([
        denoise("verify", dense, len(events), (2, 0, 2, 0, 28), near=0, nonfresh=28,
                strict=(35,), previous=previous, edited=2),
        denoise("verify", dense, len(events) + 1, (2, 0, 1, 0, 29), near=0, nonfresh=29,
                previous=(35,)),
        event("commit", dense),
    ])
    # Denoise clocks and evidence reset at the next block. This case has no selected passes.
    if index == 29:
        events.extend([
            denoise("bootstrap", range(64, 96), 1, (0, 32, 0, 0, 0), block=2),
            denoise("verify", range(64, 96), 2, (0, 0, 32, 0, 0), block=2),
            event("commit", range(64, 96), block=2),
        ])
    return package_case(index, policy, events)


def package_case(index, policy, events):
    generation = [e for e in events if e["phase"] != "prefill"]
    q = sum(e["physical_q"] for e in generation)
    return dict(index=index, dataset=("gsm8k", "math500", "humaneval")[index // 10],
                task_id=f"case-{index}", seed=17000 + index, model="llada21", policy=policy,
                status="success", output_ids=[888888 + (policy == RANDOM and index == 0)],
                output_text="raw output must not appear", events=events,
                generation_forwards=len(generation), generation_query_rows=q,
                generation_physical_padded_rows=q, prefill_rows=32,
                t2t_edits=sum(e.get("edited", 0) for e in events))


def files(folder, *policies):
    arms = [dict(budget=4, fresh=1, mask=1, selection=p.split("_", 4)[-1]) for p in policies]
    entries = {
        folder / "run.json": dict(model="llada21", block=32, batch_size=1, cap=2048,
                                  policies=list(policies), arms=arms),
        folder / "qualified.json": dict(model="llada21", full_width=32, matched=30),
    }
    for policy in policies:
        entries[folder / f"complete-{policy}.json"] = dict(policy=policy, requests=30)
        entries.update({folder / f"{i:02d}-{policy}.json": case(i, policy) for i in range(30)})
    return entries


class GateSummaryTest(unittest.TestCase):
    def setUp(self):
        self.entries = files(RUN, NEAR, RANDOM)
        reader = patch.object(Path, "read_text", autospec=True, side_effect=self.read)
        reader.start()
        self.addCleanup(reader.stop)

    def read(self, path, *, encoding):
        if path not in self.entries:
            raise FileNotFoundError(2, "No such file", str(path))
        return json.dumps(self.entries[path])

    def test_cli_aggregates_and_compares_own_trajectories(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            summary.main(["--run", str(RUN)])
        text = stdout.getvalue()
        self.assertEqual(len(text.splitlines()), 1)
        self.assertNotRegex(text, r"88888[89]|raw output|case-0|absolute_positions")
        report = json.loads(text)
        self.assertEqual(report["scope"], "online_own_trajectories")
        arms = {a["policy"]: a for a in report["arms"]}
        near, random = arms[NEAR], arms[RANDOM]
        self.assertEqual(near["requests"], 30)
        self.assertEqual(near["selected_physical_checked"], 232)
        self.assertEqual(random["selected_physical_checked"], 120)
        self.assertEqual(near["denoise_physical_checked"], 3176)
        self.assertEqual(near["generation_physical_query_rows"], 4168)
        self.assertEqual(near["work_by_stage"]["prefill"]["physical_query_rows"], 960)
        self.assertEqual(near["work_by_stage"]["verify"]["strict_edits"], 30)
        self.assertEqual(near["selected_checked_by_category"],
                         dict(prompt=0, mask=145, fresh=58, near_miss=29, other_resolved=0))
        self.assertEqual((near["reserved_seats"], near["reserved_near_miss_seats"],
                          near["strict_edits"], near["native_edited"], near["reserved_strict_hits"],
                          near["released_edit_slots_to_mask"]), (29, 29, 59, 89, 29, 87))
        self.assertEqual((random["reserved_seats"], random["reserved_near_miss_seats"],
                          random["strict_edits"], random["reserved_strict_hits"],
                          random["released_edit_slots_to_mask"]), (2, 0, 31, 0, 58))
        self.assertEqual(near["reserved_hit_rate"], 1)
        self.assertEqual(near["reserved_by_observation_age"]["previous"],
                         dict(seats=29, hits=29, precision=1))
        self.assertEqual(random["reserved_by_observation_age"]["two_steps"],
                         dict(seats=2, hits=0, precision=0))
        self.assertIsNone(near["reserved_by_observation_age"]["unknown"]["precision"])
        self.assertEqual(random["reserved_share_of_observed_selected_strict_edits"], 0)
        comparison, = report["online_own_trajectory_comparisons"]
        delta = comparison["totals_near_miss_minus_gate_random"]
        self.assertEqual((delta["reserved_seats"], delta["reserved_strict_hits"],
                          delta["released_edit_slots_to_mask"]), (27, 29, 29))
        self.assertEqual(comparison["cases_with_different_counts"]["strict_edits"], 28)
        self.assertEqual(comparison["cases_with_different_output_ids"], 1)
        self.assertEqual(comparison["cases_with_different_denoise_check_path"], 29)
        self.assertEqual(comparison["cases_with_different_strict_edit_path"], 29)
        self.assertEqual(comparison["cases_with_different_reservation_path"], 29)

    def test_split_runs_pair_only_same_quotas(self):
        self.entries = files(RUN, NEAR) | files(OTHER, RANDOM)
        report = summary.summarize_runs([OTHER, RUN])
        self.assertEqual(len(report["online_own_trajectory_comparisons"]), 1)
        other_policy = "t2t_q4_f1_m0_gate_random"
        entries = files(OTHER, other_policy)
        entries[OTHER / "run.json"]["arms"][0]["mask"] = 0
        for i in range(30):
            for e in entries[OTHER / f"{i:02d}-{other_policy}.json"]["events"]:
                if e["stage"] == "selected":
                    g = e["gate"]
                    g["released_edit_slots_to_mask"] = 3 - len(g["reserved_positions"])
        self.entries = files(RUN, NEAR) | entries
        self.assertEqual(summary.summarize_runs([RUN, OTHER])["online_own_trajectory_comparisons"], [])

    def test_same_counts_can_hide_different_reservations(self):
        for i in range(30):
            row = case(i, NEAR)
            row["policy"] = RANDOM
            self.entries[RUN / f"{i:02d}-{RANDOM}.json"] = row
        g = self.entries[RUN / f"00-{RANDOM}.json"]["events"][3]["gate"]
        g.update(reserved_positions=[37], strict_edit_positions=[37],
                 reserved_strict_edit_positions=[37])
        verify = self.entries[RUN / f"00-{RANDOM}.json"]["events"][4]["gate"]
        verify["previous_strict_edit_positions"] = [37]
        result, = summary.summarize_runs([RUN])["online_own_trajectory_comparisons"]
        self.assertFalse(any(result["cases_with_different_counts"].values()))
        self.assertEqual(result["cases_with_different_denoise_check_path"], 0)
        self.assertEqual(result["cases_with_different_reservation_path"], 1)
        self.assertEqual(result["cases_with_different_strict_edit_path"], 1)

    def test_zero_selected_rows_have_null_rates(self):
        self.entries = files(RUN, NEAR)
        for i in range(30):
            row = self.entries[RUN / f"{i:02d}-{NEAR}.json"]
            events = [e for e in row["events"] if e["stage"] != "selected"]
            verifies = [e for e in events if e["stage"] == "verify" and e["block"] == 1]
            for step, e in enumerate(verifies, 2):
                e["step_in_block"] = step
            verifies[0]["gate"]["previous_strict_edit_positions"] = []
            self.entries[RUN / f"{i:02d}-{NEAR}.json"] = package_case(i, NEAR, events)
        near, = summary.summarize_runs([RUN])["arms"]
        self.assertEqual((near["selected_physical_checked"], near["reserved_seats"]), (0, 0))
        self.assertIsNone(near["reserved_hit_rate"])
        self.assertIsNone(near["reserved_share_of_observed_selected_strict_edits"])

    def test_unmarked_arms_skip_only_when_requested(self):
        del self.entries[RUN / f"complete-{RANDOM}.json"]
        with self.assertRaisesRegex(FileNotFoundError, "complete-"):
            summary.summarize_runs([RUN])
        report = summary.summarize_runs([RUN], only_complete=True)
        self.assertEqual([a["policy"] for a in report["arms"]], [NEAR])
        self.assertEqual(report["online_own_trajectory_comparisons"], [])
        del self.entries[RUN / f"complete-{NEAR}.json"]
        with self.assertRaisesRegex(ValueError, "No complete"):
            summary.summarize_runs([RUN], only_complete=True)

    def test_marked_arm_with_missing_case_fails_without_partial_stdout(self):
        del self.entries[RUN / f"29-{RANDOM}.json"]
        stderr, stdout = io.StringIO(), io.StringIO()
        with redirect_stderr(stderr), redirect_stdout(stdout), self.assertRaises(SystemExit) as exit:
            summary.main(["--run", str(RUN), "--only-complete"])
        self.assertEqual(exit.exception.code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn(f"29-{RANDOM}.json", stderr.getvalue())

    def test_marker_and_qualification_are_validated(self):
        self.entries[RUN / f"complete-{RANDOM}.json"]["requests"] = 29
        with self.assertRaisesRegex(ValueError, "invalid completion marker"):
            summary.summarize_runs([RUN], only_complete=True)
        self.entries[RUN / "qualified.json"]["matched"] = 29
        with self.assertRaisesRegex(ValueError, "expected full-width qualification"):
            summary.summarize_runs([RUN])

    def test_merge_rejects_wrong_seed_and_duplicate_policy(self):
        self.entries = files(RUN, NEAR) | files(OTHER, RANDOM)
        self.entries[OTHER / f"04-{RANDOM}.json"]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "identities or seeds differ"):
            summary.summarize_runs([RUN, OTHER])
        self.entries = files(RUN, NEAR) | files(OTHER, NEAR)
        with self.assertRaisesRegex(ValueError, "Duplicate completed policy"):
            summary.summarize_runs([RUN, OTHER])

    def test_malformed_event_abi_and_accounting(self):
        def update_event(row, **updates):
            row["events"][3].update(updates)

        def update_gate(row, **updates):
            row["events"][3]["gate"].update(updates)

        for corrupt, message in (
            (lambda r: update_event(r, layer_physical_q=[4, 5]), "Physical query audit"),
            (lambda r: r["events"][3].pop("gate"), "'gate'"),
            (lambda r: update_gate(r, reserved_positions=[99]), "Invalid reservation"),
            (lambda r: update_gate(r, reserved_strict_edit_positions=[]), "reserved hit"),
            (lambda r: update_gate(r, previous_strict_edit_positions=[35]), "Previous strict edits"),
            (lambda r: update_gate(r, released_edit_slots_to_mask=2), "gate budget"),
            (lambda r: update_gate(r, selected_evidence_age=[1]), "evidence ages"),
            (lambda r: r.update(generation_physical_padded_rows=999), "generation_physical_padded_rows"),
        ):
            with self.subTest(message=message):
                path = RUN / f"00-{NEAR}.json"
                self.entries[path] = case(0, NEAR)
                corrupt(self.entries[path])
                with self.assertRaisesRegex(ValueError, message) as raised:
                    summary.summarize_runs([RUN])
                self.assertIn(path.name, str(raised.exception))


if __name__ == "__main__":
    unittest.main()

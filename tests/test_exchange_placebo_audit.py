import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.exchange_placebo_audit import (  # noqa: E402
    ArmData,
    audit_paths,
    audit_request,
    main,
    parse_policy,
)


WIDTH = 32
MASK_SLOTS = {4: 3, 8: 7}


def policy(budget, kind):
    return "t2t_q%d_f0_m%d_hard_exchange_random_%s" % (
        budget, MASK_SLOTS[budget], kind)


def ranked_policy(budget):
    return "t2t_q%d_f0_m%d_hard_exchange_edit4" % (
        budget, MASK_SLOTS[budget])


def arm_data(budget, kind):
    name = policy(budget, kind)
    parsed = parse_policy(name)
    return ArmData(policy=name, budget=budget, fresh=parsed["fresh"],
                   mask=parsed["mask"], selection=parsed["selection"],
                   kind=kind, width=WIDTH, model="llada21",
                   sources=["<memory>"])


def reference_data(budget):
    name = ranked_policy(budget)
    parsed = parse_policy(name)
    return ArmData(policy=name, budget=budget, fresh=parsed["fresh"],
                   mask=parsed["mask"], selection=parsed["selection"],
                   kind="ranked", width=WIDTH, model="llada21",
                   sources=["<memory>"])


def bootstrap(budget=4, block=1, clock=2):
    offset = block * WIDTH
    positions = list(range(offset, offset + WIDTH))
    return {
        "phase": "denoise", "block": block, "q_start": offset,
        "q_end": offset + WIDTH, "q_tokens": WIDTH,
        "prefix_kv_length": offset, "store_kv": False,
        "context_version": 0, "stage": "bootstrap",
        "absolute_positions": positions, "actual_q": WIDTH,
        "physical_q": WIDTH, "layer_physical_q": [WIDTH, WIDTH],
        "F": 1, "visible_kv": offset + WIDTH, "q_budget": budget,
        "approximate_same_block_kv": False, "hard_clock": clock,
        "mandatory_fresh": 0, "reserved_extra_edit_positions": [],
        "reserved_strict_edit_positions": [],
        "overflow": max(0, WIDTH - budget), "hard_strict_edits": 0,
        "masked_before": WIDTH, "selected_masked_before": WIDTH,
        "accepted": 3, "edited": 0, "masked_after": WIDTH - 3,
        "step_in_block": 1, "post_steps": 0,
    }


def selected_event(budget, local, *, block=1, clock=3, reserved=(),
                   strict=(), doc=None, edited=0, hard_strict=0,
                   masked_before=None, masked_after=None,
                   selected_masked_before=1, accepted=1,
                   mandatory_fresh=0):
    offset = block * WIDTH
    positions = sorted(offset + position for position in local)
    if masked_before is None:
        masked_before = budget + 4
    if masked_after is None:
        masked_after = masked_before - accepted
    event = {
        "phase": "denoise", "block": block, "q_start": positions[0],
        "q_end": positions[-1] + 1, "q_tokens": len(local),
        "prefix_kv_length": offset, "store_kv": False,
        "context_version": 0, "stage": "selected",
        "absolute_positions": positions, "actual_q": len(local),
        "physical_q": len(local),
        "layer_physical_q": [len(local), len(local)],
        "F": 1, "visible_kv": offset + WIDTH, "q_budget": budget,
        "approximate_same_block_kv": True, "hard_clock": clock,
        "mandatory_fresh": mandatory_fresh,
        "reserved_extra_edit_positions": sorted(reserved),
        "reserved_strict_edit_positions": sorted(strict),
        "overflow": max(0, len(local) - budget),
        "hard_strict_edits": hard_strict,
        "masked_before": masked_before,
        "selected_masked_before": selected_masked_before,
        "accepted": accepted, "edited": edited, "masked_after": masked_after,
        "step_in_block": 2, "post_steps": 0,
    }
    if doc is not None:
        event["placebo_exchange"] = doc
    return event


def result(events, *, policy_name, index=0, output_ids=(11, 12, 13),
           forwards=None, query_rows=None):
    query_rows = (sum(event.get("physical_q", 0) for event in events)
                  if query_rows is None else query_rows)
    forwards = len(events) if forwards is None else forwards
    return {
        "status": "success", "output_ids": list(output_ids),
        "output_text": "", "output_tokens": len(output_ids),
        "generation_forwards": forwards,
        "generation_query_rows": query_rows,
        "generation_physical_padded_rows": query_rows,
        "max_q": 4, "tpf": 0.1, "token_efficiency": 0.2, "eos": True,
        "truncated": False, "prefill_rows": WIDTH,
        "forwards_by_phase": {"prefill": 1, "bootstrap": 1, "selected": 1},
        "q_by_phase": {"prefill": WIDTH, "bootstrap": WIDTH, "selected": 4},
        "q_histogram": {"4": 1}, "t2t_edits": 0, "events": events,
        "index": index, "task_id": "toy/%d" % index, "dataset": "toy",
        "seed": 17000 + index, "model": "llada21", "policy": policy_name,
    }


def scenario(budget=4, *, victim="mask", collision=False, block=1,
             fresh=(), index=0):
    base = list(range(budget))
    victim_position = budget - 1 if victim == "mask" else budget - 2
    replacement = budget
    alternate = budget + 1
    ranked = sorted((set(base) - {victim_position}) | {replacement})
    executed = ranked if collision else sorted(
        (set(base) - {victim_position}) | {alternate})
    victims = [victim_position]
    masked_victims = [victim_position] if victim == "mask" else []
    pool = list(range(budget, budget + 4))
    offset = block * WIDTH
    reserved_candidate = replacement if collision else alternate
    reserved = [offset + reserved_candidate]
    strict = [offset + reserved_candidate] if victim == "old" else []
    edited = 1 if victim == "old" else 0
    hard_strict = edited
    doc = {
        "base": base, "ranked": ranked, "executed": executed,
        "victims": victims, "pool": pool, "fresh": sorted(fresh),
        "masked_victims": masked_victims,
    }
    new_event = selected_event(
        budget, executed, block=block, reserved=reserved, strict=strict,
        doc=doc, edited=edited, hard_strict=hard_strict,
        mandatory_fresh=len(fresh))
    old_event = selected_event(
        budget, ranked, block=block, reserved=[offset + replacement],
        strict=[offset + replacement] if victim == "old" else [],
        edited=edited, hard_strict=hard_strict, mandatory_fresh=len(fresh))
    new = result([bootstrap(budget, block), new_event],
                 policy_name=policy(budget, "a"), index=index)
    old = result([bootstrap(budget, block), old_event],
                 policy_name=ranked_policy(budget), index=index)
    return new, old


def run_specs(*items):
    return list(items)


def write_run(root, specs, *, model="llada21", block=WIDTH,
              qualified=True, complete=True, matched=30,
              completed_requests=30):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    policies, arms = [], []
    for name, results in specs:
        parsed = parse_policy(name)
        policies.append(name)
        arms.append({key: parsed[key]
                     for key in ("budget", "fresh", "mask", "selection")})
        for index, data in results.items():
            (root / ("%02d-%s.json" % (index, name))).write_text(
                json.dumps(data))
    (root / "run.json").write_text(json.dumps({
        "model": model, "block": block, "policies": policies, "arms": arms,
    }))
    if qualified:
        (root / "qualified.json").write_text(json.dumps({
            "model": model, "full_width": block, "matched": matched,
        }))
    if complete:
        for name, _ in specs:
            (root / ("complete-%s.json" % name)).write_text(json.dumps({
                "policy": name, "requests": completed_requests,
            }))
    return root


def manifest_for(*maps):
    merged = {}
    for mapping in maps:
        merged.update(mapping)
    return {"samples": [
        {"index": index, "task_id": data["task_id"],
         "dataset": data["dataset"], "seed": data["seed"]}
        for index, data in sorted(merged.items())]}


class ExchangePlaceboAuditTest(unittest.TestCase):
    def audit(self, new, old, budget=4, kind="a", index=0):
        return audit_request(new, old, arm_data(budget, kind),
                             reference_data(budget), index, "test")

    def test_clean_pair_records_fork_and_swap_type(self):
        new, old = scenario()
        row, violations = self.audit(new, old)
        self.assertEqual(violations, [])
        self.assertEqual(row["forked"], 1)
        self.assertEqual(row["nondivergent"], 0)
        self.assertEqual(row["first_fork"], 1)
        self.assertEqual(row["first_q_split"], 1)
        self.assertEqual(row["first_swap"], 1)
        self.assertEqual(row["first_fork_victim"], "mask")
        self.assertEqual(row["first_fork_ranked_replay"], "ok")
        self.assertEqual(row["swaps_total"], 1)
        self.assertEqual(row["swaps_mask"], 1)
        self.assertEqual(row["swaps_old"], 0)
        self.assertEqual(row["collisions"], 0)
        self.assertEqual(row["direct_strict_hits"], 0)
        self.assertEqual(row["fork_new_admissions"], 1)
        self.assertEqual(row["fork_ranked_admissions"], 1)
        self.assertEqual(row["fork_new_accepted"], 1)
        self.assertEqual(row["fork_new_selected_masked_before"], 1)
        self.assertEqual(row["first_fork_new_strict_hits"], 0)
        self.assertEqual(row["first_fork_ranked_strict_hits"], 0)
        self.assertEqual(row["first_fork_new_reserved_rows"], "37")
        self.assertEqual(row["first_fork_ranked_reserved_rows"], "36")

    def test_old_victim_and_direct_strict_hit(self):
        new, old = scenario(victim="old")
        row, violations = self.audit(new, old)
        self.assertEqual(violations, [])
        self.assertEqual(row["first_fork_victim"], "old")
        self.assertEqual(row["swaps_old"], 1)
        self.assertEqual(row["direct_strict_hits"], 1)
        self.assertEqual(row["direct_strict_rows"], "37")
        self.assertEqual(row["first_fork_new_strict_hits"], 1)
        self.assertEqual(row["first_fork_ranked_strict_hits"], 1)
        self.assertEqual(row["first_fork_new_strict_rows"], "37")
        self.assertEqual(row["first_fork_ranked_strict_rows"], "36")

    def test_collision_is_not_a_physical_fork(self):
        new, old = scenario(collision=True)
        row, violations = self.audit(new, old)
        self.assertEqual(violations, [])
        self.assertEqual(row["forked"], 0)
        self.assertEqual(row["nondivergent"], 1)
        self.assertEqual(row["first_fork"], "")
        self.assertEqual(row["first_q_split"], "")
        self.assertEqual(row["first_fork_victim"], "none")
        self.assertEqual(row["outputs_identical"], 1)
        self.assertEqual(row["physical_fq_identical"], 1)
        self.assertEqual(row["collisions"], 1)
        self.assertEqual(row["swap_events"], 1)

    def test_diagnostic_events_and_fields_are_stripped(self):
        new, old = scenario()
        old["events"].insert(1, {
            "stage": "diagnostic", "diagnostic": True, "hard_clock": 99,
            "absolute_positions": [0], "screened_candidates": 4,
            "screen_hits": 1,
        })
        old["events"][0]["trajectory_sha256"] = "old"
        new["events"][0]["trajectory_sha256"] = "new"
        row, violations = self.audit(new, old)
        self.assertEqual(violations, [])
        self.assertEqual(row["first_fork"], 1)

    def test_pre_divergence_history_mismatch_is_flagged(self):
        new, old = scenario()
        old["events"][0]["hard_clock"] = 99
        row, violations = self.audit(new, old)
        self.assertTrue(any("pre-divergence history mismatch" in item
                            for item in violations))
        self.assertEqual(row["history_mismatch"], 0)
        self.assertEqual(row["first_fork"], 0)
        self.assertEqual(row["first_q_split"], 1)

    def test_physical_rows_must_follow_executed_locals(self):
        new, old = scenario()
        new["events"][-1]["absolute_positions"][-1] = WIDTH + 6
        _, violations = self.audit(new, old)
        self.assertTrue(any("reconstruct physical rows" in item
                            for item in violations))

    def test_ranked_locals_must_replay_archived_positions(self):
        new, old = scenario()
        old["events"][-1]["absolute_positions"][-1] = WIDTH + 9
        _, violations = self.audit(new, old)
        self.assertTrue(any("ranked locals do not replay archived positions"
                            in item for item in violations))

    def test_ranked_reserved_rows_must_replay(self):
        new, old = scenario()
        old["events"][-1]["reserved_extra_edit_positions"] = [WIDTH + 9]
        _, violations = self.audit(new, old)
        self.assertTrue(any("ranked locals do not replay archived reserved rows"
                            in item for item in violations))

    def test_first_q_split_without_document_is_flagged(self):
        new, old = scenario()
        del new["events"][-1]["placebo_exchange"]
        _, violations = self.audit(new, old)
        self.assertTrue(any("first Q split without placebo_exchange document"
                            in item for item in violations))

    def test_fresh_seats_are_protected(self):
        new, old = scenario(victim="old", fresh=(2,))
        _, violations = self.audit(new, old)
        self.assertTrue(any("fresh outside executed" in item
                            or "fresh displaced" in item for item in violations))

    def test_candidates_must_come_from_pool(self):
        new, old = scenario()
        doc = new["events"][-1]["placebo_exchange"]
        doc["executed"] = [0, 1, 2, 13]
        new["events"][-1]["absolute_positions"] = [32, 33, 34, 45]
        new["events"][-1]["reserved_extra_edit_positions"] = [45]
        _, violations = self.audit(new, old)
        self.assertTrue(any("executed replacements outside pool" in item
                            for item in violations))

    def test_masked_victims_must_be_victims(self):
        new, old = scenario()
        new["events"][-1]["placebo_exchange"]["masked_victims"] = [9]
        _, violations = self.audit(new, old)
        self.assertTrue(any("masked_victims outside victims" in item
                            for item in violations))

    def test_reserved_rows_and_strict_hits_overlap(self):
        new, old = scenario()
        event = new["events"][-1]
        event["reserved_extra_edit_positions"] = [32, 37]
        event["reserved_strict_edit_positions"] = [32]
        _, violations = self.audit(new, old)
        self.assertTrue(any("reserved rows are not the executed replacements"
                            in item for item in violations))
        self.assertTrue(any("reserved rows overlap base" in item
                            for item in violations))

    def test_admission_chain_checked_for_ranked_arm(self):
        new, old = scenario()
        old["events"][-1].update(accepted=0, selected_masked_before=2)
        _, violations = self.audit(new, old)
        self.assertTrue(any("admission chain" in item
                            for item in violations))

    def test_admission_revocations_are_counted(self):
        new, old = scenario()
        new["events"][-1].update(edited=2, hard_strict_edits=1,
                                 masked_before=6, masked_after=6)
        old["events"][-1].update(edited=2, hard_strict_edits=1,
                                 masked_before=6, masked_after=6)
        row, violations = self.audit(new, old)
        self.assertEqual(violations, [])
        self.assertEqual(row["fork_new_revocations"], 1)
        self.assertEqual(row["fork_new_admissions"], 1)
        self.assertEqual(row["fork_ranked_revocations"], 1)
        self.assertEqual(row["fork_ranked_admissions"], 1)

    def test_negative_revocations_are_flagged(self):
        new, old = scenario()
        new["events"][-1].update(edited=0, hard_strict_edits=1)
        _, violations = self.audit(new, old)
        self.assertTrue(any("negative revocations" in item
                            for item in violations))

    def test_nondivergent_outputs_and_physical_fq_must_match(self):
        new, old = scenario(collision=True)
        new["output_ids"] = [99]
        _, violations = self.audit(new, old)
        self.assertTrue(any("nondivergent output mismatch" in item
                            for item in violations))

        new, old = scenario(collision=True)
        new["generation_query_rows"] += 1
        _, violations = self.audit(new, old)
        self.assertTrue(any("nondivergent physical F/Q mismatch" in item
                            for item in violations))

    def test_after_fork_paths_are_not_compared(self):
        new, old = scenario()
        new["events"].append(selected_event(
            4, [0, 1, 2, 9], clock=4, reserved=[41],
            masked_after=4, selected_masked_before=1))
        _, violations = self.audit(new, old)
        self.assertEqual(violations, [])

    def test_absolute_offset_uses_block_times_width(self):
        new, old = scenario(block=2)
        row, violations = self.audit(new, old)
        self.assertEqual(violations, [])
        self.assertEqual(row["first_fork_victim"], "mask")
        new["events"][-1]["absolute_positions"][-1] = WIDTH + 5
        _, violations = self.audit(new, old)
        self.assertTrue(any("outside block" in item for item in violations))

    def test_q8_is_supported(self):
        new, old = scenario(budget=8)
        row, violations = self.audit(new, old, budget=8)
        self.assertEqual(violations, [])
        self.assertEqual(row["budget"], 8)
        self.assertEqual(row["first_fork"], 1)
        self.assertEqual(row["first_fork_base_size"], 8)
        self.assertEqual(row["first_fork_pool_size"], 4)

    def test_width_violation(self):
        new, old = scenario()
        new["events"][-1]["placebo_exchange"]["base"] = [0, 1, 2, WIDTH]
        _, violations = self.audit(new, old)
        self.assertTrue(any("outside width" in item for item in violations))

    def test_exchange_rate_table_must_match(self):
        new, old = scenario()
        arm = arm_data(4, "a")
        reference = reference_data(4)
        arm.rates = "new"
        reference.rates = "old"
        _, violations = audit_request(new, old, arm, reference, 0, "test")
        self.assertTrue(any("exchange-rate tables" in item
                            for item in violations))

    def test_multiple_run_directories_and_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def named(value, budget, kind):
                value = dict(value)
                value["policy"] = policy(budget, kind)
                return value

            q4 = {index: scenario(4, index=index)[0]
                  for index in range(2)}
            q4_old = {index: scenario(4, index=index)[1]
                      for index in range(2)}
            q8 = {index: scenario(8, index=index)[0]
                  for index in range(2)}
            q8_old = {index: scenario(8, index=index)[1]
                      for index in range(2)}
            q4b = {index: named(value, 4, "b") for index, value in q4.items()}
            q8a = {index: named(value, 8, "a") for index, value in q8.items()}
            q8b = {index: named(value, 8, "b") for index, value in q8.items()}
            archived = write_run(
                root / "archived",
                run_specs((ranked_policy(4), q4_old),
                          (ranked_policy(8), q8_old)))
            q4a = write_run(root / "q4-a", run_specs((policy(4, "a"), q4)))
            q4b_path = write_run(
                root / "q4-b", run_specs((policy(4, "b"), q4b)))
            q8both = write_run(
                root / "q8-both",
                run_specs((policy(8, "a"), q8a), (policy(8, "b"), q8b)))
            rows, totals, summary = audit_paths(
                [archived], [q4a, q4b_path, q8both], budgets=(4, 8),
                manifest=manifest_for(q4, q8))
            self.assertEqual(len(rows), 8)
            self.assertEqual(summary["requests"], 8)
            self.assertEqual(summary["forks"], 8)
            self.assertEqual(summary["violations"], 0)
            overall = [row for row in totals if row["group"] == "all"][0]
            self.assertEqual(overall["requests"], 8)
            self.assertEqual(overall["swaps_total"], 8)
            self.assertEqual(overall["swaps_mask"], 8)
            self.assertEqual(overall["first_fork_mask"], 8)
            self.assertEqual(overall["fork_at_first_swap"], 8)
            groups = {row["group"] for row in totals}
            self.assertIn("budget=4/random_a", groups)
            self.assertIn("budget=8/random_b", groups)

    def test_cli_writes_request_and_total_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            new, old = scenario()
            other = dict(new)
            other["policy"] = policy(4, "b")
            archived = write_run(
                root / "archived", run_specs((ranked_policy(4), {0: old})))
            run = write_run(root / "run", run_specs(
                (policy(4, "a"), {0: new}), (policy(4, "b"), {0: other})))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(manifest_for({0: new})))
            out = root / "audit.csv"
            code = main(["--ranked", str(archived), "--run", str(run),
                         "--out", str(out), "--manifest", str(manifest),
                         "--budget", "4"])
            self.assertEqual(code, 0)
            with out.open() as stream:
                rows = list(csv.DictReader(stream))
            kinds = [row["row_type"] for row in rows]
            self.assertEqual(kinds.count("request"), 2)
            self.assertIn("total", kinds)
            request = [row for row in rows if row["row_type"] == "request"
                       and row["kind"] == "random_a"][0]
            self.assertEqual(request["first_fork_victim"], "mask")
            self.assertEqual(request["violations"], "")
            self.assertNotIn("/", request["run"])

    def test_cli_returns_nonzero_for_violation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            new, old = scenario(collision=True)
            new["output_ids"] = [99]
            other = dict(new)
            other["policy"] = policy(4, "b")
            archived = write_run(
                root / "archived", run_specs((ranked_policy(4), {0: old})))
            run = write_run(root / "run", run_specs(
                (policy(4, "a"), {0: new}), (policy(4, "b"), {0: other})))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(manifest_for({0: new})))
            code = main(["--ranked", str(archived), "--run", str(run),
                         "--out", str(root / "audit.csv"),
                         "--manifest", str(manifest), "--budget", "4"])
            self.assertEqual(code, 1)

    def test_missing_request_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            new0, old0 = scenario(index=0)
            _, old1 = scenario(index=1)
            new0b = dict(new0)
            new0b["policy"] = policy(4, "b")
            archived = write_run(
                root / "archived",
                run_specs((ranked_policy(4), {0: old0, 1: old1})))
            run = write_run(root / "run", run_specs((policy(4, "a"), {0: new0})))
            run_b = write_run(root / "run-b",
                              run_specs((policy(4, "b"), {0: new0b})))
            rows, _, summary = audit_paths(
                [archived], [run, run_b],
                budgets=(4,),
                manifest=manifest_for({0: new0, 1: old1}))
            missing = [row for row in rows if row["status"] == "missing"]
            self.assertEqual(len(missing), 2)
            self.assertEqual(summary["missing"], 2)
            self.assertGreater(summary["violations"], 0)

    def test_placeholder_arm_without_files_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, old = scenario()
            archived = write_run(
                root / "archived", run_specs((ranked_policy(4), {0: old})))
            empty = write_run(root / "empty", run_specs(
                (policy(4, "a"), {}), (policy(4, "b"), {})))
            rows, _, summary = audit_paths(
                [archived], [empty], budgets=(4,),
                manifest=manifest_for({0: old}))
            self.assertTrue(any(row["status"] == "missing" for row in rows))
            self.assertTrue(any(row["status"] == "arm" for row in rows))
            self.assertGreater(summary["violations"], 0)

    def test_missing_arm_fails_even_with_allow_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            new, old = scenario()
            archived = write_run(
                root / "archived", run_specs((ranked_policy(4), {0: old})))
            run = write_run(root / "run", run_specs((policy(4, "a"), {0: new})))
            manifest = manifest_for({0: new})
            for allow_missing in (False, True):
                with self.assertRaisesRegex(ValueError, "Missing placebo arm"):
                    audit_paths([archived], [run], allow_missing=allow_missing,
                                manifest=manifest)

    def test_non_edit4_reference_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            new, old = scenario()
            other = dict(new)
            other["policy"] = policy(4, "b")
            equal = "t2t_q4_f0_m3_hard_exchange_equal"
            old = dict(old)
            old["policy"] = equal
            archived = write_run(root / "archived",
                                 run_specs((equal, {0: old})))
            run = write_run(
                root / "run",
                run_specs((policy(4, "a"), {0: new}),
                          (policy(4, "b"), {0: other})))
            with self.assertRaisesRegex(ValueError, "edit4 reference"):
                audit_paths([archived], [run], budgets=(4,),
                            manifest=manifest_for({0: new}))

    def test_same_index_missing_on_both_sides_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            new0, old0 = scenario(index=0)
            new1, old1 = scenario(index=1)
            new0b = dict(new0)
            new0b["policy"] = policy(4, "b")
            archived = write_run(
                root / "archived", run_specs((ranked_policy(4), {0: old0})))
            run_a = write_run(root / "a",
                              run_specs((policy(4, "a"), {0: new0})))
            run_b = write_run(root / "b",
                              run_specs((policy(4, "b"), {0: new0b})))
            rows, _, summary = audit_paths(
                [archived], [run_a, run_b],
                budgets=(4,),
                manifest=manifest_for({0: new0, 1: new1}))
            missing = [row for row in rows if row["status"] == "missing"]
            self.assertEqual(len(missing), 2)
            self.assertTrue(all(row["index"] == 1 for row in missing))
            self.assertEqual(summary["missing"], 2)
            self.assertGreater(summary["violations"], 0)

    def test_unqualified_run_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            new, old = scenario()
            other = dict(new)
            other["policy"] = policy(4, "b")
            archived = write_run(
                root / "archived", run_specs((ranked_policy(4), {0: old})))
            run = write_run(
                root / "run",
                run_specs((policy(4, "a"), {0: new}),
                          (policy(4, "b"), {0: other})),
                qualified=False)
            rows, _, summary = audit_paths(
                [archived], [run], budgets=(4,),
                manifest=manifest_for({0: new}))
            self.assertGreater(summary["violations"], 0)
            self.assertTrue(any(row["status"] == "arm"
                                and "unqualified run" in row["violations"]
                                for row in rows))
            self.assertFalse(any(str(root) in row["violations"]
                                 for row in rows))

    def test_incomplete_arm_marker_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            new, old = scenario()
            other = dict(new)
            other["policy"] = policy(4, "b")
            archived = write_run(
                root / "archived", run_specs((ranked_policy(4), {0: old})))
            run = write_run(
                root / "run",
                run_specs((policy(4, "a"), {0: new}),
                          (policy(4, "b"), {0: other})),
                completed_requests=29)
            rows, _, summary = audit_paths(
                [archived], [run], budgets=(4,),
                manifest=manifest_for({0: new}))
            self.assertGreater(summary["violations"], 0)
            self.assertTrue(any("incomplete arm" in row["violations"]
                                for row in rows))

    def test_unrelated_manifest_identity_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            new, old = scenario()
            other = dict(new)
            other["policy"] = policy(4, "b")
            archived = write_run(
                root / "archived", run_specs((ranked_policy(4), {0: old})))
            run = write_run(
                root / "run",
                run_specs((policy(4, "a"), {0: new}),
                          (policy(4, "b"), {0: other})))
            wrong = manifest_for({0: new})
            wrong["samples"][0]["seed"] += 1
            rows, _, summary = audit_paths(
                [archived], [run], budgets=(4,), manifest=wrong)
            self.assertGreater(summary["violations"], 0)
            self.assertTrue(any("manifest seed mismatch" in row["violations"]
                                for row in rows))

    def test_csv_is_lf_only_and_has_no_absolute_run_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            new, old = scenario()
            other = dict(new)
            other["policy"] = policy(4, "b")
            archived = write_run(
                root / "archived", run_specs((ranked_policy(4), {0: old})))
            run = write_run(
                root / "run",
                run_specs((policy(4, "a"), {0: new}),
                          (policy(4, "b"), {0: other})))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(manifest_for({0: new})))
            out = root / "audit.csv"
            self.assertEqual(main([
                "--ranked", str(archived), "--run", str(run),
                "--out", str(out), "--manifest", str(manifest),
                "--budget", "4"]), 0)
            data = out.read_bytes()
            self.assertNotIn(b"\r\n", data)
            self.assertNotIn(b"/data1/", data)
            self.assertNotIn(b"n164", data)
            with out.open() as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                if row["run"]:
                    self.assertFalse(row["run"].startswith("/"))
                    self.assertNotIn("..", row["run"])


if __name__ == "__main__":
    unittest.main()

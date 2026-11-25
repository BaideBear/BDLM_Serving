"""CPU-only tests for tools/hard_audit.py."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import hard_audit as audit


LLADA_ARM = audit.parse_policy("t2t_q4_f0_m2_hard_frontier")
GEMMA_ARM = audit.parse_policy("t2t_q4_f0_m2_hard_frontier")


def llada_event(stage, block, rows, positions, mandatory, extra=(), reserved_strict=(),
                strict=0, accepted=None, edited=None, masked_before=None, masked_after=None,
                overflow=None, budget=4, clock=None):
    event = dict(
        phase="prefill" if stage == "prefill" else "commit" if stage == "commit" else "denoise",
        block=block, stage=stage, physical_q=rows, actual_q=rows,
        absolute_positions=list(positions), q_budget=budget,
        mandatory_fresh=mandatory,
        reserved_extra_edit_positions=list(extra),
        reserved_strict_edit_positions=list(reserved_strict),
        overflow=max(0, rows - budget) if overflow is None else overflow,
        hard_strict_edits=strict, F=1,
    )
    if clock is not None:
        event["hard_clock"] = clock
    if accepted is not None:
        event["accepted"] = accepted
    if edited is not None:
        event["edited"] = edited
    if masked_before is not None:
        event["masked_before"] = masked_before
    if masked_after is not None:
        event["masked_after"] = masked_after
    return event


def llada_result(events, index=0, policy=None):
    work = [event for event in events if event["phase"] != "prefill"]
    query_rows = sum(int(event["physical_q"]) for event in work)
    prefill_rows = sum(int(event["physical_q"]) for event in events if event["phase"] == "prefill")
    return dict(
        model="llada21", policy=policy or LLADA_ARM["policy"], index=index,
        dataset="gsm8k", task_id="task-%02d" % index, seed=index,
        events=events,
        generation_forwards=len(work),
        generation_query_rows=query_rows,
        generation_physical_padded_rows=query_rows,
        max_q=max(int(event["physical_q"]) for event in work),
        prefill_rows=prefill_rows,
        t2t_edits=sum(int(event.get("edited", 0)) for event in events),
    )


def gemma_event(stage, block, rows, positions, available_fresh, selected_fresh,
                mandatory, fresh_selected, extra=(), reserved_strict=(),
                strict=0, edited=0, admitted=0, revoked=0, edited_in_fresh=0,
                overflow=0, resolved=0, selected_mask=0, clock=None):
    event = dict(
        block=block, clock=block, stage=stage, q=rows, positions=list(positions),
        available_fresh=available_fresh, selected_fresh=selected_fresh,
        mandatory_fresh=mandatory, fresh_selected=fresh_selected,
        reserved_extra_edit_positions=list(extra),
        reserved_strict_edit_positions=list(reserved_strict),
        hard_strict_edits=strict, edited=edited, admitted=admitted,
        revoked=revoked, resolved=resolved, selected_mask=selected_mask,
        edited_in_fresh=edited_in_fresh, overflow=overflow,
    )
    if clock is not None:
        event["hard_clock"] = clock
    return event


def gemma_result(events, index=0, policy=None, prefill=8, reencode=(8,)):
    physical = [["encoder", prefill]]
    physical += [["encoder", rows] for rows in reencode]
    physical += [["decoder", int(event["q"])] for event in events]
    work_rows = [int(event["q"]) for event in events] + list(reencode)
    query_rows = sum(work_rows)
    return dict(
        model="gemma", policy=policy or GEMMA_ARM["policy"], index=index,
        dataset="gsm8k", task_id="task-%02d" % index, seed=index,
        events=events, physical_events=physical,
        generation_forwards=len(events) + len(reencode),
        generation_query_rows=query_rows,
        generation_physical_padded_rows=query_rows,
        max_q=max(work_rows),
        prefill_rows=prefill,
        t2t_edits=sum(int(event.get("edited", 0)) for event in events),
    )


def llada_good_events():
    return [
        llada_event("prefill", 0, 8, range(8), 0, clock=1),
        llada_event("bootstrap", 0, 8, range(8), 0, accepted=2, edited=0,
                    masked_before=8, masked_after=6, clock=2),
        llada_event("selected", 0, 4, [0, 1, 2, 3], 2, extra=[2, 3],
                    reserved_strict=[3], strict=1, accepted=0, edited=1,
                    masked_before=6, masked_after=6, clock=3),
        llada_event("verify", 0, 8, range(8), 1, accepted=0, edited=0,
                    masked_before=6, masked_after=6, clock=4),
        llada_event("commit", 0, 8, range(8), 0, clock=5),
    ]


def gemma_good_events():
    return [
        gemma_event("bootstrap", 8, 8, range(8), 0, 0, 0, 0,
                    edited=0, admitted=2, revoked=0, resolved=2, selected_mask=8,
                    clock=1),
        gemma_event("selected", 8, 4, [0, 1, 2, 3], 2, 2, 2, 2,
                    extra=[2, 3], reserved_strict=[2], strict=1,
                    edited=1, admitted=0, revoked=0, resolved=2, selected_mask=2,
                    edited_in_fresh=0, clock=2),
        gemma_event("verify", 8, 8, range(8), 1, 1, 1, 1,
                    edited=0, admitted=0, revoked=0, resolved=2, selected_mask=6,
                    edited_in_fresh=0, clock=3),
    ]


class HardAuditTest(unittest.TestCase):
    def test_llada_good(self):
        counters, violations = audit.audit_request(
            llada_result(llada_good_events()), LLADA_ARM, "llada21", 8)
        self.assertEqual(violations, [])
        self.assertEqual(counters["selected_steps"], 1)
        self.assertEqual(counters["selected_q"], 4)
        self.assertEqual(counters["reserved_seats"], 2)
        self.assertEqual(counters["reserved_hits"], 1)
        self.assertEqual(counters["strict_edits"], 1)
        self.assertEqual(counters["commit_f"], 1)
        self.assertEqual(counters["prefill_q"], 8)

    def test_gemma_good(self):
        counters, violations = audit.audit_request(
            gemma_result(gemma_good_events()), GEMMA_ARM, "gemma", 8)
        self.assertEqual(violations, [])
        self.assertEqual(counters["selected_steps"], 1)
        self.assertEqual(counters["selected_q"], 4)
        self.assertEqual(counters["reserved_seats"], 2)
        self.assertEqual(counters["reserved_hits"], 1)
        self.assertEqual(counters["strict_edits"], 1)
        self.assertEqual(counters["reencode_f"], 1)
        self.assertEqual(counters["reencode_q"], 8)
        self.assertEqual(counters["selected_fresh_hit"], 2)

    def test_gemma_fresh_overflow_is_allowed(self):
        events = [
            gemma_event("bootstrap", 8, 8, range(8), 0, 0, 0, 0,
                        edited=0, admitted=6, revoked=0, resolved=6,
                        selected_mask=8, clock=1),
            gemma_event("selected", 8, 6, range(6), 6, 6, 6, 6,
                        edited=0, admitted=0, revoked=0, resolved=6,
                        selected_mask=0, overflow=2, clock=2),
        ]
        _, violations = audit.audit_request(
            gemma_result(events), GEMMA_ARM, "gemma", 8)
        self.assertEqual(violations, [])

    def test_gemma_bad_overflow(self):
        events = [
            gemma_event("bootstrap", 8, 8, range(8), 0, 0, 0, 0,
                        edited=0, admitted=4, revoked=0, resolved=4,
                        selected_mask=8, clock=1),
            gemma_event("selected", 8, 6, range(6), 4, 4, 4, 4,
                        extra=[4], edited=0, admitted=0, revoked=0, resolved=4,
                        selected_mask=2, overflow=2, clock=2),
        ]
        _, violations = audit.audit_request(
            gemma_result(events), GEMMA_ARM, "gemma", 8)
        self.assertTrue(any("pure fresh overflow" in item for item in violations))
        self.assertTrue(any("reserved extra" in item for item in violations))

    def test_gemma_fresh_miss(self):
        events = [
            gemma_event("bootstrap", 8, 8, range(8), 0, 0, 0, 0,
                        edited=0, admitted=2, revoked=0, resolved=2,
                        selected_mask=8, clock=1),
            gemma_event("selected", 8, 4, [0, 1, 2, 3], 2, 1, 2, 1,
                        edited=0, admitted=0, revoked=0, resolved=2,
                        selected_mask=2, clock=2),
        ]
        _, violations = audit.audit_request(
            gemma_result(events), GEMMA_ARM, "gemma", 8)
        self.assertTrue(any("fresh not fully selected" in item for item in violations))

    def test_reserved_must_be_q_subset(self):
        events = [
            gemma_event("bootstrap", 8, 8, range(8), 0, 0, 0, 0,
                        edited=0, admitted=2, revoked=0, resolved=2,
                        selected_mask=8, clock=1),
            gemma_event("selected", 8, 4, [0, 1, 2, 3], 2, 2, 2, 2,
                        extra=[7], edited=0, admitted=0, revoked=0, resolved=2,
                        selected_mask=2, clock=2),
        ]
        _, violations = audit.audit_request(
            gemma_result(events), GEMMA_ARM, "gemma", 8)
        self.assertTrue(any("Q subset" in item for item in violations))

    def test_llada_reserved_must_be_q_subset(self):
        events = llada_good_events()
        events[2]["reserved_extra_edit_positions"] = [7]
        _, violations = audit.audit_request(
            llada_result(events), LLADA_ARM, "llada21", 8)
        self.assertTrue(any("Q subset" in item for item in violations))

    def test_llada_bad_overflow(self):
        events = llada_good_events()
        events[2].update(physical_q=6, actual_q=6, absolute_positions=list(range(6)),
                         mandatory_fresh=4, reserved_extra_edit_positions=[4],
                         overflow=2)
        _, violations = audit.audit_request(
            llada_result(events), LLADA_ARM, "llada21", 8)
        self.assertTrue(any("pure fresh overflow" in item for item in violations))
        self.assertTrue(any("reserved extra" in item for item in violations))

    def test_llada_top_level_fields(self):
        cases = {
            "mandatory": ("mandatory_fresh", None),
            "clock": ("hard_clock", None),
            "strict": ("hard_strict_edits", 2),
            "overflow": ("overflow", 1),
            "mandatory_rows": ("mandatory_fresh", 5),
        }
        for name, (key, value) in cases.items():
            events = llada_good_events()
            if value is None:
                del events[2][key]
            else:
                events[2][key] = value
            _, violations = audit.audit_request(
                llada_result(events), LLADA_ARM, "llada21", 8)
            self.assertTrue(violations, name)

        events = llada_good_events()
        events[1].update(accepted=9, masked_before=8, masked_after=6)
        _, violations = audit.audit_request(
            llada_result(events), LLADA_ARM, "llada21", 8)
        self.assertTrue(any("accepted exceeds MASK before" in item for item in violations))

        events = llada_good_events()
        events[1].update(masked_before=8, masked_after=9, edited=0)
        _, violations = audit.audit_request(
            llada_result(events), LLADA_ARM, "llada21", 8)
        self.assertTrue(any("masked_after exceeds MASK before + edits" in item
                            for item in violations))

        events = llada_good_events()
        events[3]["hard_clock"] = 2
        _, violations = audit.audit_request(
            llada_result(events), LLADA_ARM, "llada21", 8)
        self.assertTrue(any("hard_clock not increasing" in item for item in violations))

    def test_llada_mask_writeback_chain(self):
        # One resolved->MASK edit is a mask writeback. The next mandatory
        # fresh count is edited + masked_before - masked_after = 1+6-7 = 0,
        # not accepted + edited + mask_writes.
        events = [
            llada_event("bootstrap", 0, 8, range(8), 0, accepted=2, edited=0,
                        masked_before=8, masked_after=6, clock=1),
            llada_event("selected", 0, 4, [0, 1, 2, 3], 2, accepted=0, edited=1,
                        masked_before=6, masked_after=7, clock=2),
            llada_event("verify", 0, 8, range(8), 0, accepted=0, edited=0,
                        masked_before=7, masked_after=7, clock=3),
            llada_event("commit", 0, 8, range(8), 0, clock=4),
        ]
        _, violations = audit.audit_request(
            llada_result(events), LLADA_ARM, "llada21", 8)
        self.assertFalse(any("fresh chain" in item for item in violations))
        events[2]["mandatory_fresh"] = 2
        _, violations = audit.audit_request(
            llada_result(events), LLADA_ARM, "llada21", 8)
        self.assertTrue(any("fresh chain mismatch" in item for item in violations))

    def test_clock_resets_per_block(self):
        events = [
            gemma_event("bootstrap", 8, 8, range(8), 0, 0, 0, 0,
                        edited=0, admitted=2, revoked=0, resolved=2,
                        selected_mask=8, clock=1),
            gemma_event("selected", 8, 4, [0, 1, 2, 3], 2, 2, 2, 2,
                        strict=1, edited=1, admitted=0, revoked=0, resolved=2,
                        selected_mask=2, clock=2),
            gemma_event("bootstrap", 9, 8, range(8), 0, 0, 0, 0,
                        edited=0, admitted=2, revoked=0, resolved=2,
                        selected_mask=8, clock=1),
        ]
        _, violations = audit.audit_request(
            gemma_result(events), GEMMA_ARM, "gemma", 8)
        self.assertFalse(any("hard_clock" in item for item in violations))

    def test_gemma_revoked_chain(self):
        events = [
            gemma_event("bootstrap", 8, 8, range(8), 0, 0, 0, 0,
                        edited=0, admitted=2, revoked=0, resolved=2,
                        selected_mask=8, clock=1),
            gemma_event("selected", 8, 4, [0, 1, 2, 3], 2, 2, 2, 2,
                        strict=1, edited=1, admitted=1, revoked=1, resolved=2,
                        selected_mask=2, edited_in_fresh=0, clock=2),
            gemma_event("verify", 8, 8, range(8), 2, 2, 2, 2,
                        edited=0, admitted=0, revoked=0, resolved=2,
                        selected_mask=6, clock=3),
        ]
        _, violations = audit.audit_request(
            gemma_result(events), GEMMA_ARM, "gemma", 8)
        self.assertFalse(any("chain" in item for item in violations))

        events[2]["available_fresh"] = 1
        _, violations = audit.audit_request(
            gemma_result(events), GEMMA_ARM, "gemma", 8)
        self.assertTrue(any("fresh chain mismatch" in item for item in violations))

        events[2]["available_fresh"] = 2
        events[2]["resolved"] = 3
        _, violations = audit.audit_request(
            gemma_result(events), GEMMA_ARM, "gemma", 8)
        self.assertTrue(any("resolved chain mismatch" in item for item in violations))

    def test_gemma_bounds(self):
        def build(**changes):
            selected = dict(
                available_fresh=2, selected_fresh=2, mandatory=2, fresh_selected=2,
                strict=1, edited=1, admitted=1, revoked=1, resolved=2,
                selected_mask=2, edited_in_fresh=0,
            )
            selected.update(changes)
            return [
                gemma_event("bootstrap", 8, 8, range(8), 0, 0, 0, 0,
                            edited=0, admitted=2, revoked=0, resolved=2,
                            selected_mask=8, clock=1),
                gemma_event("selected", 8, 4, [0, 1, 2, 3],
                            clock=2, **selected),
            ]
        checks = [
            (dict(admitted=3), "admitted+edited+revoked exceeds Q"),
            (dict(edited_in_fresh=2), "edited_in_fresh exceeds edited"),
            (dict(selected_fresh=3), "selected MASK+fresh exceeds Q"),
            (dict(revoked=-1), "negative revoked"),
            (dict(strict=0), "hard_strict_edits != edited"),
            (dict(resolved=9), "resolved outside block"),
            (dict(selected_mask=4), "resolved changes exceed non-MASK Q"),
            (dict(mandatory=5), "mandatory fresh exceeds Q"),
        ]
        for changes, message in checks:
            events = build(**changes)
            _, violations = audit.audit_request(
                gemma_result(events), GEMMA_ARM, "gemma", 8)
            self.assertTrue(any(message in item for item in violations),
                            (changes, violations))

    def test_traffic_mismatch(self):
        result = llada_result(llada_good_events())
        result["generation_forwards"] += 1
        _, violations = audit.audit_request(result, LLADA_ARM, "llada21", 8)
        self.assertTrue(any("generation_forwards" in item for item in violations))

    def test_selected_max_q_is_max_not_sum(self):
        arm = audit.parse_policy("t2t_q8_f0_m4_hard_frontier")

        def result_with_q(q, index):
            events = [
                llada_event("prefill", 0, 8, range(8), 0, budget=8, clock=1),
                llada_event("bootstrap", 0, 8, range(8), 0, accepted=2, edited=0,
                            masked_before=8, masked_after=6, budget=8, clock=2),
                llada_event("selected", 0, q, list(range(q)), 2, extra=[2],
                            reserved_strict=[2], strict=1, accepted=0, edited=1,
                            masked_before=6, masked_after=6, budget=8, clock=3),
                llada_event("verify", 0, 8, range(8), 1, accepted=0, edited=0,
                            masked_before=6, masked_after=6, budget=8, clock=4),
                llada_event("commit", 0, 8, range(8), 0, budget=8, clock=5),
            ]
            return llada_result(events, index=index)

        counters, violations = audit.audit_arm(
            "llada21", arm, [result_with_q(4, 0), result_with_q(6, 1)], 8)
        self.assertEqual(violations, [])
        self.assertEqual(counters["selected_max_q"], 6)
        row = audit.summary_row("llada21", arm, counters, violations, 2, 8)
        self.assertEqual(row["selected_max_q"], 6)

    def test_saturation_label_and_unverified_row(self):
        arm = audit.parse_policy("t2t_q64_f0_m64_hard_noedit")
        events = [
            llada_event("prefill", 0, 32, range(32), 0, budget=64, clock=1),
            llada_event("bootstrap", 0, 32, range(32), 0, accepted=2, edited=0,
                        masked_before=32, masked_after=30, budget=64, clock=2),
            llada_event("selected", 0, 32, range(32), 2, accepted=0, edited=0,
                        masked_before=30, masked_after=30, budget=64, clock=3),
            llada_event("commit", 0, 32, range(32), 0, budget=64, clock=4),
        ]
        counters, violations = audit.audit_arm(
            "llada21", arm, [llada_result(events)], 32)
        self.assertEqual(violations, [])
        row = audit.summary_row("llada21", arm, counters, violations, 1, 32)
        self.assertEqual(row["saturation"], 1)
        self.assertEqual(row["selected_max_q"], 32)
        self.assertEqual(row["ok"], 1)

        unverified = audit.unverified_row(
            "llada21", arm, 32, 0, "incomplete arm: 0/30 requests")
        self.assertEqual(unverified["saturation"], 1)
        self.assertEqual(unverified["requests"], 0)
        self.assertEqual(unverified["ok"], 0)
        self.assertEqual(unverified["fresh_verified"], 0)

    def test_main_reports_incomplete_arm(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "run.json").write_text(json.dumps(dict(
                model="llada21", block=32,
                policies=["t2t_q64_f0_m64_hard_noedit"],
                arms=[dict(budget=64, fresh=0, mask=64, selection="hard_noedit")])))
            manifest = folder / "manifest.json"
            manifest.write_text(json.dumps(dict(samples=[])))
            out = folder / "out.csv"
            code = audit.main(["--run", str(folder), "--out", str(out),
                               "--manifest", str(manifest), "--allow-violations"])
            self.assertEqual(code, 0)
            with out.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["ok"], "0")
            self.assertEqual(rows[0]["saturation"], "1")
            self.assertEqual(rows[0]["requests"], "0")
            self.assertIn("incomplete arm", rows[0]["first_violation"])

    def test_summary_keeps_model_differences(self):
        llada_counts, llada_bad = audit.audit_request(
            llada_result(llada_good_events()), LLADA_ARM, "llada21", 8)
        gemma_counts, gemma_bad = audit.audit_request(
            gemma_result(gemma_good_events()), GEMMA_ARM, "gemma", 8)
        llada_row = audit.summary_row("llada21", LLADA_ARM, llada_counts, llada_bad, 30, 8)
        gemma_row = audit.summary_row("gemma", GEMMA_ARM, gemma_counts, gemma_bad, 30, 8)
        self.assertEqual(llada_row["reencode_f"], "")
        self.assertEqual(llada_row["commit_f"], 1)
        self.assertEqual(llada_row["fresh_edits"], "")
        self.assertEqual(llada_row["fresh_exact"], 0)
        self.assertEqual(llada_row["fresh_evidence"], "none")
        self.assertEqual(llada_row["fresh_chain_ok"], 1)
        self.assertEqual(llada_row["fresh_verified"], 0)
        self.assertEqual(llada_row["saturation"], 0)
        self.assertEqual(gemma_row["commit_f"], "")
        self.assertEqual(gemma_row["reencode_f"], 1)
        self.assertEqual(gemma_row["fresh_exact"], 1)
        self.assertEqual(gemma_row["fresh_evidence"], "counts")
        self.assertEqual(gemma_row["fresh_chain_ok"], 1)
        self.assertEqual(gemma_row["fresh_verified"], 0)
        self.assertEqual(gemma_row["selected_fresh_hit"], 2)

    def test_arm_checks(self):
        bad = dict(LLADA_ARM, fresh=4)
        self.assertTrue(audit.arm_violations(bad))
        noedit = audit.parse_policy("t2t_q4_f0_m4_hard_noedit")
        self.assertEqual(audit.arm_violations(noedit), [])
        broken = dict(noedit, edit=1)
        self.assertTrue(audit.arm_violations(broken))

    def test_incomplete_arm_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for index in range(29):
                path = folder / ("%02d-%s.json" % (index, LLADA_ARM["policy"]))
                path.write_text(json.dumps(dict(
                    model="llada21", policy=LLADA_ARM["policy"], index=index,
                    dataset="gsm8k", task_id="task-%02d" % index, seed=index)))
            with self.assertRaises(ValueError):
                audit.read_arm(folder, LLADA_ARM["policy"], LLADA_ARM, "llada21", {})


if __name__ == "__main__":
    unittest.main()

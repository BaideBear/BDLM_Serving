"""Bounded, model-free tests for close_opportunity.py.

No torch, no checkpoint, no GPU.  Synthetic traces exercise every legal and
illegal close shape; the sealed CPU receipts are used only when present.
"""
import json
from pathlib import Path
import unittest

from experiments.t2t.serving_slots import close_opportunity as co


B = 256
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONTROLS = ROOT / "experiments/t2t/serving_slots/controls"
COST = CONTROLS / "gemma-strict-q128-close-cost-cpu.json"
PAIRED = CONTROLS / "gemma-strict-q128-close-paired.json"
DRAIN = CONTROLS / "gemma-strict-drain-grid-cpu3.json"
METRICS = CONTROLS / "gemma-strict-q128-cpu.json"


def event(tick, phase, rows, *, offer=128, dirty=(), admitted=(), edited=(),
          revoked=(), encoded=False, ordinal=0, stop_reason=None):
    rows = list(rows)
    sampled = rows if phase == "sample" else []
    return dict(tick=tick, ordinal=ordinal, phase=phase, offer=offer, q=len(rows),
                rows=rows, sampled=sampled, dirty_before=list(dirty),
                admitted=list(admitted), edited=list(edited), revoked=list(revoked),
                encoded=encoded, stop_reason=stop_reason)


def block(*, dirty, drain, offer=128, stop="limit", admit=None, tick=0, ordinal=0,
          has_next=False):
    """One valid control block: full sample, one pure drain, sequential verify."""
    dirty = list(dirty)
    drain = list(drain)
    admitted = dirty[: len(dirty) // 2] if admit is None else list(admit)
    # A full-canvas sample is a legal Q256 call; the drain and verify chunks use
    # the smaller offer under test.  The strict path allows per-step offers.
    trace = [event(tick, "sample", range(B), offer=256, dirty=(), admitted=admitted,
                   ordinal=ordinal)]
    cursor = tick + 1
    if drain:
        trace.append(event(cursor, "drain", drain, offer=offer, dirty=dirty,
                           ordinal=ordinal))
        cursor += 1
    chunks = [list(range(i, min(B, i + offer))) for i in range(0, B, offer)]
    for index, rows in enumerate(chunks):
        trace.append(event(cursor + index, "verify", rows, offer=offer, dirty=(),
                           encoded=(index == len(chunks) - 1 and has_next),
                           ordinal=ordinal, stop_reason=stop))
    return trace


class TraceAudit(unittest.TestCase):
    def test_all_terminal_drain_in_first_chunk_is_exact(self):
        trace = block(dirty=range(16), drain=range(16))
        report = co.audit_trace(trace)
        result = report["blocks"][0]
        self.assertEqual(result["variant"], "control")
        self.assertEqual((result["exact_suffix_save_F"], result["exact_suffix_save_Q"]), (1, 16))
        self.assertEqual((result["repack_ceiling_F"], result["repack_ceiling_Q"]), (1, 16))
        self.assertEqual((result["gap_F"], result["gap_Q"]), (0, 0))
        self.assertEqual(result["proof"], "exact_all_terminal_drains")
        self.assertEqual(result["dirty_by_verify_chunk"], [16, 0])
        self.assertTrue(result["repack_exact"])
        self.assertFalse(result["repack_changes_partition"])
        self.assertTrue(all(result["conditions"].values()))
        self.assertEqual(result["non_droppable_terminal"], [])

    def test_drain_in_second_chunk_is_not_exact(self):
        trace = block(dirty=[128], drain=[128])
        result = co.audit_trace(trace)["blocks"][0]
        self.assertEqual((result["exact_suffix_save_F"], result["exact_suffix_save_Q"]), (0, 0))
        self.assertEqual((result["repack_ceiling_F"], result["repack_ceiling_Q"]), (1, 1))
        self.assertEqual((result["gap_F"], result["gap_Q"]), (1, 1))
        self.assertEqual(result["proof"], "no_exact_drop")
        self.assertEqual(result["dirty_by_verify_chunk"], [0, 1])
        self.assertTrue(result["repack_fit"])
        self.assertFalse(result["repack_exact"])
        self.assertTrue(result["repack_changes_partition"])
        self.assertEqual(result["non_droppable_terminal"][0]["reason"],
                         "rows_outside_first_verify")
        self.assertEqual(result["non_droppable_terminal"][0]["own_rows_outside_first_verify"],
                         [128])

    def test_later_drain_outside_first_chunk_blocks_an_earlier_exact_drain(self):
        dirty = list(range(32)) + list(range(128, 160))
        trace = block(dirty=dirty, drain=dirty[:32], offer=32)
        # The helper emits one drain only; add the second terminal drain.
        second = event(2, "drain", list(range(128, 160)), offer=32,
                       dirty=list(range(128, 160)), ordinal=0)
        trace = trace[:2] + [second] + trace[2:]
        trace = [dict(e, tick=i) for i, e in enumerate(trace)]
        result = co.audit_trace(trace)["blocks"][0]
        self.assertEqual((result["exact_suffix_save_F"], result["repack_ceiling_F"]), (0, 2))
        reasons = {item["tick"]: item["reason"] for item in result["non_droppable_terminal"]}
        self.assertEqual(reasons[1], "retained_later_drain_would_read_stale_kv")
        self.assertEqual(reasons[2], "rows_outside_first_verify")

    def test_non_terminal_drain_is_never_exact(self):
        trace = block(dirty=[0], drain=[], stop="limit")
        # Insert a drain after the first sample, then a second sample before verify.
        trace = [trace[0],
                 event(1, "drain", [0], offer=128, dirty=[0], ordinal=0),
                 event(2, "sample", [0], offer=128, dirty=(), ordinal=0)] + [
                    dict(e, tick=e["tick"] + 2) for e in trace[1:]]
        result = co.audit_trace(trace)["blocks"][0]
        self.assertEqual((result["other_drain_F"], result["other_drain_Q"]), (1, 1))
        self.assertEqual(result["non_terminal_drains"][0]["reason"],
                         "not_terminal_a_sample_step_follows_before_verify")
        self.assertEqual((result["exact_suffix_save_F"], result["repack_ceiling_F"]), (0, 0))

    def test_prior_pairing_rejects_debt_that_drops_an_admission(self):
        trace = block(dirty=[0], drain=[0], admit=[1])
        with self.assertRaisesRegex(co.AuditError, "prior pairing"):
            co.audit_trace(trace)

    def test_self_conditioning_rejects_unseen_dirty_row(self):
        trace = [event(0, "sample", [1], offer=128, dirty=(), admitted=[1]),
                 event(1, "drain", [0], offer=128, dirty=[0]),
                 event(2, "verify", list(range(128)), offer=128, dirty=(),
                       stop_reason="limit"),
                 event(3, "verify", list(range(128, 256)), offer=128, dirty=(),
                       stop_reason="limit")]
        with self.assertRaisesRegex(co.AuditError, "self-conditioning"):
            co.audit_trace(trace)

    def test_encoder_flag_must_follow_the_block_boundary(self):
        trace = block(dirty=[], drain=[], has_next=False)
        trace[-1]["encoded"] = True
        with self.assertRaisesRegex(co.AuditError, "encoder flag"):
            co.audit_trace(trace)

    def test_verify_must_cover_the_canvas(self):
        trace = block(dirty=[], drain=[])
        trace[-1]["rows"] = list(range(129, 256))
        trace[-1]["q"] = len(trace[-1]["rows"])
        with self.assertRaisesRegex(co.AuditError, "cover the canvas|sequential"):
            co.audit_trace(trace)

    def test_hard_cap_rejects_q_above_offer(self):
        trace = block(dirty=[], drain=[])
        trace[0]["offer"] = 128
        with self.assertRaisesRegex(co.AuditError, "outside"):
            co.audit_trace(trace)

    def test_repacked_close_trace_is_rejected_as_control(self):
        trace = block(dirty=[0], drain=[0])
        # The close variant leaves its first verify with nonempty dirty_before.
        first_verify = next(e for e in trace if e["phase"] == "verify")
        first_verify["dirty_before"] = [0]
        with self.assertRaisesRegex(co.AuditError, "repacked close trace"):
            co.audit_trace(trace)

    def test_trace_audit_reproduces_the_k0_block_shapes(self):
        shapes = [
            ([128], [0, 1], 0, 1),
            (list(range(16)), [16, 0], 1, 1),
            ([0] + list(range(128, 141)), [1, 13], 0, 1),
            (list(range(2)), [2, 0], 1, 1),
            ([0, 1] + list(range(128, 158)), [2, 30], 0, 1),
        ]
        for dirty, hits, exact, repack in shapes:
            with self.subTest(dirty=len(dirty)):
                trace = block(dirty=dirty, drain=sorted(dirty)[:128], offer=128,
                              stop="limit")
                result = co.audit_trace(trace)["blocks"][0]
                self.assertEqual(result["dirty_by_verify_chunk"], hits)
                self.assertEqual(result["exact_suffix_save_F"], exact)
                self.assertEqual(result["repack_ceiling_F"], repack)


class SummaryAudit(unittest.TestCase):
    @unittest.skipUnless(COST.is_file(), "requires sealed close-cost receipt")
    def test_sealed_q128_k0_and_k32(self):
        document = json.loads(COST.read_text())
        report = co.audit_summary(document)
        self.assertEqual(report["kind"], "close_cost")
        by_k = {arm["k"]: arm for arm in report["by_k"]}
        self.assertEqual((by_k[0]["exact_suffix_save_F"], by_k[0]["exact_suffix_save_Q"]), (2, 18))
        self.assertEqual((by_k[0]["repack_ceiling_F"], by_k[0]["repack_ceiling_Q"]), (5, 65))
        self.assertEqual((by_k[0]["gap_F"], by_k[0]["gap_Q"]), (3, 47))
        self.assertEqual((by_k[32]["exact_suffix_save_F"], by_k[32]["exact_suffix_save_Q"]), (2, 3))
        self.assertEqual((by_k[32]["repack_ceiling_F"], by_k[32]["repack_ceiling_Q"]), (5, 22))
        # Per-block evidence: the three extra drains sit in chunk 1.
        k0 = [block for arm in report["arms"] if arm["k"] == 0 for block in arm["blocks"]]
        self.assertEqual([b["dirty_by_verify_chunk"] for b in k0],
                         [[0, 1], [16, 0], [1, 13], [2, 0], [2, 30]])
        self.assertEqual([b["exact_suffix_save_F"] for b in k0], [0, 1, 0, 1, 0])
        self.assertEqual([b["repack_ceiling_F"] for b in k0], [1, 1, 1, 1, 1])

    @unittest.skipUnless(COST.is_file(), "requires sealed close-cost receipt")
    def test_summary_drain_composition_totals(self):
        document = json.loads(COST.read_text())
        report = co.audit_summary(document)
        arms0 = [arm for arm in report["arms"] if arm["k"] == 0]
        self.assertEqual(sum(arm["terminal_drain_F"] for arm in arms0), 5)
        self.assertEqual(sum(arm["terminal_drain_Q"] for arm in arms0), 65)
        self.assertEqual(sum(b["drain_write_or_flip_Q"] for arm in arms0 for b in arm["blocks"]), 882)
        self.assertEqual(sum(b["drain_unaccepted_noise_Q"] for arm in arms0 for b in arm["blocks"]), 1871)
        self.assertEqual(sum(b["pure_drains"] for arm in arms0 for b in arm["blocks"]), 26)


class PairedAudit(unittest.TestCase):
    @unittest.skipUnless(PAIRED.is_file(), "requires sealed paired close receipt")
    def test_measured_save_and_exact_gap(self):
        document = json.loads(PAIRED.read_text())
        report = co.audit_paired(document)
        totals = report["totals"]
        self.assertEqual((totals["control_F"], totals["control_Q"]), (76, 8511))
        self.assertEqual((totals["dirty_first_F"], totals["dirty_first_Q"]), (71, 8446))
        self.assertEqual((totals["measured_save_F"], totals["measured_save_Q"]), (5, 65))
        self.assertEqual((totals["control_terminal_drain_F"], totals["control_terminal_drain_Q"]),
                         (5, 65))
        self.assertTrue(totals["measured_equals_terminal"])
        self.assertEqual(totals["identical_output_pairs"], 3)
        self.assertEqual([p["first_divergent_decoder_tick_0based"] for p in report["pairs"]],
                         [15, 17, 10])
        self.assertTrue(all(p["identical_output_ids"] for p in report["pairs"]))
        self.assertTrue(all(p["delta_beyond_terminal_drains"] == {"F": 0, "Q": 0}
                            for p in report["pairs"]))

    @unittest.skipUnless(COST.is_file() and PAIRED.is_file(), "requires both sealed receipts")
    def test_verdict_separates_exact_ceiling_and_measured(self):
        summary = co.audit_summary(json.loads(COST.read_text()))
        paired = co.audit_paired(json.loads(PAIRED.read_text()))
        verdict = co._verdict(summary, paired)
        with self.assertRaisesRegex(co.AuditError, "manifest mismatch"):
            co._verdict(summary, paired | {"samples":"0"*64})
        with self.assertRaisesRegex(co.AuditError, "artifact mismatch"):
            co._verdict(summary, paired | {"historical":{}})
        k0 = next(item for item in verdict if item["k"] == 0)
        self.assertEqual((k0["exact_suffix_save_F"], k0["exact_suffix_save_Q"]), (2, 18))
        self.assertEqual((k0["repack_ceiling_F"], k0["repack_ceiling_Q"]), (5, 65))
        self.assertEqual((k0["measured_save_F"], k0["measured_save_Q"]), (5, 65))
        self.assertEqual((k0["measured_beyond_exact_F"], k0["measured_beyond_exact_Q"]), (3, 47))
        self.assertTrue(k0["measured_equals_ceiling"])
        self.assertFalse(k0["measured_equals_exact"])


class MetricsAudit(unittest.TestCase):
    @unittest.skipUnless(METRICS.is_file(), "requires sealed six-record metrics")
    def test_phase_accounting(self):
        report = co.audit_metrics(json.loads(METRICS.read_text()))
        self.assertEqual(report["kind"], "metrics")
        self.assertEqual(len(report["rows"]), 6)
        first = report["rows"][0]
        self.assertEqual((first["index"], first["k"]), (0, 0))
        self.assertEqual(first["phase_F"], {"sample": 16, "drain": 10, "verify": 4})
        self.assertEqual(first["phase_Q"], {"sample": 1592, "drain": 1041, "verify": 512})
        self.assertEqual((first["F"], first["decoder_F"], first["encoder_F"]), (31, 30, 1))
        self.assertEqual((first["Q"], first["decoder_Q"], first["encoder_Q"]), (3401, 3145, 256))

    @unittest.skipUnless(METRICS.is_file(), "requires sealed six-record metrics")
    def test_bad_phase_sum_is_rejected(self):
        document = json.loads(METRICS.read_text())
        document["rows"][0]["phase_F"]["sample"] += 1
        with self.assertRaisesRegex(co.AuditError, "phase_F"):
            co.audit_metrics(document)


class EndToEnd(unittest.TestCase):
    @unittest.skipUnless(all(path.is_file() for path in (COST, DRAIN, METRICS, PAIRED)),
                         "requires all sealed receipts")
    def test_build_report(self):
        report = co.build_report([COST, DRAIN, METRICS], PAIRED, root=ROOT)
        self.assertEqual(
            report["conclusion"],
            "k=0: exact 2 F / 18 Q; ceiling 5 F / 65 Q; measured 5 F / 65 Q "
            "(beyond exact 3 F / 47 Q) | k=32: exact 2 F / 3 Q; ceiling 5 F / 22 Q")
        self.assertEqual(len(report["code_evidence"]), len(co.CODE_CLAIMS))
        self.assertEqual([s["kind"] for s in report["summaries"]],
                         ["close_cost", "drain_grid", "metrics"])
        self.assertEqual(report["paired"]["totals"]["measured_save_F"], 5)
        self.assertEqual(report["paired"]["totals"]["measured_save_Q"], 65)


class CodeEvidence(unittest.TestCase):
    @unittest.skipUnless(ROOT.joinpath("experiments/t2t/serving_slots/gemma_strict_rows.py").is_file(),
                         "requires checkout sources")
    def test_every_invariant_has_a_line(self):
        evidence = co.code_evidence(ROOT)
        claims = {item["claim"]: item for item in evidence}
        self.assertEqual(len(claims), len(co.CODE_CLAIMS))
        self.assertEqual(claims["hard_cap_offers"]["line"], 16)
        self.assertEqual(claims["token_cap"]["line"], 17)
        self.assertEqual(claims["denoise_steps"]["line"], 49)
        self.assertEqual(claims["drain_debt_prefix"]["line"], 83)
        self.assertEqual(claims["prior_pairing_dirty"]["line"], 119)
        self.assertEqual(claims["encoder_barrier_commit"]["line"], 124)
        self.assertEqual(claims["kv_write_before_read_k"]["line"], 319)
        self.assertEqual(claims["kv_live_after_write"]["line"], 327)
        self.assertEqual(claims["same_block_visibility"]["line"], 338)
        self.assertEqual(claims["encoder_flag_audit"]["line"], 99)
        self.assertEqual(claims["close_chunk_merge"]["line"], 31)


if __name__ == "__main__":
    unittest.main()

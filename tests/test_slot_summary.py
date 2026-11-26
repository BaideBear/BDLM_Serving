#!/usr/bin/env python3
"""Tests for slot_summary.py -- synthetic fixtures, no n164 data required."""

from __future__ import annotations

import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

from experiments.t2t.serving_slots import slot_summary as ss


COH = {
    "original-0-29": dict(req=30, a_c=20, b_c=18, a_N=600, a_F=100,
                          a_Qp=3000, b_N=700, b_F=110, b_Qp=3200),
    "added-30-89": dict(req=60, a_c=40, b_c=38, a_N=1200, a_F=200,
                        a_Qp=6000, b_N=1400, b_F=220, b_Qp=6400),
    "all-0-89": dict(req=90, a_c=60, b_c=56, a_N=1800, a_F=300,
                     a_Qp=9000, b_N=2100, b_F=330, b_Qp=9600),
}


def pair_rows(model, arm, base, overrides=None):
    overrides = overrides or {}
    rows = []
    for cohort in ss.COHORTS:
        v = dict(COH[cohort])
        v.update(overrides.get(cohort, {}))
        a_q = v.get("a_Q", v["a_Qp"])
        b_q = v.get("b_Q", v["b_Qp"])
        rows.append({
            "model": model, "arm_policy": arm, "baseline_policy": base,
            "nominal_q": "", "window": v["req"], "cohort": cohort,
            "scope": "all", "requests": v["req"],
            "arm_correct": v["a_c"], "baseline_correct": v["b_c"],
            "arm_macro_tpf": v.get("a_tpf", v["a_N"] / v["a_F"]),
            "baseline_macro_tpf": v.get("b_tpf", v["b_N"] / v["b_F"]),
            "arm_pooled_token_efficiency":
                v.get("a_eff", v["a_N"] / v["a_Qp"]),
            "baseline_pooled_token_efficiency":
                v.get("b_eff", v["b_N"] / v["b_Qp"]),
            "arm_N": v["a_N"], "arm_F": v["a_F"], "arm_Q": a_q,
            "arm_physical_Q": v["a_Qp"],
            "arm_q_per_f": v.get("a_qpf", v["a_Qp"] / v["a_F"]),
            "baseline_N": v["b_N"], "baseline_F": v["b_F"], "baseline_Q": b_q,
            "baseline_physical_Q": v["b_Qp"],
            "baseline_q_per_f": v.get("b_qpf", v["b_Qp"] / v["b_F"]),
        })
    return rows


def fixed_rows(model, slots, overrides=None):
    overrides = overrides or {}
    rows = []
    for slot in slots:
        for cohort in ss.COHORTS:
            v = dict(COH[cohort])
            v.update(overrides.get((slot, cohort), {}))
            rows.append({
                "model": model, "slot": slot,
                "role": "diagnostic" if (model, slot) == ("gemma", 16)
                        else "primary",
                "cohort": cohort, "requests": v["req"],
                "correct": v["a_c"],
                "macro_tpf": v.get("a_tpf", v["a_N"] / v["a_F"]),
                "pooled_token_efficiency":
                    v.get("a_eff", v["a_N"] / v["a_Qp"]),
                "N": v["a_N"], "F": v["a_F"], "physical_Q": v["a_Qp"],
                "q_per_f": v.get("a_qpf", v["a_Qp"] / v["a_F"]),
            })
    return rows


def write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.controls = self.root / "controls"
        self.controls.mkdir()
        self.fixed = self.root / "fixed-k0.csv"
        rows = (fixed_rows("llada21", (4, 8, 16, 32))
                + fixed_rows("gemma", (16, 32, 64, 128, 256)))
        write(self.fixed, rows)

    def tearDown(self):
        self._tmp.cleanup()

    def control(self, name, rows):
        write(self.controls / name, rows)

    def summary(self, allowlist):
        return ss.summarize(self.fixed, self.controls, allowlist=allowlist)


class TestValid(Base):
    def test_unified_names_do_not_overwrite_old_k0_and_replicate_random_anchor(self):
        self.assertEqual(ss.variant("gemma","q128_k32_random"),ss.variant("gemma","q128_slack_random"))
        self.assertNotEqual(ss.variant("gemma","q32")[0],"k0_zero")
        self.assertEqual(ss.variant("gemma","q64_k16_oldest")[0],"unified_k16_oldest")
        with self.assertRaises(ss.SummaryError):
            ss.variant("gemma","q256_k16_oldest")
    def test_long_format_and_external_labels(self):
        self.control("llada21-q8-original-ibs.csv",
                     pair_rows("llada21", "slottrace_8",
                               "optimus_ibs_c8"))
        rows = self.summary(("llada21-q8-original-ibs.csv",))
        k0 = [r for r in rows if r["endpoint"] == "llada21:k0_zero:8"]
        self.assertEqual(len(k0), 3)
        self.assertEqual({r["cohort"] for r in k0}, set(ss.COHORTS))
        self.assertTrue(all(r["family"] == "internal" for r in k0))
        self.assertIn("fixed-k0.csv", k0[0]["source"])
        self.assertIn("llada21-q8-original-ibs.csv#arm", k0[0]["source"])
        self.assertEqual(k0[0]["raw_Q"], k0[0]["physical_Q"])
        ext = [r for r in rows if r["endpoint"] == "llada21:ibs_original:8"]
        self.assertEqual(len(ext), 3)
        self.assertTrue(all(r["family"] == "external" for r in ext))
        self.assertTrue(all(r["cap"] == "nominal_only" for r in ext))
        self.assertEqual(ext[0]["policy"], "optimus_ibs_c8")
        for row in rows:
            self.assertAlmostEqual(row["efficiency"],
                                   row["N"] / row["physical_Q"])
            self.assertAlmostEqual(row["q_per_f"],
                                   row["physical_Q"] / row["F"])

    def test_gemma16_excluded(self):
        rows = self.summary(())
        self.assertFalse([r for r in rows
                          if r["model"] == "gemma" and r["offer"] == 16])
        self.assertTrue([r for r in rows
                         if r["model"] == "gemma" and r["offer"] == 32])

    def test_no_fabricated_gap_rows(self):
        rows = self.summary(())
        self.assertEqual(
            {r["endpoint"] for r in rows},
            {"llada21:k0_zero:4", "llada21:k0_zero:8",
             "llada21:k0_zero:16", "llada21:k0_zero:32",
             "gemma:k0_zero:32", "gemma:k0_zero:64",
             "gemma:k0_zero:128", "gemma:k0_zero:256"})

    def test_flow_q_per_f_is_row_derived(self):
        overrides = {
            "original-0-29": {"a_F": 50},
            "added-30-89": {"a_F": 100},
            "all-0-89": {"a_F": 150},
        }
        self.control("gemma-wave-vs-fixed.csv",
                     pair_rows("gemma", "flow_w2", "flow_w1", overrides))
        rows = self.summary(("gemma-wave-vs-fixed.csv",))
        flow = [r for r in rows if r["endpoint"] == "gemma:flow_w2:512"]
        self.assertEqual(len(flow), 3)
        all90 = [r for r in flow if r["cohort"] == "all-0-89"][0]
        self.assertEqual(all90["offer"], 512)
        self.assertEqual(all90["physical_Q"], 9000)
        self.assertEqual(all90["F"], 150)
        self.assertEqual(all90["q_per_f"], 60.0)
        self.assertNotEqual(all90["q_per_f"], 30.0)


class TestValidation(Base):
    def test_sum_mismatch(self):
        rows = pair_rows("llada21", "slottrace_8", "optimus_ibs_c8")
        rows[2]["arm_correct"] = 61
        self.control("llada21-q8-original-ibs.csv", rows)
        with self.assertRaises(ss.SummaryError):
            self.summary(("llada21-q8-original-ibs.csv",))

    def test_efficiency_mismatch(self):
        overrides = {"original-0-29": {"a_eff": 0.9}}
        rows = pair_rows("llada21", "slottrace_8", "optimus_ibs_c8", overrides)
        self.control("llada21-q8-original-ibs.csv", rows)
        with self.assertRaises(ss.SummaryError):
            self.summary(("llada21-q8-original-ibs.csv",))

    def test_raw_above_physical(self):
        overrides = {"original-0-29": {"a_Q": 4000}}
        rows = pair_rows("llada21", "slottrace_8", "optimus_ibs_c8", overrides)
        self.control("llada21-q8-original-ibs.csv", rows)
        with self.assertRaises(ss.SummaryError):
            self.summary(("llada21-q8-original-ibs.csv",))

    def test_q_per_f_column_mismatch(self):
        overrides = {c: {"a_qpf": 999.0} for c in ss.COHORTS}
        rows = pair_rows("llada21", "slottrace_8", "optimus_ibs_c8", overrides)
        self.control("llada21-q8-original-ibs.csv", rows)
        with self.assertRaises(ss.SummaryError):
            self.summary(("llada21-q8-original-ibs.csv",))

    def test_missing_cohort(self):
        rows = pair_rows("llada21", "slottrace_8", "optimus_ibs_c8")[:2]
        self.control("llada21-q8-original-ibs.csv", rows)
        with self.assertRaises(ss.SummaryError):
            self.summary(("llada21-q8-original-ibs.csv",))

    def test_duplicate_cohort(self):
        rows = pair_rows("llada21", "slottrace_8", "optimus_ibs_c8")
        self.control("llada21-q8-original-ibs.csv", rows + [rows[0]])
        with self.assertRaises(ss.SummaryError):
            self.summary(("llada21-q8-original-ibs.csv",))

    def test_consistency_across_files(self):
        first = pair_rows("llada21", "slottrace_8", "optimus_ibs_c8")
        second = pair_rows("llada21", "slottrace_8", "optimus_obs_c8")
        second[0]["arm_correct"] = 21
        second[1]["arm_correct"] = 41
        second[2]["arm_correct"] = 62
        self.control("llada21-q8-original-ibs.csv", first)
        self.control("llada21-q8-original-obs.csv", second)
        with self.assertRaises(ss.SummaryError):
            self.summary(("llada21-q8-original-ibs.csv",
                          "llada21-q8-original-obs.csv"))

    def test_missing_allowlisted_file(self):
        with self.assertRaises(ss.SummaryError):
            self.summary(("does-not-exist.csv",))

    def test_fixed_q_per_f_mismatch(self):
        rows = fixed_rows("llada21", (4,))
        rows[0]["q_per_f"] = 999.0
        bad = self.root / "bad-fixed.csv"
        write(bad, rows)
        with self.assertRaises(ss.SummaryError):
            ss.summarize(bad, self.controls, allowlist=())


class TestCli(Base):
    def test_missing_fixed_returns_2(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code = ss.main(["--fixed-k0", str(self.root / "nope.csv"),
                            "--controls", str(self.controls)])
        self.assertEqual(code, 2)

    def test_cli_writes_csv(self):
        self.control("llada21-q8-original-ibs.csv",
                     pair_rows("llada21", "slottrace_8",
                               "optimus_ibs_c8"))
        out = self.root / "summary.csv"
        old = ss.ALLOWLIST
        ss.ALLOWLIST = ("llada21-q8-original-ibs.csv",)
        try:
            code = ss.main(["--fixed-k0", str(self.fixed),
                            "--controls", str(self.controls),
                            "--out", str(out)])
        finally:
            ss.ALLOWLIST = old
        self.assertEqual(code, 0)
        with out.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(list(rows[0]), list(ss.FIELDS))
        self.assertIn("llada21:ibs_original:8",
                      {r["endpoint"] for r in rows})


if __name__ == "__main__":
    unittest.main(verbosity=2)

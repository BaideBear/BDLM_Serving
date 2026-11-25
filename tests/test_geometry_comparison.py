import csv
import tempfile
import unittest
from pathlib import Path

from tools.geometry_comparison import Arm, DEFAULTS, PLACEMENT, build, comparisons


MANIFEST = {i: ("gsm8k", f"gsm8k/{i}") for i in range(4)}
CANDIDATE = Arm("geometry", 16, 2, "hard_geometry")
REFERENCE = Arm("hard", 16, 2, "hard_frontier")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def config(arm):
    return dict(model="llada21", policy=arm.policy, nominal_q=arm.q,
                fresh=0, mask=arm.mask, edit=arm.edit)


class GeometryComparisonTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.src = {label: Path(tmp.name) / label for label in DEFAULTS}

    def write_arm(self, arm, outcomes, *, partial=False, with_summary=True,
                  summary_overrides=None, wrong_task=False, duplicate=False):
        rows = []
        for i, correct in enumerate(outcomes):
            dataset, task_id = MANIFEST[i]
            rows.append(dict(**config(arm), index=i, dataset=dataset,
                             task_id="wrong" if wrong_task and i == 1 else task_id,
                             correct=correct, failed=0, truncated=0, N=4, F=2 if i == 0 else 4,
                             Q=32, physical_Q=32, max_q=16))
        source = self.src[arm.source]
        # The candidate and reference are intentionally in different row order.
        order = rows[::-1] if arm.source == "geometry" else rows
        write_csv(source / "llada21-online-requests.csv",
                  order[:2] if partial else order + order[:1] if duplicate else order)
        if with_summary:
            summary = dict(**config(arm), dataset="all", requests=len(rows), correct=sum(outcomes),
                           failed=0, truncated=0, N=16, F=14, Q=128, physical_Q=128,
                           tpf=1.25, token_efficiency=0.125, q_per_f=128/14, max_q=16)
            summary.update(summary_overrides or {})
            write_csv(source / "llada21-online-summary.csv", [summary])

    def write_audit(self, arm, *, ok=1, hits=3, selected_q=64):
        meta = config(arm)
        budget = meta.pop("nominal_q")
        path = self.src[arm.source] / "llada21-audit.csv"
        row = dict(**meta, budget=budget, requests=4, ok=ok, reserved_seats=10,
                   reserved_hits=hits, selected_steps=10, selected_q=selected_q,
                   bootstrap_f=4, bootstrap_q=64, verify_f=0, verify_q=0,
                   commit_f=0, commit_q=0, dense_f=0, dense_q=0, reencode_f="", reencode_q="")
        write_csv(path, [row])

    def pair(self):
        _, pairs = build(self.src, MANIFEST)
        return next(r for r in pairs if r["geometry_policy"] == CANDIDATE.policy
                    and r["reference_policy"] == REFERENCE.policy)

    def test_pair_by_manifest_identity_and_recompute_metrics(self):
        self.write_arm(CANDIDATE, [1, 0, 1, 1])
        self.write_arm(REFERENCE, [1, 1, 0, 0])
        self.write_audit(CANDIDATE)
        row = self.pair()
        self.assertEqual((row["paired_requests"], row["wins"], row["losses"],
                          row["both_correct"], row["neither_correct"]), (4, 2, 1, 1, 0))
        self.assertEqual(row["delta_accuracy"], 0.25)
        self.assertEqual(row["geometry_tpf"], 1.25)
        self.assertEqual(row["geometry_token_efficiency"], 0.125)
        self.assertEqual(row["geometry_q_per_f"], 128/14)
        self.assertEqual(row["geometry_reserved_hits"], 3)
        self.assertEqual(row["geometry_reserved_hit_rate"], 0.3)
        self.assertEqual(row["reference_reserved_status"], "missing_evidence")

    def test_missing_partial_and_summary_only_are_not_paired(self):
        self.write_arm(REFERENCE, [1, 1, 0, 0])
        for expected, detail in (("missing", "requests 0/4"), ("incomplete", "requests 2/4"),
                                 ("incomplete", "missing all-summary")):
            with self.subTest(detail=detail):
                if detail == "requests 2/4":
                    self.write_arm(CANDIDATE, [1, 0, 1, 1], partial=True)
                elif detail == "missing all-summary":
                    self.write_arm(CANDIDATE, [1, 0, 1, 1])
                    (self.src["geometry"] / "llada21-online-summary.csv").unlink()
                row = self.pair()
                self.assertEqual(row["status"], "missing_geometry")
                self.assertEqual(row["geometry_status"], expected)
                self.assertIn(detail, row["geometry_detail"])
                self.assertIsNone(row["wins"])
                self.assertEqual(row["reference_correct"], 2)

    def test_wrong_identity_duplicates_and_stale_summary_rejected(self):
        for kw, msg in ((dict(wrong_task=True), "frozen manifest"),
                        (dict(duplicate=True), "duplicate request"),
                        (dict(summary_overrides={"F": 15}), "summary F")):
            with self.subTest(kw=kw), self.assertRaisesRegex(ValueError, msg):
                self.write_arm(CANDIDATE, [1, 0, 1, 1], **kw)
                self.pair()

    def test_unverified_audit_leaves_hits_blank_but_bad_counts_rejected(self):
        self.write_arm(CANDIDATE, [1, 0, 1, 1])
        self.write_audit(CANDIDATE, ok=0)
        row = self.pair()
        self.assertEqual(row["geometry_status"], "ok")
        self.assertEqual(row["geometry_reserved_status"], "unverified_audit")
        self.assertIsNone(row["geometry_reserved_hits"])
        self.write_audit(CANDIDATE, hits=11)
        with self.assertRaisesRegex(ValueError, "impossible reservation"):
            self.pair()
        self.write_audit(CANDIDATE, selected_q=60)
        with self.assertRaisesRegex(ValueError, "audit Q"):
            self.pair()

    def test_paired_cost_is_an_optional_reservation_source(self):
        self.write_arm(REFERENCE, [1, 1, 0, 0])
        path = self.src["hard"] / "llada21-paired-cost.csv"
        row = dict(**config(REFERENCE), requests=4, N=16, F=14, Q=128,
                   reserved_seats=8, reserved_strict_hits=2)
        write_csv(path, [row])
        self.assertEqual(self.pair()["reference_reserved_hits"], 2)
        row["Q"] = 129
        write_csv(path, [row])
        with self.assertRaisesRegex(ValueError, "paired-cost work"):
            self.pair()

    def test_matrix_matches_launched_arms_and_separate_zero_reserve(self):
        pairs = list(comparisons())
        self.assertEqual(len(pairs), 12)
        self.assertEqual({(arm.q, arm.edit) for arm, _ in pairs if arm.selection == "hard_geometry"},
                         {(16, 2)})
        self.assertEqual({(ref.selection, ref.edit) for arm, ref in pairs if arm == CANDIDATE},
                         {("hard_frontier", 2), ("hard_random", 2), ("hard_noedit", 0)})
        for arm, ref in pairs:
            self.assertEqual(arm.q, ref.q)
            self.assertEqual(ref.edit, 0 if ref.selection == "hard_noedit" else arm.edit)

    def test_q8_placebo_uses_geometry_tables_even_when_archived_copy_exists(self):
        self.assertEqual(DEFAULTS["geometry"], PLACEMENT / "geometry-online")
        self.assertEqual(DEFAULTS["placebo"], PLACEMENT)
        current = Arm("geometry", 8, 1, "hard_slack_placebo")
        archive = Arm("placebo", 8, 1, "hard_slack_placebo")
        self.write_arm(current, [1, 1, 1, 0])
        self.write_arm(archive, [1, 0, 0, 0])
        write_csv(self.src["geometry"] / "llada21-paired-cost.csv",
                  [dict(**config(current), requests=4, N=16, F=14, Q=128,
                        reserved_seats=4, reserved_strict_hits=1)])
        _, pairs = build(self.src, MANIFEST)
        q8 = next(row for row in pairs if row["reference_policy"] == current.policy)
        self.assertEqual(q8["reference_source"], str(self.src["geometry"]))
        self.assertEqual(q8["reference_correct"], 3)
        self.assertEqual(q8["reference_reserved_hits"], 1)
        for q in (4, 16):
            policy = Arm("placebo", q, 1 if q == 4 else 2, "hard_slack_placebo").policy
            row = next(row for row in pairs if row["reference_policy"] == policy)
            self.assertEqual(row["reference_source"], str(self.src["placebo"]))


if __name__ == "__main__":
    unittest.main()

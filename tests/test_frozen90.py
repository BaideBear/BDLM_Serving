import copy
import json
import unittest

from bdlm_serving.evaluation import frozen, frozen90


class RegisteredCases(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(frozen90.MANIFEST.read_text())
        self.rows = self.manifest["samples"]

    def test_cohorts_and_exclusions(self):
        old = json.loads(frozen.MANIFEST.read_text())
        self.assertEqual(self.rows[:30], old["samples"])
        self.assertEqual(self.manifest["source_sha256"], old["source_sha256"])
        self.assertEqual(self.manifest["original_samples_sha256"], old["samples_sha256"])
        self.assertEqual(self.manifest["scorer_version"], old["scorer_version"])
        frozen90.validate(self.rows)
        for row in self.rows[30:]:
            for source in self.manifest["exclusions"].values():
                self.assertNotIn(row["source_index"], source["indices"][row["dataset"]])

    def test_missing_duplicate_and_mixed_cohorts_rejected(self):
        for change in ("duplicate", "swap", "missing"):
            rows = copy.deepcopy(self.rows)
            if change == "duplicate":
                rows[31]["source_index"] = rows[30]["source_index"]
            elif change == "swap":
                rows[49]["dataset"], rows[0]["dataset"] = rows[0]["dataset"], "math500"
            else:
                rows.pop()
            with self.subTest(change=change), self.assertRaises(ValueError):
                frozen90.validate(rows)

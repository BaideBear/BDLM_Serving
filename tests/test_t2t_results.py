import unittest

from tools.t2t_results import audit_edits, paired_quality


class ScoringTest(unittest.TestCase):
    def test_realized_edits_and_optional_fresh_labels(self):
        events = [dict(stage="bootstrap", q=256, edited=0),
                  dict(stage="selected", q=16, edited=2, revoked=1, selected_fresh=3, edited_in_fresh=1),
                  dict(stage="verify", q=256, edited=1)]
        out = audit_edits(dict(events=events, t2t_edits=3))
        self.assertEqual((out["strict_edits"], out["edited_on_verify"], out["fresh_labeled_hits"]),
                         (3, 1, 1))
        events[1].pop("edited_in_fresh")
        self.assertEqual(audit_edits(dict(events=events, t2t_edits=3))["fresh_labeled_steps"], 0)
        with self.assertRaises(ValueError):
            audit_edits(dict(events=events, t2t_edits=0))

    def test_paired_correctness_is_not_difference_of_averages(self):
        def arm(name, marks):
            return [dict(policy=name, nominal_q=16, fresh=4, mask=8, dataset="d",
                         task_id=f"d/{i}", index=i, correct=c) for i, c in enumerate(marks)]
        rows = [arm("native", [1, 1]), arm("t2t_q16_f4_m8_targeted", [1, 0]),
                arm("t2t_q16_f4_m8_random", [0, 1])]
        pair = next(x for x in paired_quality(rows) if x["equal_nominal_q"])
        self.assertEqual((pair["only_policy_correct"], pair["only_reference_correct"], pair["delta_correct"]),
                         (1, 1, 0))
        rows[2].reverse()
        with self.assertRaises(ValueError):
            list(paired_quality(rows))

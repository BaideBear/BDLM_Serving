import unittest
from copy import deepcopy

from experiments.t2t.serving_slots.spawn_mask_report import commit_pairs


def tape():
    head=list(range(64,96))
    tail=list(range(96,128))
    return [dict(top_action="pair_denoise",block=2,absolute_positions=head+tail),
            dict(top_action="commit_head",nominal_q=64,block=2,offer_tick=4,context_version=1,
                 absolute_positions=head+tail[:3],mandatory_head_positions=[],mandatory_tail_positions=tail[:3],
                 sampled_q=0,physical_q=35),
            dict(top_action="promote_verify",nominal_q=64,block=3,offer_tick=5,context_version=2,
                 absolute_positions=tail,sampled_positions=tail,physical_q=32)]


class CommitAudit(unittest.TestCase):
    def test_real_duplicate_rows_not_a_blanket_32(self):
        c=commit_pairs(tape())
        self.assertEqual((c["adjacent_pairs"],c["duplicate_tail_rows"],c["previous_full_head_without_head_debt"]),
                         (1,3,1))

    def test_terminal_and_fresh_heads_are_not_pairs(self):
        events=tape()[:2]
        self.assertEqual(commit_pairs(events)["terminal_commits"],1)
        c=commit_pairs(events+[dict(top_action="head_bootstrap",block=3)])
        self.assertEqual((c["fresh_head_commits"],c["adjacent_pairs"],c["duplicate_tail_rows"]),(1,0,0))

    def test_reject_wrong_context_geometry_and_hidden_physical_work(self):
        for index,key,value in ((2,"context_version",1),(2,"sampled_positions",[96]),(1,"physical_q",64)):
            with self.subTest(key=key):
                events=deepcopy(tape())
                events[index][key]=value
                with self.assertRaises(ValueError):
                    commit_pairs(events)

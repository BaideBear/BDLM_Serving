"""Regression tests for the CPU-only historical ranked-path study."""

import copy
import unittest

from experiments.t2t.exchange.write_veto.historical import (
    check_first_swap, policy, scan, strata, unique,
)


def selected(clock, *, hit=False, reserve=False, before=2, fresh=0, masks=2,
             picked=(0, 1, 2, 3)):
    return dict(phase="denoise", stage="selected", block=0, context_version=0,
                prefix_kv_length=0, step_in_block=clock, hard_clock=clock,
                q_budget=4, F=1, physical_q=4, actual_q=4, q_tokens=4,
                layer_physical_q=[4]*20, overflow=0, absolute_positions=list(picked),
                mandatory_fresh=fresh, approximate_same_block_kv=True,
                reserved_extra_edit_positions=[3] if reserve else [],
                reserved_strict_edit_positions=[3] if hit else [], hard_strict_edits=int(hit),
                masked_before=before, selected_masked_before=masks, accepted=0,
                masked_after=before, edited=int(hit))


def request(events):
    f = len(events)
    return dict(index=0, dataset="fixture", task_id="fixture/0", seed=17000,
                status="success", output_ids=[42], output_tokens=1, eos=False, truncated=True,
                generation_forwards=f, generation_query_rows=4*f,
                generation_physical_padded_rows=4*f, max_q=4, tpf=1/f, token_efficiency=1/(4*f),
                prefill_rows=0, t2t_edits=sum(e["edited"] for e in events), events=events,
                forwards_by_phase={"selected": f}, q_by_phase={"selected": 4*f}, q_histogram={"4": f})


class TestExchangeVetoHistory(unittest.TestCase):
    def test_hits_count_events_and_first_actual_hit_not_first_reservation(self):
        r = request([selected(1, reserve=True), selected(2, reserve=True, hit=True),
                     selected(3, reserve=True, hit=True)])
        cost, swap, hit, forward, *_ = scan(r, 4, 0, True)
        self.assertEqual((cost["reserved_seats"], cost["reserved_strict_hits"]), (3, 2))
        self.assertEqual((swap, hit["hard_clock"], forward), (1, 2, 2))
        with self.assertRaisesRegex(ValueError, "reserved/strict"):
            scan(r, 4, 0, False)

    def test_rejects_bad_reservation_width_masks_and_totals(self):
        r = request([selected(1, reserve=True, hit=True)])
        for field, value, error in (
            ("reserved_strict_edit_positions", [2], "reserved/strict"),
            ("layer_physical_q", [4]*19 + [5], "Physical width"),
            ("selected_masked_before", 0, "MASK/mandatory"),
            ("masked_after", 3, "MASK-transition"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, error):
                bad = copy.deepcopy(r)
                bad["events"][0][field] = value
                scan(bad, 4, 0, True)
        r["q_by_phase"]["selected"] += 1
        with self.assertRaisesRegex(ValueError, "totals disagree"):
            scan(r, 4, 0, True)

    def test_full_mandatory_selection_can_skip_masks(self):
        cost, swap, hit, *_ = scan(request([selected(1, fresh=4, masks=0)]), 4, 0, True)
        self.assertEqual((cost["mandatory_Q"], cost["mask_Q"], swap, hit), (4, 0, None, None))

    def test_anchor_checks_added_seat_and_no_swap_output(self):
        a = request([selected(1), selected(2, reserve=True)])
        b = request([selected(1), selected(2, picked=(0, 1, 2, 4))])
        self.assertEqual(check_first_swap(a, b, 2), 4)
        b["events"][1]["absolute_positions"] = [0, 1, 3, 4]
        with self.assertRaisesRegex(ValueError, "replace exactly"):
            check_first_swap(a, b, 2)
        b = copy.deepcopy(a)
        b["events"][0]["context_version"] += 1
        with self.assertRaisesRegex(ValueError, "before first swap"):
            check_first_swap(a, b, 2)
        a = request([selected(1)])
        b = copy.deepcopy(a)
        self.assertIsNone(check_first_swap(a, b, None))
        b["output_ids"] = [43]
        with self.assertRaisesRegex(ValueError, "No-swap request diverged"):
            check_first_swap(a, b, None)

    def test_descriptive_strata_keep_zero_hit_flips_and_weighted_delta(self):
        rows = [dict(q=8, rank_correct=a, zero_correct=b, outcome=outcome, hits=h,
                     first_hit_clock=c, first_swap_clock=1, rank_tpf=t, zero_tpf=1, delta_tpf=t-1)
                for a, b, outcome, h, c, t in [(1, 0, "win", 0, None, 1.5),
                                                (0, 1, "loss", 0, None, 0.75),
                                                (1, 1, "both", 2, 129, 1.25)]]
        g = {(r["group"], r["level"]): r for r in strata(rows)}
        self.assertEqual((g["ever_hit", "no"]["win"], g["ever_hit", "no"]["loss"]), (1, 1))
        self.assertEqual(g["hit_count", "2"]["hits"], 2)
        self.assertEqual(g["first_hit_clock", "129+"]["first_hit_clock_median"], 129)
        self.assertAlmostEqual(g["all", "all"]["mean_delta_tpf"], 0.5/3)
        self.assertAlmostEqual(sum(g["ever_hit", k]["contribution_to_all_delta_tpf"] for k in ("yes", "no")),
                               0.5/30)

    def test_policy_and_complete_unique_keys(self):
        self.assertEqual(policy(8, True), "t2t_q8_f0_m7_hard_exchange_edit4")
        self.assertEqual(policy(8, False), "t2t_q8_f0_m8_hard_noedit")
        for rows in ([0, 0, 1], [0]):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "missing or extra"):
                unique(rows, int, (0, 1))


if __name__ == "__main__":
    unittest.main()

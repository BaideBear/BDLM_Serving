import gzip
from itertools import combinations
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.exchange_ceiling import FROZEN, ceiling, counts, evaluate, lock_gate


def subsets(items):
    return [set(s) for n in range(len(items) + 1) for s in combinations(items, n)]


class ExchangeCeilingTest(unittest.TestCase):
    def test_fixed_gate_keeps_width_victim_type_and_original_gate(self):
        state = (list(range(7)),[True,False,False,False,True,True,True],{0},set(),[1]*7,3)
        rates = {("old",):.2, ("mask",):.3}
        base, ranked = {0,1,2,3}, {0,1,2,4}
        for mode,want in (("locked_candidate_oracle",{0,1,2,5}),
                          ("locked_victim_oracle",{0,1,3,4}),("locked_both_oracle",{0,1,3,5}),
                          ("locked_mask_rate",ranked)):
            self.assertEqual(lock_gate(state,rates,base,ranked,{5},{3},mode),want)
            self.assertEqual(lock_gate(state,rates,base,base,{5},{3},mode),base)
    def test_exact_at_every_cost_against_all_physical_swaps(self):
        fresh, masks, old = {0}, [1, 2, 3, 4], {5, 6, 7}
        for base in ({0, 1, 2, 3}, {0, 1, 2, 3, 4, 5}):
            plans = [base - remove | add for remove in subsets(base - fresh - {1})
                     for add in subsets(old - base) if len(remove) == len(add)]
            for targets in subsets(old):
                for admissions in subsets(masks):
                    def score(picked):
                        return (len(picked & targets), -len(admissions & (base - picked)),
                                -len(picked - base), -len(set(masks) & (base - picked)))

                    for cap in (0, 1, 2, None):
                        for budget in (0, 1, 2, 3, None):
                            feasible = [p for p in plans
                                        if (cap is None or len(p - base) <= cap) and
                                        (budget is None or len(admissions & (base - p)) <= budget)]
                            picked = ceiling(base, fresh, masks, targets, admissions,
                                             cap=cap, max_lost=budget)
                            self.assertIn(picked, feasible)
                            self.assertEqual(score(picked), max(map(score, feasible)))

    def test_overflow_protection_and_separate_mask_costs(self):
        overflow = set(range(5))
        self.assertEqual(ceiling(overflow, overflow, [5], {6}, {5}), overflow)
        base = {0, 1, 2, 3}
        fresh, masks, targets, admit = {0}, [1, 2, 3, 4], {5, 6, 7}, {1, 3, 4}
        free = ceiling(base, fresh, masks, targets, admit, max_lost=0)
        paid = ceiling(base, fresh, masks, targets, admit)
        self.assertEqual((free, paid), ({0, 1, 3, 5}, {0, 1, 5, 6}))
        for picked, expected in ((free, (1, 0, 1)), (paid, (2, 1, 1))):
            c = counts(base, picked, set(masks), targets, admit)
            self.assertEqual(tuple(c[k] for k in ("displaced_masks", "lost_mask_admissions",
                                                  "displaced_nonadmitting_masks")), expected)

    def test_native_prompt_bootstrap_verify_and_strict_edit_boundaries(self):
        task = dict(index=0, dataset="toy", task_id="mixed-prompt-block", seed=1)
        prompt = [77] * 26
        canvases = [prompt + tail for tail in ([0, 0, 0, 0, 0, 0], [11, 12, 13, 14, 0, 0],
                                              [21, 12, 13, 0, 16, 0], [21, 22, 13, 19, 16, 20],
                                              [31, 32, 13, 19, 16, 20])]
        # A changed prompt position, even if present in a row, is not a label.
        for canvas in canvases[2:]:
            canvas[25] = 78
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            marker = folder / "verified.json"
            marker.write_text(json.dumps(dict(model="llada21", matched_requests=1)))
            with gzip.open(folder / "00.trace.jsonl.gz", "wt") as f:
                for row in [dict(schema=1, **task), dict(prompt_length=26, mask_id=0)] + [
                    dict(block=0, before=b, after=a, accepted=[x != 0 for x in a])
                    for b, a in zip(canvases, canvases[1:])
                ]:
                    f.write(json.dumps(row) + "\n")
            rates = {("old",): .5, ("mask",): .5}
            requests, _ = evaluate(folder, [task], rates)
            marker.write_text(json.dumps(dict(model="gemma", matched_requests=1)))
            with self.assertRaisesRegex(ValueError, "verified native LLaDA"):
                evaluate(folder, [task], rates)

        by = {(r["q"], r["policy"], r["max_new_seats"]): r for r in requests}
        self.assertEqual(len(by), 30)
        base = by[4, "zero_reserve", 0]
        paid = by[4, "oracle_unconstrained_loss", "uncapped"]
        self.assertEqual((base["states"], base["old_targets"], base["fresh_strict_hits"],
                          base["natural_mask_admissions"], base["physical_q"]), (2, 1, 1, 3, 8))
        self.assertEqual((paid["new_old_hits"], paid["lost_mask_admissions"]), (1, 1))
        self.assertEqual(by[4, "oracle_zero_loss", 1]["selected_old_hits"], 0)
        self.assertEqual(by[4, FROZEN, 1]["selected_old_hits"], 1)
        self.assertEqual(by[8, "zero_reserve", 0]["base_old_hits"], 1)

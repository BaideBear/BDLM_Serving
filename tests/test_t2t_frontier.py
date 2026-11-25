"""CPU coverage for the unified hard-refresh allocator. No model or GPU."""
import time
import unittest

from bdlm_serving.methods.t2t.frontier import select_hard


def run(n, *, resolved=None, fresh=(), write=None, query=None, clock=1,
        budget=0, edit_slots=0, policy="hard_frontier", seed=0, editable=None):
    resolved = [True] * n if resolved is None else list(resolved)
    write = [-1] * n if write is None else list(write)
    query = [-1] * n if query is None else list(query)
    editable = list(range(n)) if editable is None else list(editable)
    return select_hard(editable, resolved, set(fresh), write, query, clock,
                       budget, edit_slots, policy, seed)


class SelectHardTest(unittest.TestCase):
    def assert_total(self, chosen, eligible, budget, fresh):
        self.assertEqual(len(chosen), min(len(eligible), max(budget, len(fresh))))

    def test_budget_mandatory_mask_and_backfill(self):
        chosen, reserve = run(6, resolved=[False, True, True, True, True, True],
                              budget=3, edit_slots=1)
        self.assertEqual(chosen, [0, 1, 2])
        self.assertEqual(reserve, [1])
        self.assert_total(chosen, range(6), 3, set())

    def test_fresh_overflow_is_the_only_overflow(self):
        chosen, reserve = run(4, resolved=[True, True, True, False],
                              fresh={0, 1, 2}, budget=2, edit_slots=1)
        self.assertEqual(chosen, [0, 1, 2])
        self.assertEqual(reserve, [])
        self.assert_total(chosen, range(4), 2, {0, 1, 2})

    def test_fresh_exactly_budget_has_no_extra_mask(self):
        chosen, reserve = run(3, resolved=[True, False, True], fresh={0},
                              budget=1, edit_slots=1)
        self.assertEqual(chosen, [0])
        self.assertEqual(reserve, [])

    def test_mandatory_mask_beats_edit_quota(self):
        chosen, reserve = run(4, resolved=[False, True, True, True],
                              budget=2, edit_slots=2)
        self.assertEqual(chosen, [0, 1])
        self.assertEqual(reserve, [1])

    def test_budget_larger_than_eligible(self):
        chosen, reserve = run(3, budget=10, edit_slots=1)
        self.assertEqual(chosen, [0, 1, 2])
        self.assertEqual(reserve, [0])

    def test_zero_budget(self):
        self.assertEqual(run(3, budget=0, edit_slots=0, policy="hard_noedit"),
                         ([], []))
        self.assertEqual(run(3, budget=0, fresh={1}, edit_slots=0,
                             policy="hard_noedit"), ([1], []))

    def test_backfill_when_edit_pool_is_short(self):
        chosen, reserve = run(4, resolved=[False, True, False, False],
                              budget=4, edit_slots=2)
        self.assertEqual(chosen, [0, 1, 2, 3])
        self.assertEqual(reserve, [1])

    def test_backfill_when_mask_pool_is_short(self):
        chosen, reserve = run(4, resolved=[True, True, True, False],
                              budget=4, edit_slots=1)
        self.assertEqual(chosen, [0, 1, 2, 3])
        self.assertEqual(reserve, [0])

    def test_backfill_outside_reserved_is_ascending(self):
        # The step-5 write at 4 makes position 3 the top frontier candidate.
        # Only that seat is reserved; the other two seats are the lowest
        # remaining positions, not the next-ranked ones.
        chosen, reserve = run(5, write=[-1, -1, -1, -1, 5],
                              query=[-1] * 5, clock=6,
                              budget=3, edit_slots=1)
        self.assertEqual(chosen, [0, 1, 3])
        self.assertEqual(reserve, [3])

    def test_noedit_reserves_nothing(self):
        chosen, reserve = run(4, resolved=[False, True, True, True],
                              budget=2, edit_slots=0, policy="hard_noedit")
        self.assertEqual(chosen, [0, 1])
        self.assertEqual(reserve, [])
        chosen, reserve = run(4, budget=3, edit_slots=0, policy="hard_noedit")
        self.assertEqual(chosen, [0, 1, 2])
        self.assertEqual(reserve, [])

    def test_same_round_compute_then_write_counts(self):
        # Position 2 is queried and written at step 5.  Candidate 1 was also
        # queried at step 5, so the write must count for it (>=, not >).
        chosen, reserve = run(4, write=[-1, -1, 5, -1],
                              query=[-1, 5, -1, 4], clock=6,
                              budget=1, edit_slots=1)
        self.assertEqual(chosen, [1])
        self.assertEqual(reserve, [1])

    def test_write_before_last_query_does_not_count(self):
        # The only write is at step 4, while every candidate was last queried
        # at step 5.  No candidate has a signal; ties fall back to position 0.
        chosen, reserve = run(3, write=[-1, -1, 4], query=[5, 5, 5],
                              clock=6, budget=1, edit_slots=1)
        self.assertEqual(chosen, [0])
        self.assertEqual(reserve, [0])

    def test_no_history_is_positional(self):
        chosen, reserve = run(4, budget=2, edit_slots=2)
        self.assertEqual(chosen, [0, 1])
        self.assertEqual(reserve, [0, 1])

    def test_tie_breaks_by_position(self):
        # Positions 0 and 2 are symmetric around the step-5 write at 1.
        chosen, reserve = run(3, write=[-1, 5, -1], query=[5, 5, 5],
                              clock=6, budget=1, edit_slots=1)
        self.assertEqual(chosen, [0])
        self.assertEqual(reserve, [0])

    def test_random_same_pool_and_count(self):
        args = dict(write=[-1, 5, -1, -1], query=[-1, 5, -1, 5],
                    clock=6, budget=2, edit_slots=2)
        _, frontier = run(4, policy="hard_frontier", **args)
        self.assertEqual(frontier, [0, 2])
        seen = set()
        for seed in range(10):
            chosen, reserve = run(4, policy="hard_random", seed=seed, **args)
            self.assertEqual(len(reserve), 2)
            self.assertTrue(set(reserve).issubset({0, 1, 2, 3}))
            self.assertEqual(chosen, sorted(reserve))
            seen.add(tuple(reserve))
        self.assertGreater(len(seen), 1)
        self.assertEqual(run(4, policy="hard_random", seed=3, **args),
                         run(4, policy="hard_random", seed=3, **args))

    def test_random_seed_includes_clock(self):
        subsets = set()
        for clock in range(1, 6):
            chosen, reserve = run(8, policy="hard_random", seed=7, clock=clock,
                                  budget=3, edit_slots=3)
            self.assertEqual(len(reserve), 3)
            self.assertEqual(chosen, sorted(reserve))
            subsets.add(tuple(reserve))
        self.assertGreater(len(subsets), 1)
        self.assertEqual(
            run(8, policy="hard_random", seed=7, clock=3, budget=3, edit_slots=3),
            run(8, policy="hard_random", seed=7, clock=3, budget=3, edit_slots=3))

    def test_random_has_no_implicit_history_gate(self):
        # Positions 0 and 2 have never been written; a full-size random draw
        # must still be able to take them.
        chosen, reserve = run(3, write=[-1, 5, -1], query=[-1, 5, -1],
                              clock=6, budget=3, edit_slots=3,
                              policy="hard_random", seed=0)
        self.assertEqual(chosen, [0, 1, 2])
        self.assertEqual(reserve, [0, 1, 2])

    def test_b256_smoke(self):
        n = 256
        start = time.perf_counter()
        chosen, reserve = run(n, write=[i % 10 for i in range(n)],
                              query=[-1] * n, clock=10, budget=n,
                              edit_slots=64, policy="hard_frontier")
        elapsed = time.perf_counter() - start
        self.assertEqual(len(chosen), n)
        self.assertEqual(len(reserve), 64)
        self.assertLess(elapsed, 2.0)

    def test_validation(self):
        with self.assertRaises(ValueError):
            run(2, budget=1, edit_slots=1, policy="unknown")
        with self.assertRaises(ValueError):
            run(2, budget=1, edit_slots=1, fresh={5})
        with self.assertRaises(ValueError):
            select_hard([0], [True], set(), [-1], [-1, -1], 0, 1, 1,
                        "hard_frontier", 0)
        with self.assertRaises(ValueError):
            run(2, budget=1, edit_slots=0, policy="hard_frontier")
        with self.assertRaises(ValueError):
            run(2, budget=1, edit_slots=1, policy="hard_noedit")
        with self.assertRaises(ValueError):
            run(2, budget=-1, edit_slots=1, policy="hard_frontier")
        with self.assertRaises(ValueError):
            run(2, budget=1, edit_slots=1, write=[3, -1], clock=2)


if __name__ == "__main__":
    unittest.main()

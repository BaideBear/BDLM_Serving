"""Check the swap contract using a scripted baseline and targeted selector."""
import unittest
from unittest import mock

from bdlm_serving.methods.t2t import intervention


class FakeSelector:
    """Scripted stand-in for ``frontier.select_hard``."""

    def __init__(self, base, targeted, reserved):
        self.base = list(base)
        self.targeted = list(targeted)
        self.reserved = list(reserved)
        self.calls = []

    def __call__(self, editable, resolved, fresh, last_write, last_query,
                 clock, budget, edit_slots, policy, seed, *, previous_strict=()):
        self.calls.append(dict(
            editable=tuple(editable),
            resolved=tuple(resolved),
            fresh=set(fresh),
            last_write=tuple(last_write),
            last_query=tuple(last_query),
            clock=clock,
            budget=budget,
            edit_slots=edit_slots,
            policy=policy,
            seed=seed,
            previous_strict=tuple(previous_strict),
        ))
        if policy == "hard_noedit":
            return list(self.base), []
        return list(self.targeted), list(self.reserved)


class PlanSeatTest(unittest.TestCase):
    def setUp(self):
        self._original = intervention.select_hard
        self.addCleanup(setattr, intervention, "select_hard", self._original)
        self.selector = None

    def use(self, base, targeted, reserved):
        self.selector = FakeSelector(base, targeted, reserved)
        intervention.select_hard = self.selector
        return self.selector

    def make_plan(self, *, eligible, resolved, fresh, last_write=None,
                  last_query=None, clock=9, budget=4, model="llada",
                  seed="seed", previous_strict=()):
        if last_write is None:
            last_write = [-1] * len(resolved)
        if last_query is None:
            last_query = [-1] * len(resolved)
        return intervention.plan_seat(
            eligible, resolved, fresh, last_write, last_query, clock,
            budget, model, seed, previous_strict=previous_strict,
        )

    def test_llada_slack_swap(self):
        resolved = [True, True, True, True, False, False]
        selector = self.use(
            base=[0, 2, 4, 5], targeted=[0, 1, 4, 5], reserved=[1],
        )
        plan = self.make_plan(eligible=range(6), resolved=resolved, fresh={0})
        self.assertIsNotNone(plan)
        self.assertEqual(plan["base"], [0, 2, 4, 5])
        self.assertEqual(plan["targeted"], [0, 1, 4, 5])
        self.assertEqual(plan["victim"], 2)
        self.assertEqual(plan["candidate"], 1)
        self.assertEqual(plan["stratum"], "slack")
        self.assertEqual(plan["fresh"], [0])
        self.assertEqual(plan["masks"], [4, 5])
        self.assertEqual(plan["clock"], 9)
        self.assertIn(plan["random_candidate"], {1, 3})
        self.assertEqual(
            plan["random"], sorted({0, 4, 5, plan["random_candidate"]}))
        self.assertEqual(len(selector.calls), 2)
        self.assertEqual(selector.calls[0]["policy"], "hard_noedit")
        self.assertEqual(selector.calls[0]["edit_slots"], 0)
        self.assertEqual(selector.calls[1]["policy"], "hard_slack_geometry")
        self.assertEqual(selector.calls[1]["edit_slots"], 1)

    def test_gemma_mask_swap(self):
        resolved = [True, True, True, False, False, False]
        selector = self.use(
            base=[0, 1, 3, 4], targeted=[0, 1, 2, 4], reserved=[2],
        )
        plan = self.make_plan(
            eligible=range(6), resolved=resolved, fresh={0}, model="gemma",
            previous_strict={1},
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan["victim"], 3)
        self.assertEqual(plan["candidate"], 2)
        self.assertEqual(plan["stratum"], "mask")
        self.assertEqual(plan["targeted"], [0, 1, 2, 4])
        self.assertEqual(plan["random_candidate"], 2)
        self.assertEqual(plan["random"], [0, 1, 2, 4])
        self.assertEqual(plan["masks"], [3, 4, 5])
        self.assertEqual(selector.calls[1]["policy"], "hard_trigger")
        self.assertEqual(selector.calls[1]["edit_slots"], 4)

    def test_no_event_when_targeted_reserves_nothing(self):
        self.use(base=[0, 1, 2], targeted=[0, 1, 2], reserved=[])
        plan = self.make_plan(
            eligible=range(3), resolved=[True, True, True], fresh={0},
        )
        self.assertIsNone(plan)

    def test_no_event_when_reservation_is_already_in_base(self):
        # The targeted policy reserved a seat, but the no-edit baseline
        # already backfills it.  There is no physical difference.
        self.use(base=[0, 1, 2], targeted=[0, 1, 2], reserved=[1])
        plan = self.make_plan(
            eligible=range(3), resolved=[True, True, True], fresh={0},
        )
        self.assertIsNone(plan)

    def test_no_event_when_last_mask_would_be_displaced(self):
        self.use(base=[0, 2], targeted=[0, 1], reserved=[1])
        plan = self.make_plan(
            eligible=range(4), resolved=[True, True, False, False],
            fresh={0}, model="gemma",
        )
        self.assertIsNone(plan)

    def test_deterministic_candidate_and_victim(self):
        self.use(base=[0, 2, 3], targeted=[0, 4, 5], reserved=[4, 5])
        plan = self.make_plan(
            eligible=range(6), resolved=[True] * 6, fresh={0},
        )
        self.assertEqual(plan["candidate"], 4)  # lowest reserved outside base
        self.assertEqual(plan["victim"], 3)     # highest dropped old row
        self.assertEqual(plan["targeted"], [0, 2, 4])
        self.assertEqual(plan["stratum"], "slack")

    def test_random_arm_can_collide_with_candidate(self):
        self.use(base=[0, 2, 3], targeted=[0, 1, 2], reserved=[1])
        plan = self.make_plan(
            eligible=range(4), resolved=[True] * 4, fresh={0},
        )
        self.assertEqual(plan["candidate"], 1)
        self.assertEqual(plan["random_candidate"], 1)
        self.assertEqual(plan["targeted"], [0, 1, 2])
        self.assertEqual(plan["random"], [0, 1, 2])

    def test_random_pool_is_all_nonfresh_old_outside_base(self):
        self.use(base=[0, 2, 3], targeted=[0, 4, 5], reserved=[4])
        instances = []

        class FakeRandom:
            def __init__(self, seed):
                self.seed = seed
                self.pool = None
                instances.append(self)

            def choice(self, seq):
                self.pool = list(seq)
                return seq[0]

        with mock.patch.object(intervention.random, "Random", FakeRandom):
            plan = self.make_plan(
                eligible=range(6), resolved=[True] * 6, fresh={0},
                seed="s", clock=9,
            )
        self.assertEqual(instances[0].seed, "seat-random:s:9")
        self.assertEqual(instances[0].pool, [1, 4, 5])
        self.assertEqual(plan["random_candidate"], 1)
        self.assertEqual(plan["random"], [0, 1, 2])

    def test_model_selects_policy_and_cap(self):
        cases = (
            ("llada", "hard_slack_geometry", 1),
            ("llada21", "hard_slack_geometry", 1),
            ("LLaDA2.1", "hard_slack_geometry", 1),
            ("gemma", "hard_trigger", 4),
            ("DiffusionGemma", "hard_trigger", 4),
        )
        for model, policy, slots in cases:
            with self.subTest(model=model):
                selector = self.use(base=[0, 1], targeted=[0, 1], reserved=[])
                self.make_plan(
                    eligible=range(2), resolved=[True, True], fresh={0},
                    model=model,
                )
                self.assertEqual(selector.calls[1]["policy"], policy)
                self.assertEqual(selector.calls[1]["edit_slots"], slots)

    def test_unknown_model_rejected_before_selector(self):
        selector = self.use(base=[], targeted=[], reserved=[])
        with self.assertRaises(ValueError):
            self.make_plan(
                eligible=range(2), resolved=[True, True], fresh={0},
                model="unknown",
            )
        self.assertEqual(selector.calls, [])

    def test_arguments_are_passed_to_both_selector_calls(self):
        selector = self.use(base=[0, 1], targeted=[0, 1], reserved=[])
        self.make_plan(
            eligible={1, 0},  # deliberately unsorted
            resolved=[True, True],
            fresh={0},
            last_write=[0, -1],
            last_query=[1, 0],
            clock=11,
            budget=2,
            model="gemma",
            seed="req-7",
            previous_strict={1},
        )
        self.assertEqual(len(selector.calls), 2)
        for call in selector.calls:
            self.assertEqual(call["editable"], (0, 1))
            self.assertEqual(call["resolved"], (True, True))
            self.assertEqual(call["fresh"], {0})
            self.assertEqual(call["last_write"], (0, -1))
            self.assertEqual(call["last_query"], (1, 0))
            self.assertEqual(call["clock"], 11)
            self.assertEqual(call["budget"], 2)
            self.assertEqual(call["seed"], "req-7")
            self.assertEqual(call["previous_strict"], (1,))

    def test_input_validation(self):
        selector = self.use(base=[0], targeted=[0], reserved=[])
        cases = (
            dict(eligible=range(2), resolved=[True, False], fresh={1}),
            dict(eligible=range(1), resolved=[True, False], fresh={1}),
            dict(eligible=range(2), resolved=[True, False], fresh={0},
                 previous_strict={1}),
            dict(eligible=range(2), resolved=[True, False], fresh={0},
                 previous_strict={2}),
            dict(eligible=range(2), resolved=[True, True], fresh={0},
                 last_write=[-1]),
            dict(eligible=range(2), resolved=[True, True], fresh={0},
                 last_write=[2, -1], clock=1),
            dict(eligible=range(2), resolved=[True, True], fresh={0},
                 budget=-1),
            dict(eligible=range(2), resolved=[True, True], fresh={0},
                 clock=-1),
            dict(eligible=range(2), resolved=[True, True], fresh={0},
                 budget=True),
        )
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    self.make_plan(**case)
        self.assertEqual(selector.calls, [])

    def test_selector_contract_violations_are_rejected(self):
        # Duplicate baseline position.
        self.use(base=[0, 0], targeted=[0], reserved=[])
        with self.assertRaises(ValueError):
            self.make_plan(
                eligible=range(1), resolved=[True], fresh={0},
            )
        # Baseline dropped a mandatory fresh row.
        self.use(base=[1], targeted=[0], reserved=[])
        with self.assertRaises(ValueError):
            self.make_plan(
                eligible=range(2), resolved=[True, True], fresh={0},
            )
        # Reserved position is not selected by the targeted policy.
        self.use(base=[0], targeted=[0], reserved=[1])
        with self.assertRaises(ValueError):
            self.make_plan(
                eligible=range(2), resolved=[True, True], fresh={0},
            )
        # Reserved a mandatory fresh row.
        self.use(base=[0], targeted=[0], reserved=[0])
        with self.assertRaises(ValueError):
            self.make_plan(
                eligible=range(1), resolved=[True], fresh={0},
            )

    def test_fresh_overflow_is_returned_without_event(self):
        self.use(base=[0, 1, 2], targeted=[0, 1, 2], reserved=[])
        plan = self.make_plan(
            eligible=range(3), resolved=[True, True, True],
            fresh={0, 1, 2}, budget=2,
        )
        self.assertIsNone(plan)

    def test_masks_lists_all_eligible_unresolved(self):
        self.use(base=[0, 2, 3], targeted=[0, 1, 2], reserved=[1])
        plan = self.make_plan(
            eligible=range(6),
            resolved=[True, True, True, True, False, False],
            fresh={0},
        )
        self.assertEqual(plan["masks"], [4, 5])
        self.assertEqual(plan["stratum"], "slack")


if __name__ == "__main__":
    unittest.main()

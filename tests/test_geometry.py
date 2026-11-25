import unittest

from bdlm_serving.methods.t2t.geometry import rank_old


def rank(pool, *, masks=(), fresh=(), strict=(), writes=None, queries=None,
         editable=range(12), clock=6):
    writes = dict(writes or {})
    for i in set(fresh) | set(strict):
        writes.setdefault(i, 5)
    queries = queries or {}
    return rank_old(
        pool, editable, [i not in masks for i in range(12)], fresh,
        [writes.get(i, -1) for i in range(12)],
        [queries.get(i, -1) for i in range(12)], clock, strict,
    )


class GeometryTest(unittest.TestCase):
    def test_joint_mask_fresh_distance_beats_mask_only(self):
        self.assertEqual(rank([2, 6, 4], masks={1}, fresh={7}), [4, 2, 6])
        self.assertEqual(rank([2, 6, 4], masks={1}), [2, 4, 6])

    def test_same_clock_is_dirty_but_newer_query_covers_write(self):
        self.assertEqual(rank([2, 4], masks={1}, fresh={7}, queries={4: 5}), [4, 2])
        self.assertEqual(rank([2, 4], masks={1}, fresh={7}, queries={4: 6}), [2, 4])

    def test_nearby_previous_strict_can_beat_intersection(self):
        args = dict(masks={1}, fresh={7, 11})
        self.assertEqual(rank([4, 10], **args), [4, 10])
        self.assertEqual(rank([4, 10], strict={11}, **args), [10, 4])
        self.assertEqual(rank([4, 10], strict={11}, queries={10: 6}, **args), [4, 10])

    def test_previous_strict_uses_distance_and_age(self):
        # At equal risk (3) prefer the intersection. One tick fresher wins.
        args = dict(masks={1}, fresh={7, 11}, strict={11})
        self.assertEqual(rank([10, 4], writes={11: 4}, **args), [4, 10])
        self.assertEqual(rank([10, 4], writes={11: 5}, **args), [10, 4])

    def test_ineligible_masks_do_not_attract_prompt_adjacent_rows(self):
        self.assertEqual(rank([2, 4], masks={1, 7}, editable=range(2, 12)), [4, 2])

    def test_missing_anchors_use_self_write_then_query_then_position(self):
        self.assertEqual(rank([7, 5, 2, 4], writes={4: 3, 5: 3}, queries={4: 2}),
                         [5, 4, 2, 7])
        self.assertEqual(rank([4, 3], masks={1}, fresh={7}, writes={7: -1}), [3, 4])
        self.assertEqual(rank([], masks={1}, fresh={7}), [])

    def test_preserves_candidate_subset_and_input(self):
        pool = [10, 8]
        self.assertEqual(rank(pool, masks={6}, fresh={7}), [8, 10])
        self.assertEqual(pool, [10, 8])

    def test_invalid_pool_and_clocks(self):
        for pool in ([1], [6], [7], [2, 2]):
            with self.subTest(pool=pool), self.assertRaises(ValueError):
                rank(pool, masks={6}, fresh={7}, editable=range(2, 12))
        with self.assertRaises(ValueError):
            rank([2], writes={2: 7})
        with self.assertRaises(ValueError):
            rank_old([0], [0], [True], (), [], [-1], 1, ())


if __name__ == "__main__":
    unittest.main()

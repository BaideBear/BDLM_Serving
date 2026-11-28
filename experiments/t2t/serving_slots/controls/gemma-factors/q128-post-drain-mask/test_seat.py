import unittest

from seat import FULL, choose


class SeatTest(unittest.TestCase):
    def test_farthest_prefix_for_first_omitted(self):
        self.assertEqual(choose({0, 2, 5, 8}, {0, 2, 5, 8}, {0, 2}, set(),
                                {5, 8, 9, 10}, FULL), (0, 9))

    def test_does_not_touch_frontier_or_debt(self):
        self.assertIsNone(choose({0, 8}, {0, 8}, {0}, set(), {5, 8, 9}, FULL))
        self.assertIsNone(choose({0, 5}, {0, 5}, {0}, {0}, {5, 9}, FULL))

    def test_no_unresident_or_already_selected_receiver(self):
        self.assertIsNone(choose({0, 5}, {0, 5}, {0}, set(), {5, 9}, {0, 5}))

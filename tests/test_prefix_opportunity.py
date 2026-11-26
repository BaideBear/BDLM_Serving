import unittest

from tools.prefix_opportunity import stable_prefix


class PrefixTest(unittest.TestCase):
    def test_prompt_and_unchanged_unaccepted(self):
        self.assertEqual(stable_prefix(range(1, 5), [False, True, True, False, True],
                                       set(), [9]*5, [9]*5), [1, 2])

    def test_same_id_newly_accepted_breaks_prefix(self):
        self.assertEqual(stable_prefix(range(4), [True]*4, {1}, [6]*4, [6]*4), [0])

    def test_changed_input_breaks_even_if_accepted(self):
        self.assertEqual(stable_prefix(range(3), [True]*3, set(), [4, 5, 6], [1, 5, 6]), [])

import unittest

from tools.gemma_ibs_bridge_summary import paired_uncertainty


class IntervalUnits(unittest.TestCase):
    def test_repeating_same_pair_does_not_multiply_mean_delta(self):
        rows = [
            dict(dataset="test", index=i,
                 off=dict(correct=False, N=4, F=4, decoder_Q=16),
                 on=dict(correct=True, N=6, F=3, decoder_Q=16))
            for i in range(4)
        ]
        ci = paired_uncertainty(rows, repeats=80)
        self.assertEqual(ci, dict(correct_delta_pp=[100., 100.],
                                  macro_TPF_delta=[1., 1.],
                                  pooled_TE_delta_pp=[12.5, 12.5]))

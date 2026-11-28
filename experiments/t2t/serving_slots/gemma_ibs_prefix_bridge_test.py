"""No GPU or model required."""
import unittest
from .gemma_ibs_prefix_bridge import suffix


class TestBridge(unittest.TestCase):
    def base(self, rows, seen=None, dirty=None):
        return suffix(rows,set(range(12)),dirty or set(),set(range(21)) if seen is None else seen,10,32)

    def test_warm_suffix(self):
        self.assertEqual(self.base([11,18,19]),[6,7,8,9])

    def test_prefix_must_be_contiguous(self):
        self.assertEqual(self.base([7,11,18]),[8,9])
        self.assertEqual(self.base([9,11,18]),[])

    def test_no_mask_or_cold_masks(self):
        self.assertEqual(self.base([10,11]),[])
        self.assertEqual(self.base([11,18],seen=set(range(12))),[])

    def test_full_or_one_spare(self):
        self.assertEqual(self.base(list(range(10,42))),[])
        self.assertEqual(self.base(list(range(10,41))),[9])

    def test_protect_debt(self):
        self.assertEqual(self.base([11,18],dirty={10}),[])
        with self.assertRaises(ValueError):
            self.base([11,18],dirty={9})

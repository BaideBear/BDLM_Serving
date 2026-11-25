import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from experiments.t2t.calibrated.history.study import METRICS, NAME, PastEdits, paired, passes
from tools.edit_opportunity import load


class HistoryTest(unittest.TestCase):
    def test_four_completed_rounds_sides_self_and_lifetime_clip(self):
        h = PastEdits()
        self.assertEqual(h.features(8), (0, 0, 0, 0, 0))
        for _ in range(4):
            h.observe({3, 4, 8, 9, 12, 13}, {3, 4, 7, 8, 9, 12, 13})
        self.assertEqual(h.features(8), (2, 1, 4, 8, 16))
        h.observe(set(), set())
        self.assertEqual(h.features(8), (2, 1, 3, 6, 12))
        for _ in range(3):
            h.observe(set(), set())
        self.assertEqual(h.features(8), (2, 1, 0, 0, 0))
        h.observe({31}, {31})
        self.assertEqual(h.features(0), (0, 0, 0, 0, 0))

    def test_loader_alignment_boundaries_and_no_current_label(self):
        task = dict(index=0, task_id='toy', dataset='test', seed=7)
        def features(edit):
            prompt = [5] * 26
            a = prompt + [0] * 6
            b = prompt + [1, 2, 3, 4, 0, 0]
            c = [5] * 25 + [6, 1, 7, 8, 0, 9, 0]  # prompt change and a revocation
            d = c.copy()
            if edit:
                d[28] += 1
            block2 = [0] * 32
            block2[28] = 17
            rows = [(0, a, b), (0, b, c), (0, c, c), (0, c, d),
                    (32, [0] * 32, block2), (32, block2, block2), (32, block2, block2)]
            with TemporaryDirectory() as tmp:
                path = Path(tmp) / 'trace.gz'
                with gzip.open(path, 'wt') as f:
                    for row in [task, dict(mask_id=0, prompt_length=26)] + [
                            dict(block=i, before=x, after=y) for i, x, y in rows]:
                        f.write(json.dumps(row) + '\n')
                augmented = load(path, task, history_factory=PastEdits)
                base = load(path, task)
            for full, orig in zip(augmented, base):
                if orig[3]:
                    np.testing.assert_array_equal(full[-2][:, :-5], orig[-2])
            row = augmented[3]
            np.testing.assert_array_equal(row[-2][row[3].index(28), -5:], (1, 1, 1, 0, 5))
            np.testing.assert_array_equal(augmented[-1][-2][0, -5:], (0, 0, 0, 0, 0))
            return row[-2], row[-1]
        x, y = features(True)
        x0, y0 = features(False)
        np.testing.assert_array_equal(x, x0)
        self.assertEqual((y, y0), ({28}, set()))

    def test_paired_request_resampling_and_decision(self):
        tasks = [dict(index=i, dataset=ds) for i, ds in enumerate(('a', 'a', 'b', 'b'))]
        # Opposite changes within each two-request stratum, identical across metrics.
        rows = [dict(index=i, model=model, metric=m, hits=10*i + (2*i % 4 if model == NAME else 1))
                for i in range(4) for m in METRICS for model in ('position', NAME)]
        result = paired(rows, tasks)
        self.assertTrue(all((r['delta'], r['lo95'], r['hi95']) == (0, -4, 4) for r in result))
        def gate(physical):
            return passes([dict(metric=m, delta=v) for m, v in zip(METRICS, (9, 9, 9, *physical))])
        self.assertTrue(gate((1, 0, 1)))
        self.assertFalse(gate((4, 1, -1)))
        self.assertFalse(gate((0, 0, 5)))

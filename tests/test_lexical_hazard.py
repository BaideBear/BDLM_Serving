import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from tools.edit_opportunity import load
from tools.lexical_hazard import check_split, lexical_class, measurements


class TestLexicalHazard(unittest.TestCase):
    def test_taxonomy(self):
        cases = {"": "other", "\ufffd": "other", "\U0001f680": "other",
                 " \t\n": "whitespace", " 37": "digit", "\uff11": "digit",
                 " def": "alpha", "x2": "mixed", "foo_bar": "mixed",
                 "a b": "mixed", " != ": "punct_operator", "\u2264": "punct_operator"}
        for s, expected in cases.items():
            with self.subTest(s=s):
                self.assertEqual(lexical_class(s), expected)

    def test_only_pre_step_old_token(self):
        # At t3, slot 29 is old, 30 fresh, and 31 masked.
        task = dict(index=0, seed=7, task_id="x", dataset="x")
        mask = 99
        def step(prev, nxt):
            return dict(block=0, before=[0]*29+prev, after=[0]*29+nxt)
        a, b, c = [mask]*3, [11, mask, mask], [11, 22, mask]
        def extract(end):
            with TemporaryDirectory() as root:
                path = Path(root)/"test.gz"
                with gzip.open(path, "wt") as f:
                    for row in [task, dict(prompt_length=29, mask_id=mask),
                                step(a,b), step(b,c), step(c,end)]:
                        f.write(json.dumps(row)+"\n")
                observed = []
                def encode(t):
                    observed.append(t)
                    return (t == 11, t != 11)
                path_aug = load(path, task, old_token_features=encode)
                base = load(path, task)
                self.assertEqual(observed, [11])
                np.testing.assert_array_equal(path_aug[-1][-2][:,:-2], base[-1][-2])
                return path_aug
        changed = extract([33,22,mask])
        unchanged = extract(c)
        np.testing.assert_array_equal(changed[-1][-2], unchanged[-1][-2])
        self.assertEqual(changed[-1][-1], {29})
        self.assertFalse(unchanged[-1][-1])
        scores = dict(measurements(changed, np.zeros(27)))
        self.assertEqual(scores["Q4/1"]["hits"], 1)
        self.assertEqual(scores["top-8"]["seats"], 1)

    def test_duplicate_source_and_prompt_rejected(self):
        def tasks(n, start):
            return [dict(index=d*n+i, task_id=f"{ds}/{start+i}", dataset=ds,
                         source_index=start+i, prompt_sha256=f"{ds}/{start+i}")
                    for d, ds in enumerate(("gsm8k","math500","humaneval"))
                    for i in range(n)]
        train, test = tasks(8,0), tasks(10,20)
        check_split(train, test)
        for key in ("source_index", "prompt_sha256"):
            broken = [dict(test[0], **{key:train[0][key]}), *test[1:]]
            with self.assertRaises(ValueError):
                check_split(train, broken)

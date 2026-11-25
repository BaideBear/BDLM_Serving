import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from experiments.t2t.calibrated.embedding_context.study import KEY, Projector, context
from tools.edit_opportunity import load


def test_projection_from_input_only_deterministic_unit_and_zero(tmp_path):
    table = torch.tensor([[0., 0.], [1., -2.]], dtype=torch.bfloat16)
    path = tmp_path / 'toy.safetensors'
    save_file({KEY: table, 'lm_head.weight': torch.zeros_like(table)}, path)
    with safe_open(str(path), framework='pt', device='cpu') as f:
        p = Projector(f.get_slice(KEY))
        signs = (2 * np.random.Generator(np.random.PCG64(20261126)).integers(
            0, 2, (2, 16), dtype=np.int8) - 1) / np.sqrt(2.)
        reference = np.array([1., -2.]) @ signs
        np.testing.assert_allclose(p.unit(1), reference / np.linalg.norm(reference), rtol=1e-6)
        np.testing.assert_array_equal(p.unit(0), np.zeros(16))
        assert p.unit(1) is p.unit(1)


def test_negative_max_distance_nearmask_and_empty():
    vectors = {0: [-1., 0.], 1: [1., 0.], 2: [0., 0.], 3: [-.6, .8]}
    pos = np.zeros((3, 25)); pos[0, 6] = 1
    # At 8 both negative scores compete; at 24 only the neighbour at 16 counts.
    # ID 99 is out of radius or has no neighbour and must not be fetched.
    a = context(vectors.__getitem__, [8, 24, 31], [1, 2, 99], [(0, 0), (16, 3)], pos)
    np.testing.assert_allclose(a, [[-.6, -.6, 1], [0, 0, 1], [0, 0, 0]])
    b = context(vectors.__getitem__, [8], [1], [(16, 0)], pos[1:2])
    np.testing.assert_array_equal(b, [[-1, 0, 1]])
    for fresh in ([], [(8, 99), (17, 99)]):
        np.testing.assert_array_equal(context(vectors.__getitem__, [8], [99], fresh, pos[:1]), [[0, 0, 0]])


def test_pre_step_only_prompt_and_block_reset():
    task = dict(index=0, dataset='toy', task_id='toy', seed=7)
    def capture(edit):
        a = [1] * 28 + [0, 0, 0, 0]
        b = a[:28] + [4, 5, 0, 0]
        c = b[:27] + [12, 4, 7, 9, 0]  # prompt change excluded; rewrite 29, admit 30
        d = c.copy(); d[28] = 15 if edit else 4
        e = [0] * 32; e[28] = 11
        states = [(0, a, b), (0, b, c), (0, c, d), (32, e, e)]
        seen = []
        def callback(old, ids, fresh, pos):
            seen.append((old, ids, fresh, pos.copy()))
            return np.zeros((len(old), 3))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'trace.gz'
            with gzip.open(path, 'wt') as f:
                for row in [task, dict(prompt_length=28, mask_id=0)] + [
                        dict(block=z, before=x, after=y) for z, x, y in states]:
                    f.write(json.dumps(row) + '\n')
            aug, base = load(path, task, context_features=callback), load(path, task)
            for step, orig in zip(aug, base):
                if orig[3]:
                    np.testing.assert_array_equal(step[-2][:, :-3], orig[-2])
            assert seen[0][:3] == ([28], [4], [(29, 7), (30, 9)])
            assert seen[0][3][0, 6] == 1
            assert seen[1][:3] == ([28], [11], [])
            return seen[0][3], aug[2][-1]
    x, y = capture(True)
    x0, y0 = capture(False)
    np.testing.assert_array_equal(x, x0)
    assert (y, y0) == ({28}, set())

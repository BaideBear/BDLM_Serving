"""Frozen LLaDA input-embedding context, shared by calibration and live ranking."""
from contextlib import contextmanager
import json
from time import perf_counter_ns

import numpy as np
from safetensors import safe_open

from bdlm_serving.evaluation.frozen import ROOT, sha


KEY = 'model.word_embeddings.weight'
DESIGN = dict(schema='llada-input-context-fixed-v1', input_key=KEY,
              dimensions=16, seed=20261126, radius=8, near_mask_radius=4)
CALIBRATION = ROOT / 'experiments/t2t/calibrated/weights.json'


@contextmanager
def input_table(checkpoint):
    archived = json.loads((ROOT / 'benchmarks/checkpoints/llada21.json').read_text())
    known = {r['name']: r for r in archived}
    for name in ('model.safetensors.index.json', 'config.json'):
        if sha(checkpoint / name) != known[name]['sha256']:
            raise ValueError(f'Unexpected LLaDA checkpoint: {name}')
    cfg = json.loads((checkpoint / 'config.json').read_text())
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())
    shard = index['weight_map'][KEY]
    if (checkpoint / shard).stat().st_size != known[shard]['bytes']:
        raise ValueError('Unexpected input embedding shard size')
    with safe_open(str(checkpoint / shard), framework='pt', device='cpu') as f:
        t = f.get_slice(KEY)
        if t.get_shape() != [cfg['vocab_size'], cfg['hidden_size']] or t.get_dtype() != 'BF16':
            raise ValueError('Unexpected input embedding shape or dtype')
        yield t


class Projector:
    def __init__(self, table):
        self.table = table
        self.size, width = table.get_shape()
        rng = np.random.Generator(np.random.PCG64(20261126))
        signs = rng.integers(0, 2, size=(width, 16), dtype=np.int8)
        self.matrix = (2 * signs - 1).astype(np.float32) / np.float32(np.sqrt(width))
        self.cache = {}
        self.miss_ns = 0

    def unit(self, token):
        if token not in self.cache:
            if not 0 <= token < self.size:
                raise ValueError('Visible ID outside input embedding')
            start = perf_counter_ns()
            row = self.table[token:token+1].float().numpy()[0]
            v = row @ self.matrix
            norm = np.linalg.norm(v)
            if not np.isfinite(norm):
                raise ValueError('Nonfinite input embedding projection')
            self.cache[token] = v / norm if norm else v
            self.miss_ns += perf_counter_ns() - start
        return self.cache[token]


def context(unit, old, ids, fresh, position):
    """Rows are aligned to old; fresh contains visible (position, ID) pairs."""
    features = np.zeros((len(old), 3), dtype=np.float32)
    if not old or not fresh:
        return features
    loc, new_ids = zip(*fresh)
    dist = np.abs(np.subtract.outer(old, loc))
    eligible = (dist >= 1) & (dist <= 8)
    has = eligible.any(axis=1)
    if has.any():
        used = eligible.any(axis=0)
        old_vec = np.stack([unit(ids[i]) for i in np.flatnonzero(has)])
        new_vec = np.stack([unit(new_ids[j]) for j in np.flatnonzero(used)])
        dots = old_vec @ new_vec.T
        scores = np.where(eligible[has][:, used], dots, -np.inf).max(axis=1).clip(-1, 1)
        features[has, 0] = scores
        features[has, 1] = scores * position[has, 6]  # Existing near_mask[2]: distance <=4.
        features[has, 2] = 1
    return features


def read_context_weights(path):
    if path is None:
        raise ValueError('hard_context_fixed requires private --context-weights')
    data = json.loads(path.read_text())
    if (any(data.get(k) != v for k, v in DESIGN.items()) or
            data.get('calibration_sha256') != sha(CALIBRATION)):
        raise ValueError('Unexpected context calibration or projection design')
    w = np.asarray(data['weights'], dtype=np.float64)
    if w.shape != (28,) or not np.isfinite(w).all():
        raise ValueError('Expected 28 finite context coefficients')
    return w


class ContextRanker:
    def __init__(self, unit, weights):
        self.unit, self.weights = unit, weights

    def scores(self, before, fresh, old, position):
        pos = np.asarray(position)
        extra = context(self.unit, old, [before[i] for i in old],
                        [(j, before[j]) for j in sorted(fresh)], pos)
        return np.column_stack((pos, extra)) @ self.weights

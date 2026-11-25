"""Fixed native input-embedding context vs position, CPU only."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np
from safetensors import safe_open

from bdlm_serving.evaluation.frozen import ROOT, sha
from tools.edit_opportunity import load
from tools.lexical_hazard import BUDGETS, TOP, run_ablation, write_rows


KEY = 'model.word_embeddings.weight'
NAME = 'position_embedding_context'


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


def compare(rows):
    by = {(r['index'], r['metric'], r['model']): r for r in rows}
    indices = sorted({r['index'] for r in rows})
    budgets = [f'Q{q}/{cap}' for q, cap in BUDGETS]
    result = []
    for m in budgets + [f'top-{k}' for k in TOP]:
        pairs = [(by[i, m, 'position'], by[i, m, NAME]) for i in indices]
        deltas = [b['hits'] - a['hits'] for a, b in pairs]
        result.append(dict(metric=m, seats=sum(a['seats'] for a, _ in pairs),
                           targets=sum(a['targets'] for a, _ in pairs),
                           position_hits=sum(a['hits'] for a, _ in pairs),
                           context_hits=sum(b['hits'] for _, b in pairs), delta=sum(deltas),
                           requests_up=sum(d > 0 for d in deltas), requests_down=sum(d < 0 for d in deltas)))
    physical = [r['delta'] for r in result if r['metric'] in budgets]
    return result, min(physical) >= 0 and sum(d > 0 for d in physical) >= 2


def benchmark(cal, table):
    states = []
    for t in json.loads((cal / 'weights.json').read_text())['samples']:
        calls = []
        def capture(*inputs):
            calls.append(inputs)
            return np.zeros((len(inputs[0]), 3))
        path = load(cal / f"{t['index']:02d}.trace.jsonl.gz", t, context_features=capture)
        states.extend(c for c, step in zip(calls, (s for s in path if s[3]), strict=True)
                      if step[0] == 'selected')
    p = Projector(table)
    def sweep():
        start = perf_counter_ns()
        for inputs in states:
            context(p.unit, *inputs)
        return (perf_counter_ns() - start) / 1e6
    cold = sweep()
    warm = median(sweep() for _ in range(21))
    return dict(dataset='calibration24', selected_states_with_old=len(states),
                old_candidates=sum(len(s[0]) for s in states), cached_rows=len(p.cache),
                miss_read_project_normalize_ms=p.miss_ns / 1e6, cold_sweep_ms=cold,
                warm_sweep_median_ms=warm, warm_us_per_state=1000 * warm / len(states))


def main(cal, native, checkpoint, out):
    out.mkdir(parents=True, exist_ok=True)
    with input_table(checkpoint) as table:
        p = Projector(table)
        # The shared runner also writes coefficients. Keep only aggregate counts.
        with TemporaryDirectory(dir=out) as scratch:
            rows, _ = run_ablation(cal, native, Path(scratch), NAME, {},
                                   context_features=lambda *args: context(p.unit, *args))
        result, passed = compare(rows)
        write_rows(out / 'comparison.csv', result)
        timing = benchmark(cal, table)
        (out / 'timing.json').write_text(json.dumps(timing, indent=2) + '\n')
    print('Frozen practical rule:', 'PASS' if passed else 'FAIL')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('cal', 'native', 'checkpoint', 'out'):
        parser.add_argument('--' + key, required=True, type=Path)
    main(**vars(parser.parse_args()))

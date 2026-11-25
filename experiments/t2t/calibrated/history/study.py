"""Five predeclared past-event features; fit and score with the shared ablation."""
import argparse
from collections import deque
from pathlib import Path

import numpy as np

from tools.lexical_hazard import BUDGETS, TOP, run_ablation, write_rows


NAME = "position_history"
FEATURES = ("own_strict_clip2", "own_ever_strict", "left4_strict_k4", "right4_strict_k4",
            "near4_nonself_writes_k4")
METRICS = tuple(f"top-{k}" for k in TOP) + tuple(f"Q{q}/{n}" for q, n in BUDGETS)


def bits(positions):
    return sum(1 << j for j in positions)


SIDES = tuple((bits(range(max(0, i-4), i)), bits(range(i+1, min(32, i+5)))) for i in range(32))


class PastEdits:
    def __init__(self):
        self.own = bytearray(32)
        self.rounds = deque(maxlen=4)

    def observe(self, strict, writes):
        for j in strict:
            self.own[j] = min(2, self.own[j] + 1)
        self.rounds.append((bits(strict), bits(writes)))

    def features(self, i):
        left, right = SIDES[i]
        near = left | right
        l = r = w = 0
        for strict, writes in self.rounds:
            l += (strict & left).bit_count()
            r += (strict & right).bit_count()
            w += (writes & near).bit_count()
        return self.own[i], int(bool(self.own[i])), l, r, w


def paired(rows, tasks):
    hits = {(r['index'], r['model'], r['metric']): r['hits'] for r in rows}
    def matrix(model):
        return np.array([[hits[t['index'], model, m] for m in METRICS] for t in tasks])

    base, history = matrix('position'), matrix(NAME)
    delta = history - base
    rng = np.random.Generator(np.random.PCG64(20261126))
    draws = np.zeros((10000, len(METRICS)), dtype=np.int64)
    for ds in sorted({t['dataset'] for t in tasks}):
        indices = [i for i, t in enumerate(tasks) if t['dataset'] == ds]
        sampled = rng.choice(indices, (len(draws), len(indices)), replace=True)
        draws += delta[sampled].sum(axis=1)
    low, high = np.quantile(draws, [.025, .975], axis=0, method='nearest')
    return [dict(metric=m, position_hits=base[:, j].sum(), history_hits=history[:, j].sum(),
                 delta=delta[:, j].sum(), lo95=low[j], hi95=high[j],
                 requests_up=(delta[:, j] > 0).sum(), requests_down=(delta[:, j] < 0).sum())
            for j, m in enumerate(METRICS)]


def passes(rows):
    delta = {r['metric']: r['delta'] for r in rows}
    physical = [delta[f'Q{q}/{n}'] for q, n in BUDGETS]
    return min(physical) >= 0 and sum(d > 0 for d in physical) >= 2


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for opt in ('cal', 'native', 'out'):
        p.add_argument('--' + opt, required=True, type=Path)
    args = p.parse_args()
    rows, tasks = run_ablation(args.cal, args.native, args.out, NAME,
                               dict(appended_features=FEATURES), history_factory=PastEdits)
    differences = paired(rows, tasks)
    write_rows(args.out / 'paired.csv', differences)
    print('Frozen practical criterion:', 'PASS' if passes(differences) else 'FAIL; no retuning')

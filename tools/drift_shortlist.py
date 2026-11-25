"""CPU-only held-request shortlists for native Gemma L15 high-drift old rows."""
import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, comb
from pathlib import Path

import numpy as np

from bdlm_serving.evaluation.frozen import MANIFEST
from bdlm_serving.methods.t2t.frontier import _frontier_scores, select_hard
from bdlm_serving.methods.t2t.gemma_sparse import _advance, _hard_touch
from bdlm_serving.methods.t2t.planner import History
from tools.native_kv_drift import _load_diagnostic, _read_trace, write_csv


BUDGETS = (16, 32, 64, 128, 256)
FEATURES = ('bias', 'position', 'position_sq', 'left_edge', 'right_edge',
            'own_write_recency', 'left4', 'right4', 'left16', 'right16')
POS_DIM = 5
IDENTITY = ('index', 'dataset', 'task_id', 'seed')
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/t2t/placement/native-kv/shortlist'


def check(condition, message):
    if not condition:
        raise ValueError(message)


def features(pool, writes, clock):
    """All inputs predate the prediction. Unknown clocks never create a write."""
    i = np.asarray(pool, dtype=int)
    w = np.asarray(writes, dtype=int)
    check(np.all((w >= -1) & (w < clock)), 'write from current/future step')
    p = i / max(1, len(w) - 1)
    cols = [np.ones_like(p), p, p*p, 1/(1+i), 1/(len(w)-i),
            np.where(w[i] >= 0, 1/(clock-w[i]), 0)]
    prefix = np.r_[0, np.cumsum((w >= 0) & (w == clock-1))]
    for radius in (4, 16):
        cols += [(prefix[i]-prefix[np.maximum(0, i-radius)])/radius,
                 (prefix[np.minimum(len(w), i+radius+1)]-prefix[i+1])/radius]
    return np.column_stack(cols)


def order(scores):
    # Pool indices are in ascending position order, including exact ties.
    return np.argsort(-np.asarray(scores), kind='stable')


@dataclass
class Step:
    x: np.ndarray
    y: np.ndarray
    frontier: np.ndarray  # permutation of indices into the sorted pool
    layout: dict


def parse(diag, meta, rows, task):
    check(all(meta[k] == diag[k] == task[k] for k in IDENTITY), 'request identity')
    check(all(diag[k] == v for k, v in dict(model='gemma', policy='native256',
              canvas=256, layer=15, top_fraction=.1, top_k=None, joint='max_dk_dv',
              drift_floor=1e-6, shared_kv=False, layer_type='sliding_attention').items()),
          'wrong diagnostic configuration')
    labels = {(s['block'], s['clock']): s for s in diag['steps']}
    check(len(labels) == len(diag['steps']), 'duplicate diagnostic frame')
    width = diag['canvas']
    all_rows = range(width)  # this source is verified native256: every row was executed
    prior_block = None
    steps = []
    for frame in rows:
        block, clock = frame['block'], frame['clock']
        if block != prior_block:
            check(prior_block is None or block == prior_block + width, 'canvas jump')
            h = History([False]*width, [None]*width, [0]*width, set(), set())
            writes, queries, step_id = [-1]*width, [-1]*width, 1
        else:
            check(clock == prior_clock-1 and frame['before'] == prior_after, 'trace continuity')
            step_id += 1
        pool = [i for i in all_rows if h.resolved[i] and i not in h.fresh]
        if pool:
            x = features(pool, writes, step_id)
            fscore = _frontier_scores(pool, writes, queries, step_id)
            rank = order([fscore[i] for i in pool])
            layout = {}
            for q in BUDGETS:
                args = (all_rows, h.resolved, h.fresh, writes, queries, step_id, q)
                base, _ = select_hard(*args, 0, 'hard_noedit', 0)
                base_old = set(base).intersection(pool)
                n0 = len(base_old)
                check(base_old == set(pool[:n0]), 'noedit neutral prefix')
                for mode, policy in (('slack', 'hard_slack'), ('fixed', 'hard_frontier')):
                    if q == width and mode == 'fixed':
                        continue
                    chosen, res = ((base, []) if q == width else
                                   select_hard(*args, q//8, policy, 0))
                    r = len(res)
                    nold = len(set(chosen).intersection(pool))
                    check(len(chosen) == len(base) and set(h.fresh) <= set(chosen)
                          and set(res) == {pool[j] for j in rank[:r]}, 'unmatched allocation')
                    layout[q, mode] = (r, nold-r, n0, len(base))
            # Attach labels only after the prestate, scores and seat counts are frozen.
            s = labels.pop((block, clock), None)
            check(s is not None and (s['step_id'], s['prev_step_id'], s['prev_clock'],
                  s['pool_size'], s['nominal_k']) == (step_id, step_id-1, clock+1,
                  len(pool), ceil(.1*len(pool))), 'label frame or pool mismatch')
            target = {v[0] for v in s['top_positions']}
            check(len(target) == s['top_count'] == len(s['top_positions']) and
                  target <= set(pool), 'invalid archived drift targets')
            check([pool[j] for j in rank[:s['nominal_k']]] == [v[0] for v in s['selected_positions']],
                  'original frontier parity')
            steps.append(Step(x, np.array([i in target for i in pool]), rank, layout))
        before, after, acc = frame['before'], frame['after'], frame['accepted']
        check(len(before) == len(after) == len(acc) == width, 'invalid native width')
        old = h
        h, _ = _advance(old, before, after, acc, all_rows, [None]*width)
        _hard_touch(writes, queries, all_rows, acc, before, after, old.resolved, step_id)
        prior_block, prior_clock, prior_after = block, clock, after
    check(not labels, 'unmatched diagnostic labels')
    return steps


def folds(tasks):
    groups = defaultdict(list)
    for t in tasks:
        groups[t['dataset']].append(t['index'])
    check(sorted(map(len, groups.values())) == [10, 10, 10] and
          len({i for g in groups.values() for i in g}) == 30, 'expected 3 x 10 distinct requests')
    return [sorted(i for g in groups.values() for i in sorted(g)[f::5]) for f in range(5)]


def fit(train, dim):
    """Small ridge logistic regression. Fixed L2=2, no holdout tuning or scaling."""
    x = np.concatenate([s.x[:, :dim] for s in train])
    y = np.concatenate([s.y for s in train])
    w = np.zeros(dim)
    penalty = np.diag([0.] + [2.]*(dim-1))
    for _ in range(40):
        p = 1/(1+np.exp(-np.clip(x@w, -50, 50)))
        g = x.T@(p-y) + penalty@w
        h = x.T@((p*(1-p))[:, None]*x) + penalty
        delta = np.linalg.solve(h, g)
        w -= delta
        if np.max(np.abs(delta)) < 1e-8:
            return w
    raise ValueError('ridge fit did not converge')


def train_fold(data, test):
    ids = sorted(data.keys()-set(test))
    rows = [s for i in ids for s in data[i]]
    return (ids, sum(len(s.y) for s in rows), sum(int(s.y.sum()) for s in rows),
            fit(rows, len(FEATURES)), fit(rows, POS_DIM))


@lru_cache(maxsize=4096)
def uniform(n, r, b):
    """P(selected) for uniform reserve r then first b non-reserved positions."""
    if r == n:
        return np.ones(n)
    p = np.full(n, r/n)
    p[:b] = 1.
    # Given i is not reserved, X reserved before i is hypergeometric.
    # Neutral iff i-X < b. Outside this narrow boundary P is already known.
    denom = comb(n-1, r)
    for i in range(b, min(n, b+r)):
        mass = sum(comb(i, j)*comb(n-1-i, r-j) for j in
                   range(max(i-b+1, r-(n-1-i)), min(i, r)+1))
        p[i] += (1-r/n)*mass/denom
    return p


def count(s, ranking, r, b, baseline, physical=None):
    """Same reserve, neutral and total Q counts; baseline is a physical prefix."""
    n = len(s.y)
    if ranking is None:
        chosen = uniform(n, r, b)
        reserve_hits = r*int(s.y.sum())/n
        reserve_common = r*int(s.y[:baseline].sum())/n
    else:
        chosen = np.zeros(n, dtype=bool)
        chosen[ranking[:r]] = True
        reserve_hits = int(s.y[chosen].sum())
        reserve_common = int(s.y[:baseline][chosen[:baseline]].sum())
        chosen[np.flatnonzero(~chosen)[:b]] = True
    hits = float(chosen[s.y].sum()) if b else reserve_hits
    c = Counter(steps=1, pool=n, targets=int(s.y.sum()), old_seats=r+b, hits=hits,
                reserved=r, reserved_hits=reserve_hits, neutral=b, neutral_hits=hits-reserve_hits)
    if physical is not None:
        common = float(chosen[:baseline][s.y[:baseline]].sum())
        base_hits = int(s.y[:baseline].sum())
        c.update(physical=physical, base_old=baseline, base_hits=base_hits, common_hits=common,
                 reserved_common_hits=reserve_common, reserved_new_hits=reserve_hits-reserve_common,
                 new_old=float(chosen[baseline:].sum()), new_hits=hits-common,
                 lost_hits=base_hits-common, net_hits=hits-base_hits, mask_lost=r+b-baseline)
    return c


def evaluate(steps, w, wp):
    acc = defaultdict(Counter)
    for s in steps:
        n = len(s.y)
        for name, rank in [('pos_write', order(s.x@w)), ('position', order(s.x[:, :POS_DIM]@wp)),
                           ('frontier', s.frontier), ('leftmost', np.arange(n)), ('uniform', None)]:
            for label, k in [('top1', 1), ('top2', 2), ('top4', 4), ('native10', ceil(.1*n))]:
                acc['list', label, name].update(count(s, rank, min(n, k), 0, 0))
            for (q, mode), args in s.layout.items():
                acc['dense' if q == 256 else mode, q, name].update(count(s, rank, *args))
    return acc


def main(raw, out):
    tasks = json.loads(MANIFEST.read_text())['samples']
    by_id = {t['index']: t for t in tasks}
    check(json.loads((raw/'verified.json').read_text()) == dict(model='gemma', matched_requests=30),
          'missing verified native parity')
    diags = _load_diagnostic(raw, 15)
    check([d['index'] for d in diags] == sorted(by_id) == list(range(30)), 'incomplete cohort')
    data = {d['index']: parse(d, *_read_trace(raw/f"{d['index']:02d}.trace.jsonl.gz"),
                              by_id[d['index']]) for d in diags}
    requests, fits, sums = [], [], defaultdict(Counter)
    for f, test in enumerate(folds(tasks)):
        train, train_n, train_y, w, wp = train_fold(data, test)
        fits.append(dict(fold=f, train=train, test=test, training_rows=train_n,
                         training_targets=train_y, pos_write=w.tolist(),
                         position=wp.tolist()))
        for i in test:
            t = by_id[i]
            for key, c in evaluate(data[i], w, wp).items():
                mode, seats, name = key
                requests.append(dict(fold=f, **{k:t[k] for k in IDENTITY}, mode=mode,
                                     seats_or_q=seats, method=name, **c))
                for g in ('all', f'fold{f}', t['dataset']):
                    sums[g, *key].update(c)
                    sums[g, *key]['requests'] += 1
    ref = sums['all', 'list', 'native10', 'frontier']
    check(tuple(ref[k] for k in ('steps','pool','targets','hits')) == (325,40350,4182,511),
          'original aggregate parity failed')
    summary = [dict(group=g, mode=m, seats_or_q=k, method=name, **c)
               for (g,m,k,name), c in sums.items()]
    out.mkdir(parents=True, exist_ok=True)
    for prefix, rows in [('requests', requests), ('summary', summary)]:
        for suffix, is_list in [('shortlist', True), ('budget', False)]:
            selected = [a for a in rows if (a['mode']=='list') == is_list]
            for a in selected:
                a['precision'] = a['hits']/a['old_seats'] if a['old_seats'] else None
                a['recall'] = a['hits']/a['targets'] if a['targets'] else None
            write_csv(out/f'{prefix}-{suffix}.csv', selected)
    (out/'folds.json').write_text(json.dumps(dict(features=FEATURES, ridge=2., folds=fits),
                                             indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True, type=Path)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    main(args.raw, args.out)

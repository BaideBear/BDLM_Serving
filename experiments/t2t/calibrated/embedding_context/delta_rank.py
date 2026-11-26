"""CPU-only Q8 gate5pct replay: two residual slopes on last-transition ID changes."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from . import native_gates as g


FROZEN_SHA = 'b0b3eebf668e039f39e84b4fd5145400de94998e21364225c894ea73c27ab516'
FIELDS = ('swaps', 'physical_q_on_swaps', 'one_seat_queries', 'list_rows', 'list_targets',
          'top8_oracle', 'original_hits', 'reranked_hits', 'gained', 'lost', 'changed_pick',
          'lost_mask_admissions', 'lost_old_hits')


def delta_features(unit, pool, visible, previous_changes):
    x = np.zeros((len(pool), 2))
    for n, i in enumerate(pool):
        v = np.zeros(16)
        mass = 0.
        for j, before, after in previous_changes:
            d = abs(i-j)
            if 0 < d <= 8:
                w = 1/d
                v += w * np.subtract(unit(after), unit(before), dtype=np.float64)
                mass += w
        if mass:
            v /= mass
            x[n] = np.linalg.norm(v), unit(visible[i]) @ v
    return x


def fit(batches):
    """Penalized Bernoulli likelihood on candidates, with the log odds offset fixed."""
    x, offset, y = map(np.concatenate, zip(*batches, strict=True))

    def objective(b):
        z = offset + x @ b
        return (np.sum(np.logaddexp(0., z) - y*z) + b @ b,
                x.T @ (expit(z)-y) + 2*b)

    result = minimize(objective, np.zeros(2), jac=True, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-12})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError('Residual fit did not converge')
    return result.x


class Observer:
    def __init__(self, unit, slopes=None):
        self.unit, self.slopes = unit, slopes
        self.batches = [] if slopes is None else None
        self.totals = {kind: Counter() for kind in ('crowded', 'slack')}
        self.pending = None

    def select(self, space, scores, picked, visible, previous_changes):
        if self.pending is not None:
            raise ValueError('Missing outcome for previous exchange')
        if picked == space.base:
            return
        offsets = dict(zip(space.old, scores, strict=True))
        pool = sorted(set(space.old) - space.base, key=lambda i: (-offsets[i], i))[:8]
        if not pool or picked-space.base != {pool[0]} or len(space.base-picked) != 1:
            raise ValueError('Not the frozen single outside exchange')
        x = delta_features(self.unit, pool, visible, previous_changes)
        offset = np.array([offsets[i] for i in pool])
        scores = offset if self.slopes is None else offset + x @ self.slopes
        chosen = pool[min(range(len(pool)), key=lambda j: (-scores[j], pool[j]))]
        self.pending = space, picked, pool, chosen, x, offset

    def outcome(self, targets, admits):
        if self.pending is None:
            return
        space, picked, pool, chosen, x, offset = self.pending
        self.pending = None
        y = np.array([i in targets for i in pool])
        if self.batches is not None:
            self.batches.append((x, offset, y))
        # Replace ONLY the incoming row, without changing the gate or the victim.
        reranked = (picked - {pool[0]}) | {chosen}
        c = g.counts(space, reranked, targets, admits)
        if space.base-reranked != space.base-picked or len(reranked) != len(picked) or c['physical_swaps'] != 1:
            raise ValueError('Reranker changed physical layout')
        old, new = int(y[0]), c['ranked_old_hits_added']
        self.totals['crowded' if c['displaced_masks'] else 'slack'].update(
            swaps=1, physical_q_on_swaps=c['physical_q'], one_seat_queries=1,
            list_rows=len(pool), list_targets=int(y.sum()), top8_oracle=int(y.any()),
            original_hits=old, reranked_hits=new, gained=int(new > old), lost=int(new < old),
            changed_pick=int(chosen != pool[0]), lost_mask_admissions=c['lost_mask_admissions'],
            lost_old_hits=c['old_hits_lost'])


def scan(folder, tasks, ranker, mask, size, slopes=None, archive=None):
    batches, totals = [], {k: Counter() for k in ('crowded', 'slack')}
    for task in tasks:
        obs = Observer(ranker.unit, slopes)
        with gzip.open(folder / f"{task['index']:02d}.trace.jsonl.gz", 'rt') as f:
            gate = g.replay(map(json.loads, f), task, ranker, mask, size, observer=obs)['gate5pct']
        both = sum(obs.totals.values(), Counter())
        if (obs.pending is not None or any(both[k] != gate[v] for k, v in (
                ('swaps', 'physical_swaps'), ('original_hits', 'ranked_old_hits_added'),
                ('lost_old_hits', 'old_hits_lost'), ('lost_mask_admissions', 'lost_mask_admissions')))):
            raise ValueError('Observer and frozen physical exchange disagree')
        if archive is not None:
            ref = archive[task['index'], task['dataset']]
            if any(gate[k] != int(ref[k]) for k in ('states', 'physical_q', 'old_targets', 'physical_swaps',
                    'displaced_masks', 'displaced_old', 'ranked_old_hits_added', 'lost_mask_admissions', 'old_hits_lost')):
                raise ValueError('Archived native gate mismatch')
        for kind, c in obs.totals.items():
            totals[kind].update(c)
        if obs.batches is not None:
            batches.extend(obs.batches)
    return batches, totals


def manifests(cal):
    if g.sha(cal/'weights.json') != g.sha(g.CALIBRATION) or g.sha(g.MANIFEST) != FROZEN_SHA:
        raise ValueError('Unexpected calibration24 or frozen30 manifest')
    train, test = (json.loads(p.read_text())['samples'] for p in (g.CALIBRATION, g.MANIFEST))
    g.check_split(train, test)
    return train, test


def run(cal, native, source_run, weights, checkpoint, out):
    if out.resolve().is_relative_to(g.ROOT):
        raise ValueError('Use a private output directory outside the checkout')
    train, test = manifests(cal)
    fixed = g.qualify(native, source_run, weights)
    mask = g.mask_token(checkpoint)
    rows = [r for r in g.read_csv(g.STUDY/'gated/native_q8e1_requests.csv') if r['mode'] == 'gate5pct']
    refs = {(int(r['request_index']), r['dataset']): r for r in rows}
    if len(rows) != 30 or set(refs) != {(t['index'], t['dataset']) for t in test}:
        raise ValueError('Incomplete native gate archive')
    with g.input_table(checkpoint) as table:
        p = g.Projector(table)
        ranker = g.ContextRanker(p.unit, fixed)
        batches, _ = scan(cal, train, ranker, mask, p.size)
        slopes = fit(batches)
        print('Calibration24: swaps / candidates / positives:', len(batches), sum(len(y) for _, _, y in batches),
              sum(int(y.sum()) for _, _, y in batches), 'slopes:', slopes)
        _, totals = scan(native, test, ranker, mask, p.size, slopes, refs)
    if (totals['crowded']['original_hits'], totals['slack']['original_hits']) != (16, 8):
        raise ValueError('Original gate hits not reproduced')
    passed = totals['crowded']['reranked_hits'] >= 19 and totals['slack']['reranked_hits'] >= 8
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    g.write_rows(out/'summary.csv', [dict(requests=len(test), victim=kind, **{k:c[k] for k in FIELDS})
                                     for kind, c in totals.items()])
    for kind, c in totals.items():
        print(kind, 'original -> reranked:', c['original_hits'], '->', c['reranked_hits'])
    print('Frozen rollout rule:', 'PASS' if passed else 'STOP')
    return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key, default in dict(cal=g.RAW/'calibration24-v1-llada21', native=g.RAW/'llada21',
            source_run=g.RAW/'context-fixed-q4q8-v1', weights=g.RAW/'context-fixed-private/calibration24.json',
            checkpoint=Path('/data0/ysy/models/LLaDA2.1-mini'), out=g.RAW/'context-delta-rank-v1').items():
        parser.add_argument('--'+key.replace('_', '-'), type=Path, default=default)
    run(**vars(parser.parse_args()))

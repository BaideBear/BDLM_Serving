import csv
import unittest
from collections import Counter
from copy import deepcopy
from itertools import combinations

import numpy as np

from tools.drift_shortlist import (FEATURES, OUT, Step, count, evaluate, features,
                                   folds, order, parse, train_fold, uniform)


def example():
    meta = dict(index=0, dataset='a', task_id='ex', seed=42)
    step = dict(block=48, clock=46, step_id=3, prev_step_id=2, prev_clock=47,
                pool_size=3, nominal_k=1, top_count=2, top_positions=[[0], [2]],
                selected_positions=[[4, .5]])
    diag = dict(meta, model='gemma', policy='native256', canvas=256, layer=15,
                top_fraction=.1, top_k=None, joint='max_dk_dv', drift_floor=1e-6,
                shared_kv=False, layer_type='sliding_attention', steps=[step])
    rows = []
    before = [0]*256
    for clock, ids in [(48, [0, 2, 4]), (47, [0, 2, 4, 5]), (46, [0, 2])]:
        accepted = [i in ids for i in range(256)]
        after = [int(a) for a in accepted]
        rows.append(dict(block=48, clock=clock, before=before, after=after, accepted=accepted))
        before = after
    return diag, meta, rows


class TestShortlist(unittest.TestCase):
    def test_prestate_only_and_canvas_reset(self):
        diag, meta, rows = example()
        s, = parse(diag, meta, rows, meta)
        modified = deepcopy(rows)
        modified[-1].update(after=[17]*256, accepted=[True]*256)
        other, = parse(diag, meta, modified, meta)
        np.testing.assert_array_equal(s.x, other.x)
        np.testing.assert_array_equal(s.frontier, other.frontier)
        self.assertEqual(s.layout, other.layout)
        # Token identity cannot be a predictor; preserve recorded write events.
        for frame in modified:
            for key in ('before', 'after'):
                frame[key] = [v+100 for v in frame[key]]
        np.testing.assert_array_equal(s.x, parse(diag, meta, modified, meta)[0].x)
        diag['steps'].append(dict(diag['steps'][0], block=304))
        s1, s2 = parse(diag, meta, rows+[dict(f, block=304) for f in rows], meta)
        np.testing.assert_array_equal(s1.x, s2.x)
        self.assertEqual(s1.layout, s2.layout)

    def test_known_history_only(self):
        feat = features([2], [-1, 4, 2, -1, -1], 5)[0]
        self.assertEqual(feat[5], 1/3)
        self.assertEqual(tuple(feat[6:]), (.25, 0., 1/16, 0.))
        self.assertEqual(features([0], [-1]*5, 1)[0, 5:].sum(), 0.)
        with self.assertRaisesRegex(ValueError, 'current/future'):
            features([0], [5], 5)

    def test_alignment_and_invalid_target(self):
        for key, val in [('prev_step_id', 3), ('pool_size', 4), ('top_positions', [[0], [5]])]:
            diag, meta, rows = example()
            diag['steps'][0][key] = val
            with self.assertRaises(ValueError):
                parse(diag, meta, rows, meta)

    def test_ties_do_not_expand_nominal_k(self):
        diag, meta, rows = example()
        s, = parse(diag, meta, rows, meta)
        tied = order([1., 1., 1.])
        self.assertEqual(tied.tolist(), [0,1,2])
        metrics = evaluate([s], np.zeros(len(FEATURES)), np.zeros(5))
        c = metrics['list','native10','pos_write']
        self.assertEqual((c['targets'], c['old_seats'], c['hits']), (2,1,1))
        self.assertEqual(metrics['list','top4','pos_write']['old_seats'], 3)

    def test_uniform_reserve_then_neutral_exact(self):
        for n in range(1, 8):
            for r in range(n+1):
                for b in range(n-r+1):
                    picks = []
                    for combi in combinations(range(n), r):
                        chosen = set(combi)
                        chosen.update([i for i in range(n) if i not in chosen][:b])
                        picks.append([i in chosen for i in range(n)])
                    np.testing.assert_allclose(uniform(n,r,b), np.mean(picks, axis=0))

    def test_extra_vs_existing_backfill(self):
        # Base old = {0,1,2}; reserved = {1,4}; neutral = {0}. The shared
        # hit at 1 is not an incremental benefit; target 2 is lost.
        s = Step(None, np.array([0,1,1,0,1], dtype=bool), None, {})
        c = count(s, np.array([1,4,3,2,0]), 2, 1, 3, 16)
        self.assertEqual([c[k] for k in ('hits','reserved_hits','reserved_common_hits',
                         'reserved_new_hits','new_hits','lost_hits','net_hits','mask_lost')],
                         [2,2,1,1,1,1,0,0])

    def test_whole_request_split_and_fit_isolation(self):
        tasks = [dict(index=i, dataset='abc'[i//10]) for i in range(30)]
        parts = folds(list(reversed(tasks)))
        self.assertEqual(sorted(i for group in parts for i in group), list(range(30)))
        for hold in parts:
            self.assertEqual(Counter(tasks[i]['dataset'] for i in hold), dict(a=2,b=2,c=2))
        hold = parts[0]
        x = features(range(4), [1,2,1,1], 3)
        example_step = Step(x, np.array([1,0,0,1], dtype=bool), None, {})
        # A held-out object cannot even be iterated during fit.
        data = {i: (object() if i in hold else [example_step]) for i in range(30)}
        ids,n,y,w,wp = train_fold(data, hold)
        self.assertEqual((len(ids),len(set(ids)&set(hold)),n,y), (24,0,96,48))
        self.assertTrue(np.isfinite(w).all() and np.isfinite(wp).all())

    def test_archived_budget_parity(self):
        with (OUT.parent/'kvdrift-budget-coverage.csv').open() as f:
            archive = {(r['budget'], r['policy']): r for r in csv.DictReader(f)}
        with (OUT/'summary-budget.csv').open() as f:
            rows = [r for r in csv.DictReader(f) if r['group']=='all' and r['method']=='frontier']
        self.assertEqual(len(rows), 9)
        for r in rows:
            q = r['seats_or_q']
            policy = {'slack':'hard_slack', 'fixed':'hard_frontier', 'dense':'hard_noedit'}[r['mode']]
            ref = archive[q, policy]
            for new, old in [('hits','total_hits'), ('reserved_hits','reserved_hits'),
                             ('neutral_hits','backfill_hits'), ('physical','seats_total'),
                             ('mask_lost','delta_mask_lost_vs_noedit')]:
                self.assertEqual(float(r[new]), float(ref[old]), (q, policy, new))
            self.assertEqual(float(r['base_hits']), float(archive[q,'hard_noedit']['total_hits']))


if __name__ == '__main__':
    unittest.main()

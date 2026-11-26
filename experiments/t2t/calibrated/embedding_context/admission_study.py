"""Learn marginal MASK-admission risk on calibration24; keep the original ranked entrant."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from bdlm_serving.methods.t2t import admission_gate as a
from . import native_gates as g
from .delta_rank import manifests


class Offers:
    def __init__(self):
        self.pending = None
        self.rows = []

    def select(self, space, scores, picked, visible, previous_changes):
        if self.pending is not None:
            raise ValueError('Missing outcome')
        dropped = space.base - picked
        if not dropped.intersection(space.masks):
            return
        entrant, = picked-space.base
        victim, = dropped
        m = dict(reason='swap_mask', victim=victim, masks=space.masks, fresh=space.fresh)
        self.pending = a.vector(m), entrant, victim

    def outcome(self, targets, admits):
        if self.pending:
            x, entrant, victim = self.pending
            self.rows.append((x, int(entrant in targets), int(victim in admits)))
            self.pending = None


def collect(tasks, folder, ranker, mask, size):
    by_request = {}
    for task in tasks:
        obs = Offers()
        with gzip.open(folder/f"{task['index']:02d}.trace.jsonl.gz", 'rt') as f:
            t = g.replay(map(json.loads, f), task, ranker, mask, size, observer=obs)['gate5pct']
        if (obs.pending is not None or t['displaced_masks'] != len(obs.rows)
                or t['lost_mask_admissions'] != sum(r[2] for r in obs.rows)):
            raise ValueError('Marginal MASK ledger disagrees')
        by_request[task['index']] = obs.rows
    return by_request


def train(by_request):
    rows = sum(by_request.values(), [])
    x = np.asarray([r[0] for r in rows])
    y = np.asarray([r[2] for r in rows])
    if len(by_request) != 24 or set(y) != {0, 1}:
        raise ValueError('Unexpected calibration cohort or outcomes')
    penalty = np.r_[0., np.full(x.shape[1]-1, 2.)]
    def fn(w):
        z = x @ w
        return (np.sum(np.logaddexp(0., z)-y*z)+.5*(penalty*w)@w,
                x.T @ (expit(z)-y)+penalty*w)
    fitted = minimize(fn, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-12})
    if not fitted.success:
        raise ValueError(fitted.message)
    fit = dict(schema=a.SCHEMA, calibration_sha256=a.CAL_SHA, ranker_sha256=a.RANK_SHA,
               features=list(a.FEATURES), weights=fitted.x.tolist(), fit_requests=24,
               training_offers=len(y), training_admissions=int(sum(y)))
    scores = [a.risk(row[0], fit) for row in rows]
    fit['cutoff'] = float(np.median(scores))
    fit['keep_probability'] = sum(p <= fit['cutoff'] for p in scores)/len(scores)
    return fit


def audit(by, fit, tasks, out):
    requests, totals = [], {}
    for task in tasks:
        i = task['index']
        group = by[i]
        selected = [a.risk(x, fit) <= fit['cutoff'] for x,_,_ in group]
        chance = sum(selected)/len(group) if group else 0.
        for name in ('always', 'low_risk', 'matched_random', 'admission_oracle', 'edit_and_safe_oracle'):
            c = Counter(dict(offers=len(group), approved=0, strict_hits=0, lost_admissions=0,
                             hit_no_loss=0, hit_and_loss=0, miss_no_loss=0, miss_and_loss=0))
            for (_,hit,loss), low in zip(group,selected,strict=True):
                w = {'always':1, 'low_risk':int(low), 'matched_random':chance,
                     'admission_oracle':1-loss, 'edit_and_safe_oracle':hit*(1-loss)}[name]
                c['approved'] += w
                c['strict_hits'] += w*hit
                c['lost_admissions'] += w*loss
                c[('hit' if hit else 'miss')+('_and_loss' if loss else '_no_loss')] += w
            requests.append(dict(index=i, dataset=task['dataset'], rule=name, **c))
            totals.setdefault(name, Counter()).update(c)
    base = totals['always']
    if (base['offers'],base['strict_hits'],base['lost_admissions']) != (120,16,45):
        raise ValueError('Did not reproduce frozen native Q8 gate')
    out.mkdir(parents=True, exist_ok=True)
    g.write_rows(out/'native-requests.csv', requests)
    g.write_rows(out/'native-summary.csv',[dict(rule=k,requests=len(by),**v) for k,v in totals.items()])


def run(args):
    cal_tasks, tasks = manifests(args.cal)
    fixed = g.qualify(args.native, args.source_run, args.weights)
    mask = g.mask_token(args.checkpoint)
    with g.input_table(args.checkpoint) as table:
        p = g.Projector(table)
        ranker = g.ContextRanker(p.unit, fixed)
        if args.action == 'fit':
            fit = train(collect(cal_tasks, args.cal, ranker, mask, p.size))
            args.fit.parent.mkdir(parents=True, exist_ok=True)
            with args.fit.open('x') as f:
                json.dump(fit, f, sort_keys=True, indent=2, allow_nan=False)
                f.write('\n')
            a.read_admission_fit(args.fit)
        else:
            fit = a.read_admission_fit(args.fit)
            audit(collect(tasks, args.native, ranker, mask, p.size), fit, tasks, args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('fit','audit'))
    for name, default in dict(cal=g.RAW/'calibration24-v1-llada21', native=g.RAW/'llada21',
            source_run=g.RAW/'context-fixed-q4q8-v1', weights=g.RAW/'context-fixed-private/calibration24.json',
            fit=g.RAW/'admission-v1-private/victim.json', out=g.RAW/'admission-native-v1',
            checkpoint=Path('/data0/ysy/models/LLaDA2.1-mini')).items():
        parser.add_argument('--'+name.replace('_','-'), type=Path, default=default)
    run(parser.parse_args())

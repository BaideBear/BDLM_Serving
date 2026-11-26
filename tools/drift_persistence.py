"""Bound or exactly replay lagged L15 drift on projected native noedit layouts.

Only bootstrap is projected dense by stage. Later states use the original
shortlist's noedit layout, never a simulated policy's past choices. This is
not sparse-cache drift, serving work, or an inference-time top-membership test.
"""
import argparse
from collections import Counter, deque
import csv
from fractions import Fraction
import json
from math import isfinite
from pathlib import Path

from tools import drift_shortlist as ds
from tools.native_kv_drift import top_set


RAW = ds.ROOT.parent/'t2t-native-1125'
BUCKETS = ('identified', 'no_prior_top_match', 'unranked_competitor')
SEAT = ('events', 'options', 'donor_eligible', 'prior_out_of_pool_options', 'ambiguous_hit_events',
        'zero_hits', 'uniform_hits', 'outside_uniform_hits', 'lag_lo_hits', 'lag_hi_hits')
LAYOUT = ('frames', 'bootstraps', 'physical_q_all', 'old_states', 'old_rows', 'target_rows',
          'physical_q_old_states', 'old_seats', 'zero_total_hits', 'slack_states', 'no_outside',
          'missing_history', 'zero_options', 'one_option')
EXACT = ('swaps', 'donor_collisions', 'tied_max_events', 'gained_hits', 'lost_hits',
         'both_hit_swaps', 'exact_hits')


def contenders(pool, picked, history):
    """The donor is the last neutral fill. Other selected rows stay fixed."""
    neutral = pool & picked
    if not neutral:
        return 'no_slack', None, set()
    donor = max(neutral)
    if not pool-picked:
        return 'no_outside', donor, set()
    if len(history) != 2:
        return 'missing_history', donor, set()
    options = ((pool-picked) | {donor}) & history[0] & history[1]
    return ('contest' if len(options) >= 2 else 'one_option' if options else 'zero_options'), donor, options


def possible_winners(options, prior_pool, prior_top):
    """Identify or bound the pre-label argmax, never use censorship to skip a swap."""
    known = options.intersection(prior_top)
    if not known:
        return 'no_prior_top_match', options
    best = min(known, key=lambda i: (-prior_top[i], i))
    # The archived top is global only within the PREVIOUS nonfresh old pool.
    # A physically observed prior fresh/MASK row was outside its ranking.
    missing = options - prior_pool
    return ('unranked_competitor' if missing else 'identified'), missing | {best}


def seat_outcome(donor, options, winners, prior_pool, target):
    hits = len(winners & target)
    outside = options - {donor}
    return Counter(events=1, options=len(options), donor_eligible=int(donor in options),
                   prior_out_of_pool_options=len(options-prior_pool),
                   ambiguous_hit_events=int(0 < hits < len(winners)), zero_hits=int(donor in target),
                   uniform_hits=Fraction(len(options & target), len(options)),
                   outside_uniform_hits=Fraction(len(outside & target), len(outside)),
                   lag_lo_hits=int(hits == len(winners)), lag_hi_hits=int(hits > 0))


def top_values(frame):
    values = {}
    for i, dk, dv, joint, _ in frame['top_positions']:
        ds.check(all(isfinite(v) and v >= 0 for v in (dk, dv, joint)) and joint == max(dk, dv),
                 'invalid archived joint L15 drift')
        values[i] = joint
    return values


def exact_frames(payload, diag, rows):
    """One finite full-width joint vector per non-bootstrap transition, in trace order."""
    ds.check(all(payload.get(k) == diag[k] for k in (*ds.IDENTITY, 'canvas', 'layer'))
             and payload.get('metric') == diag['joint'], 'lag capture identity or configuration')
    expected = [(b['block'], b['clock']) for a, b in zip(rows, rows[1:]) if a['block'] == b['block']]
    frames = payload.get('frames')
    ds.check(isinstance(frames, list) and [(f['block'], f['clock']) for f in frames] == expected,
             'missing, duplicate or shifted lag frame')
    result = {}
    for key, frame in zip(expected, frames, strict=True):
        joint = frame.get('joint')
        ds.check(isinstance(joint, list) and len(joint) == diag['canvas'] and
                 all(type(x) in (float, int) and isfinite(x) and x >= 0 for x in joint),
                 'invalid full-width joint vector')
        result[key] = joint
    return result


def read_exact_request(raw, recapture, diag, trace, task):
    stem = f"{task['index']:02d}"
    native = f'{stem}-native256.json'
    old = json.loads((raw/native).read_text())
    ds.check((old.get('model'), old.get('policy'), old.get('status')) == ('diffusiongemma', 'native256', 'success')
             and {'output_ids', 'output_tokens', 'generation_forwards', 'generation_query_rows',
                  'generation_physical_padded_rows', 'events'} <= old.keys(), f'{stem}: incomplete native record')
    ds.check(all(old[k] == task[k] for k in ds.IDENTITY) and
             json.loads((recapture/native).read_text()) == old, f'{stem}: native output/work mismatch')
    ds.check(ds._read_trace(recapture/f'{stem}.trace.jsonl.gz') == trace, f'{stem}: native trace mismatch')
    ds.check(json.loads((recapture/f'{stem}-kvdrift-layer15.json').read_text()) == diag,
             f'{stem}: historical L15 diagnostic mismatch')
    return exact_frames(json.loads((recapture/f'{stem}-lag-joint-layer15.json').read_text()), diag, trace[1])


def exact_choice(options, prior_joint):
    # No target, current frame, or censored top-membership available to selection.
    return min(options, key=lambda i: (-prior_joint[i], i))


def audit_request(diag, meta, rows, task, totals, strata, *, exact=None):
    # parse validates identities, frame clocks, old pools, frontier ranks and
    # neutral prefixes. It drops no-old frames; these must still advance history.
    steps = iter(ds.parse(diag, meta, rows, task))
    labels = {(r['block'], r['clock']): r for r in diag['steps']}
    width = diag['canvas']
    all_rows = range(width)
    never = [-1]*width  # zero reserve never ranks by last-write or last-query
    block = None
    for row in rows:
        first = block != row['block']
        if first:
            block = row['block']
            clock = 0
            h = ds.History([False]*width, [None]*width, [0]*width, set(), set())
            history = {q: deque(maxlen=2) for q in ds.BUDGETS}
            prior_pool, prior_top = set(), {}
            prior_joint = None
        clock += 1
        pool = {i for i in all_rows if h.resolved[i] and i not in h.fresh}
        step = next(steps) if pool else None
        target = {i for i, y in zip(sorted(pool), step.y, strict=True) if y} if pool else set()
        for q in ds.BUDGETS:
            c = totals[q]
            picked = set(all_rows if first else ds.select_hard(all_rows, h.resolved, h.fresh, never,
                                                              never, clock, q, 0, 'hard_noedit', 0)[0])
            c.update(frames=1, bootstraps=int(first), physical_q_all=len(picked))
            if step is not None:
                base_old = pool & picked
                ds.check(step.layout[q, 'slack'][2:] == (len(base_old), len(picked)), 'base layout mismatch')
                c.update(old_states=1, old_rows=len(pool), target_rows=len(target), old_seats=len(base_old),
                         physical_q_old_states=len(picked), zero_total_hits=len(picked & target),
                         slack_states=int(bool(base_old)))
                status, donor, options = contenders(pool, picked, history[q])
                if status == 'contest':
                    bucket, winners = possible_winners(options, prior_pool, prior_top)
                    counts = strata[q, bucket]
                    counts.update(seat_outcome(donor, options, winners, prior_pool, target))
                    if exact is not None:
                        winner = exact_choice(options, prior_joint)
                        ds.check(winner in winners, 'exact winner outside censored bounds')
                        swap = winner != donor  # donor is the only permitted already-selected row
                        hit, zero = winner in target, donor in target
                        counts.update(swaps=int(swap), donor_collisions=int(not swap), exact_hits=int(hit),
                                      tied_max_events=int(any(prior_joint[i] == prior_joint[winner]
                                                              for i in options - {winner})),
                                      gained_hits=int(swap and hit), lost_hits=int(swap and zero),
                                      both_hit_swaps=int(swap and hit and zero))
                elif status != 'no_slack':
                    c[status] += 1
            history[q].append(picked)
        prior_pool = pool
        prior_top = top_values(labels[block, row['clock']]) if pool else {}
        ds.check(prior_top.keys() == target, 'label alignment')
        if exact is not None and not first:
            # Promote today's vector only AFTER all five projected decisions. Empty-pool
            # transitions still supply real measurements to the next native prestate.
            prior_joint = exact[block, row['clock']]
            top, *_ = top_set(prior_joint, pool, fraction=diag['top_fraction'], floor=diag['drift_floor'])
            ds.check(set(top) == target and all(prior_joint[i] == v for i, v in prior_top.items()),
                     'dense joint does not reproduce historical L15 labels/values')
        h, _ = ds._advance(h, row['before'], row['after'], row['accepted'], all_rows, [None]*width)
    ds.check(next(steps, None) is None, 'unconsumed parsed frames')


def fields(c, keys):
    return {k: float(c[k]) if isinstance(c[k], Fraction) else c[k] for k in keys}


def run(raw, out, exact_raw=None):
    ds.check(not out.resolve().is_relative_to(ds.ROOT), 'choose a private output directory')
    tasks = json.loads(ds.MANIFEST.read_text())['samples']
    ds.check(json.loads((raw/'verified.json').read_text()) == dict(model='gemma', matched_requests=30),
             'expected verified Gemma frozen30')
    diags = ds._load_diagnostic(raw, 15)
    ds.check([d['index'] for d in diags] == [t['index'] for t in tasks] == list(range(30)), 'cohort mismatch')
    if exact_raw is not None:
        ds.check(exact_raw.resolve() != raw.resolve(), 'recapture must be separate from history')
        ds.check(out.resolve() not in {raw.resolve(), exact_raw.resolve()}, 'keep output separate from raw inputs')
        for suffix in ('-native256.json', '.trace.jsonl.gz', '-kvdrift-layer15.json', '-lag-joint-layer15.json'):
            ds.check({p.name for p in exact_raw.glob('*' + suffix)} == {f"{t['index']:02d}{suffix}" for t in tasks},
                     f'incomplete or unexpected recapture cohort for {suffix}')
    totals = {q: Counter() for q in ds.BUDGETS}
    strata = {(q,b): Counter() for q in ds.BUDGETS for b in BUCKETS}
    for d, task in zip(diags, tasks, strict=True):
        trace = ds._read_trace(raw/f"{d['index']:02d}.trace.jsonl.gz")
        exact = None if exact_raw is None else read_exact_request(raw, exact_raw, d, trace, task)
        audit_request(d, *trace, task, totals, strata, exact=exact)
    with (ds.OUT/'summary-budget.csv').open() as f:
        archive = {int(r['seats_or_q']):r for r in csv.DictReader(f) if r['group']=='all' and
                   r['method']=='frontier' and r['mode'] in ('slack','dense')}
    summary, coverage, exact_rows = [], [], []
    for q, c in totals.items():
        ds.check(all(c[k] == int(archive[q][a]) for k,a in [('old_states','steps'), ('old_rows','pool'),
                 ('target_rows','targets'), ('physical_q_old_states','physical'), ('old_seats','base_old'),
                 ('zero_total_hits','base_hits')]), 'archived noedit parity')
        contests = Counter()
        for bucket in BUCKETS:
            counts = strata[q,bucket]
            coverage.append(dict(q=q, bucket=bucket, **fields(counts, SEAT)))
            contests.update(counts)
        ds.check(c['slack_states'] == contests['events'] + sum(c[k] for k in
                 ('no_outside','missing_history','zero_options','one_option')), 'slack denominator mismatch')
        fixed = c['zero_total_hits'] - contests['zero_hits']
        summary.append(dict(q=q, requests=len(tasks), **fields(c, LAYOUT), **fields(contests, SEAT),
                            identified=strata[q,'identified']['events'], unknown=contests['events']-strata[q,'identified']['events'],
                            uniform_total_hits=float(fixed+contests['uniform_hits']),
                            lag_total_lo=fixed+contests['lag_lo_hits'], lag_total_hi=fixed+contests['lag_hi_hits']))
        if exact_raw is not None:
            hits = contests['exact_hits']
            ds.check(contests['lag_lo_hits'] <= hits <= contests['lag_hi_hits'] and
                     contests['swaps'] + contests['donor_collisions'] == contests['events'] and
                     hits - contests['zero_hits'] == contests['gained_hits'] - contests['lost_hits'],
                     'inconsistent exact seat accounting')
            exact_rows.append(dict(q=q, requests=len(tasks), frames=c['frames'], transitions=c['frames']-c['bootstraps'],
                                   physical_q_all=c['physical_q_all'], slack_states=c['slack_states'],
                                   **fields(contests, ('events', 'options', 'donor_eligible', *EXACT, 'zero_hits',
                                                       'uniform_hits', 'outside_uniform_hits')),
                                   old_seats=c['old_seats'], zero_total_hits=c['zero_total_hits'], exact_total_hits=fixed+hits,
                                   uniform_total_hits=float(fixed+contests['uniform_hits']),
                                   outside_uniform_total_hits=float(fixed+contests['outside_uniform_hits']),
                                   lag_total_lo=summary[-1]['lag_total_lo'], lag_total_hi=summary[-1]['lag_total_hi']))
    out.mkdir(parents=True, mode=0o700, exist_ok=True)
    if exact_raw is not None:
        ds.write_csv(out/'exact.csv', exact_rows)
        return
    ds.write_csv(out/'summary.csv', summary)
    ds.write_csv(out/'coverage.csv', coverage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=RAW/'kv-drift-v1-gemma')
    parser.add_argument('--exact-raw', type=Path, help='independent frozen30 native recapture; writes only exact.csv')
    parser.add_argument('--out', type=Path, default=RAW/'gemma-drift-persistence-v1')
    run(**vars(parser.parse_args()))

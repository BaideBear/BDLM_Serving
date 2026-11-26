from collections import Counter
import csv
from fractions import Fraction
import gzip
import json
from math import ceil

import pytest

from tools import drift_persistence as p


def test_two_previous_physical_queries_and_protected_slots():
    old = {0,1,3,4,5,6}
    base = {0,1,2,3}  # row 2 is not old; neutral seats 0 and 1 must stay fixed
    h = [{0,1,2,4,5,6}, {0,1,2,3,4,5}]
    assert p.contenders(old, base, h) == ('contest', 3, {4,5})
    h[0].add(3)
    assert p.contenders(old, base, h) == ('contest', 3, {3,4,5})
    for prior, status in [([], 'missing_history'), ([base], 'missing_history'),
                          ([{4}, {5}], 'zero_options'), ([{4}, {4}], 'one_option')]:
        assert p.contenders(old, base, prior)[0] == status
    assert p.contenders(old, {2}, h)[0] == 'no_slack'
    assert p.contenders(old, old | base, h)[0] == 'no_outside'


def test_censored_max_and_mixed_pool_bounds():
    options, prev = {2,5,8}, set(range(10))
    top = {3:100., 2:5., 5:5.}
    assert p.possible_winners(options, prev, top) == ('identified', {2})
    assert p.possible_winners(options, prev, {3:100.}) == ('no_prior_top_match', options)
    kind, possible = p.possible_winners(options, prev-{8}, top)
    assert (kind, possible) == ('unranked_competitor', {2,8})
    c = p.seat_outcome(5, options, possible, prev-{8}, {2,5})
    assert [c[k] for k in ('zero_hits','uniform_hits','outside_uniform_hits','lag_lo_hits','lag_hi_hits')] == [
        1,Fraction(2,3),Fraction(1,2),0,1]
    assert p.seat_outcome(5, options, possible, prev, {2,8})['lag_lo_hits'] == 1
    assert p.seat_outcome(5, options, possible, prev, {5})['lag_hi_hits'] == 0


def frames():
    # First 255 rows become fresh at bootstrap, stable at step 2; at step 3
    # five revocations shrink next round's neutral prefix. Both early no-old
    # frames still count as prior queries. Repeating this canvas tests reset.
    width = 256
    task = dict(index=0, task_id='fixture', dataset='synthetic', seed=7)
    diag = dict(task, model='gemma', policy='native256', canvas=width, layer=15, top_fraction=.1,
                top_k=None, joint='max_dk_dv', drift_floor=1e-6, shared_kv=False,
                layer_type='sliding_attention', steps=[])
    rows = []
    for block in (0,256):
        before = [0]*width
        h = p.ds.History([False]*width, [None]*width, [0]*width, set(), set())
        w, q = [-1]*width, [-1]*width
        for step in range(1,5):
            clock = 49-step
            after = [int(i < 255 and (step < 3 or i not in range(230,235))) for i in range(width)]
            acc = list(map(bool, after))
            rows.append(dict(block=block, clock=clock, before=before, after=after, accepted=acc))
            pool = [i for i in range(width) if h.resolved[i] and i not in h.fresh]
            if pool:
                scores = p.ds._frontier_scores(pool, w, q, step)
                k = ceil(.1*len(pool))
                leaders = sorted(pool, key=lambda i: (-scores[i],i))[:k]
                # Lag at step 4 selects 12 (tie with 14). Today's high row 13
                # cannot change that decision, even though choosing it would hit.
                top = [12,14] if step == 3 else [13]
                diag['steps'].append(dict(block=block, clock=clock, step_id=step, prev_step_id=step-1,
                    prev_clock=clock+1, pool_size=len(pool), nominal_k=k, top_count=len(top),
                    top_positions=[[i,.2,.1,.2,'context'] for i in top],
                    selected_positions=[[i,scores[i]] for i in leaders]))
            p.ds._hard_touch(w, q, range(width), acc, before, after, h.resolved, step)
            h, _ = p.ds._advance(h, before, after, acc, range(width), [None]*width)
            before = after
    return diag, task, rows


def replay(diag, task, rows, **kwargs):
    totals = {q:Counter() for q in p.ds.BUDGETS}
    strata = {(q,b):Counter() for q in totals for b in p.BUCKETS}
    p.audit_request(diag, task, rows, task, totals, strata, **kwargs)
    return totals, strata


def test_complete_frames_clock_reset_bootstrap_lag_and_dense():
    diag, task, rows = frames()
    totals, strata = replay(diag, task, rows)
    assert (totals[16]['bootstraps'], totals[16]['frames'], totals[16]['old_states']) == (2,8,4)
    for b, n in [('identified',6), ('no_prior_top_match',241)]:
        c = strata[16,b]
        assert c['events'] == 2 and c['options'] == 2*n
        assert c['lag_lo_hits'] == 0
        assert c['lag_hi_hits'] == (2 if b == 'no_prior_top_match' else 0)
    assert strata[16,'no_prior_top_match']['zero_hits'] == 2  # donor 14
    assert strata[16,'identified']['uniform_hits'] == Fraction(2,6)
    assert totals[256]['no_outside'] == totals[256]['old_states'] == 4
    assert not any(strata[256,b]['events'] for b in p.BUCKETS)
    diag['steps'][-1]['prev_clock'] -= 1
    with pytest.raises(ValueError, match='frame or pool mismatch'):
        replay(diag, task, rows)


def test_invalid_numeric_capture_not_silently_a_hit():
    with pytest.raises(ValueError, match='joint L15 drift'):
        p.top_values({'top_positions': [[7,.4,.3,.3,'context']]})


def lag_fixture(diag, rows):
    labels = {(r['block'], r['clock']): r for r in diag['steps']}
    fs = []
    for a, b in zip(rows, rows[1:]):
        block, clock = b['block'], b['clock']
        if a['block'] != block:
            continue
        joint = [0.0]*diag['canvas']
        if (block, clock) in labels:
            for i, v in p.top_values(labels[block, clock]).items():
                joint[i] = v
        else:  # empty old pool, but real previous queries include these rows
            joint[14 if block == 0 else 18] = 1.0
        fs.append(dict(block=block, clock=clock, joint=joint))
    return dict({k:diag[k] for k in (*p.ds.IDENTITY, 'canvas', 'layer')}, metric=diag['joint'], frames=fs)


def test_exact_lag_accounting_and_current_label_is_not_the_rank():
    d, t, rows = frames()
    lag = lag_fixture(d, rows)
    totals, counts = replay(d, t, rows, exact=p.exact_frames(lag, d, rows))
    c = counts[16, 'no_prior_top_match']
    assert [c[k] for k in ('events', 'swaps', 'donor_collisions', 'zero_hits', 'exact_hits', 'lost_hits')] == [2,1,1,2,1,1]
    c = counts[16, 'identified']
    assert c['exact_hits'] == 0 and c['swaps'] == c['tied_max_events'] == 2
    assert c['uniform_hits'] == Fraction(2,6)
    assert totals[256]['no_outside'] == 4
    # Move the last frame's target from row 13 to 12: only the outcome changes.
    d['steps'][-1]['top_positions'][0][0] = 12
    _, altered = replay(d, t, rows, exact=p.exact_frames(lag_fixture(d, rows), d, rows))
    a = altered[16, 'identified']
    assert a['gained_hits'] == a['exact_hits'] == 1 and a['swaps'] == a['tied_max_events'] == 2
    # Ineligible donor / protected seat with a higher score cannot win a tie.
    _, donor, options = p.contenders({0,1,2,3}, {0,1}, [{0,2,3}, {0,1,2,3}])
    assert donor == 1 and p.exact_choice(options, [999,999,1,1]) == 2


@pytest.mark.parametrize('damage', ['identity', 'metric', 'missing_empty_pool', 'duplicate', 'cross_block',
                                    'width', 'negative', 'nan', 'bool', 'wrong_top', 'wrong_top_value'])
def test_invalid_recapture_rejected(damage):
    d, t, rows = frames()
    lag = lag_fixture(d, rows)
    fs = lag['frames']
    if damage == 'identity': lag['seed'] += 1
    elif damage == 'metric': lag['metric'] = 'dk'
    elif damage == 'missing_empty_pool': fs.pop(0)
    elif damage == 'duplicate': fs.append(fs[-1])
    elif damage == 'cross_block': fs[0]['block'] = 256
    elif damage == 'width': fs[0]['joint'].pop()
    elif damage == 'wrong_top': fs[-1]['joint'][5] = 4.0
    elif damage == 'wrong_top_value': fs[-1]['joint'][13] /= 2
    else: fs[0]['joint'][0] = {'negative':-1.0, 'nan':float('nan'), 'bool':True}[damage]
    with pytest.raises(ValueError, match='lag capture|lag frame|joint'):
        replay(d, t, rows, exact=p.exact_frames(lag, d, rows))


def test_frozen30_parity_and_separate_exact_csv(tmp_path, monkeypatch):
    raw, recap, out = (tmp_path/name for name in ('archive', 'recapture', 'reports'))
    raw.mkdir()
    recap.mkdir()
    d, t, rows = frames()
    d.update(pool_kind='non_fresh_resolved', phase='t-1_to_t_same_canvas')
    base, _ = replay(d, t, rows)
    tasks = [dict(t, index=i, task_id=f'synthetic-{i}') for i in range(30)]
    for task in tasks:
        diag = dict(d, **task)
        stem = f"{task['index']:02d}"
        blob = '\n'.join(json.dumps(r) for r in [dict(task, schema='fixture'), *rows]).encode()
        for mtime, folder in enumerate([raw, recap], 1):
            (folder/f'{stem}.trace.jsonl.gz').write_bytes(gzip.compress(blob, mtime=mtime))
            (folder/f'{stem}-native256.json').write_text(json.dumps(dict(task, model='diffusiongemma', policy='native256',
                status='success', output_ids=[1], output_tokens=1, generation_forwards=8, generation_query_rows=2048,
                generation_physical_padded_rows=2048, events=[])))
            (folder/f'{stem}-kvdrift-layer15.json').write_text(json.dumps(diag))
        (recap/f'{stem}-lag-joint-layer15.json').write_text(json.dumps(lag_fixture(diag, rows)))
    manifest = tmp_path/'tasks.json'
    manifest.write_text(json.dumps(dict(samples=tasks)))
    monkeypatch.setattr(p.ds, 'MANIFEST', manifest)
    monkeypatch.setattr(p.ds, 'OUT', tmp_path)
    (raw/'verified.json').write_text(json.dumps(dict(model='gemma', matched_requests=30)))
    p.ds.write_csv(tmp_path/'summary-budget.csv', [dict(group='all', method='frontier', mode='slack', seats_or_q=q,
        steps=c['old_states']*30, pool=c['old_rows']*30, targets=c['target_rows']*30,
        physical=c['physical_q_old_states']*30, base_old=c['old_seats']*30, base_hits=c['zero_total_hits']*30)
        for q,c in base.items()])
    p.run(raw, out)
    bounds = {f.name:f.read_bytes() for f in out.iterdir()}
    p.run(raw, out, recap)
    assert set(bounds) == {'summary.csv', 'coverage.csv'} and all((out/k).read_bytes()==v for k,v in bounds.items())
    with (out/'exact.csv').open() as f:
        report = list(csv.DictReader(f))
    assert [int(r['q']) for r in report] == list(p.ds.BUDGETS)
    for r in report:
        total = int(r['exact_total_hits'])
        assert int(r['requests']) == 30 and int(r['transitions']) == 6*30
        assert int(r['lag_total_lo']) <= total <= int(r['lag_total_hi'])
        assert total-int(r['zero_total_hits']) == int(r['gained_hits'])-int(r['lost_hits'])
    assert int(report[0]['exact_hits']) == int(report[0]['donor_collisions']) == 30
    assert report[-1]['events'] == '0' and report[-1]['exact_total_hits'] == report[-1]['zero_total_hits']

    # Tamper with each independent evidence source; a collector verification flag is not evidence.
    trace = p.ds._read_trace(raw/'00.trace.jsonl.gz')
    for name, key, value, error in [ ('00-native256.json', 'output_ids', [], 'output/work'),
        ('00-native256.json', 'generation_query_rows', 1, 'output/work'),
        ('00-kvdrift-layer15.json', 'steps', [], 'diagnostic')]:
        path = recap/name
        original = path.read_text()
        path.write_text(json.dumps(dict(json.loads(original), **{key:value})))
        with pytest.raises(ValueError, match=error):
            p.read_exact_request(raw, recap, dict(d, **tasks[0]), trace, tasks[0])
        path.write_text(original)
    (recap/'00.trace.jsonl.gz').write_bytes((recap/'01.trace.jsonl.gz').read_bytes())
    with pytest.raises(ValueError, match='native trace'):
        p.run(raw, tmp_path/'invalid', recap)
    (recap/'29-lag-joint-layer15.json').unlink()
    with pytest.raises(ValueError, match='cohort'):
        p.run(raw, tmp_path/'invalid', recap)
    assert not (tmp_path/'invalid').exists()

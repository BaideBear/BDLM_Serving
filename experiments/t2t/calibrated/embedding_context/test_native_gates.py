from contextlib import nullcontext
from fractions import Fraction
import gzip
import json

import numpy as np
import pytest

from experiments.t2t.calibrated.embedding_context import native_gates as g


@pytest.mark.parametrize('score,gate_opens', [(np.nextafter(g.CUTOFF, -np.inf), False),
                                              (g.CUTOFF, True), (0., True)])
@pytest.mark.parametrize('admits', [{6}, {7}])
def test_crowded_inclusive_threshold_and_real_admission_cost(score, gate_opens, admits):
    s = g.Space.build(range(13), {1, 3}, {0, 2, 4, 5, 6, 7, 8, 9})
    assert s.base == set(range(8))
    top, allowed, plans = s.plans([score]*3)
    assert allowed and top == 10  # all tied
    for mode, pick in plans.items():
        swap = mode == 'fixed' or (mode == 'gate5pct' and gate_opens)
        assert pick == (s.base - {7} | {10} if swap else s.base)
        c = g.counts(s, pick, {10}, admits)
        assert c['physical_swaps'] == c['ranked_old_hits_added'] == c['displaced_masks'] == int(swap)
        assert c['uniform_expected_selected_old_hits'] == (Fraction(1, 3) if swap else 0)
        assert c['lost_mask_admissions'] == int(swap and 7 in admits)
        assert c['displaced_nonadmitting_masks'] == int(swap and 7 not in admits)


def test_slack_uses_last_old_and_uniform_conditions_on_outside_only():
    s = g.Space.build(range(12), {5}, {1, 3, 9})
    assert s.base == {5, 1, 3, 9, 0, 2, 4, 6}
    _, _, plans = s.plans([-100 if i == 11 else -101 for i in s.old])
    for pick in plans.values():
        assert pick == s.base - {6} | {11}
        c = g.counts(s, pick, {6, 7, 8}, {9})
        assert c['old_hits_lost'] == c['base_old_hits'] == c['displaced_old'] == 1
        assert c['ranked_net_old_hits'] == -1 and c['ranked_selected_old_hits'] == 0
        assert c['uniform_expected_old_hits_added'] == Fraction(2, 4)
        assert c['uniform_expected_net_old_hits'] == Fraction(-1, 2)
        assert c['lost_mask_admissions'] == c['displaced_masks'] == 0
    # Best in base, runner-up outside: no explicit reservation or physical change.
    top, _, plans = s.plans([3 if i == 0 else 2 if i == 11 else 0 for i in s.old])
    assert top == 0
    for pick in plans.values():
        assert pick == s.base
        c = g.counts(s, pick, {0, 11}, {9})
        assert c['uniform_expected_old_hits_added'] == c['physical_swaps'] == 0
        assert c['uniform_expected_selected_old_hits'] == 1


def test_wider_list_is_not_wider_physical_compute():
    s = g.Space.build(range(13), {1, 3}, {0, 2, 4, 5, 6, 7, 8, 9})
    scores = [3, 2, 1]
    _, _, plans = s.plans(scores)
    pool = dict(g.shortlist_rows(s, scores, plans['gate5pct'], {11, 12}, {7}))
    assert [pool['mask', k]['one_seat_oracle'] for k in g.SHORTLIST] == [0, 1, 1, 1, 1]
    assert [pool['mask', k]['list_targets'] for k in g.SHORTLIST] == [0, 1, 2, 2, 2]
    assert [pool['mask', k]['list_rows'] for k in g.SHORTLIST] == [1, 2, 3, 3, 3]
    assert all(v['swaps'] == v['lost_mask_admissions'] == v['any_outside_oracle'] == 1 for v in pool.values())
    assert not list(g.shortlist_rows(s, scores, s.base, {11, 12}, {7}))


@pytest.mark.parametrize('n', [7, 8, 9])
def test_protected_first_mask_or_no_mask_room(n):
    s = g.Space.build(range(13), range(n), {10, 11})
    top, allowed, plans = s.plans(np.zeros(len(s.old)))
    assert top is not None and not allowed
    assert s.base == set(range(n)) | ({10} if n == 7 else set())
    assert all(p == s.base for p in plans.values())
    assert len(s.base) == max(8, n)


def test_reject_mismatched_spaces():
    with pytest.raises(ValueError, match='partition'):
        g.Space.build(range(12), {2}, {2})
    s = g.Space.build(range(10), {}, range(8))
    for bad in ([0.], [np.nan, 0.]):
        with pytest.raises(ValueError, match='Scores'):
            s.plans(bad)
    for picked, targets, admits in [(s.base - {0} | {8}, {8}, {1}),
                                    (s.base, {1}, {2}), (s.base, {9}, {8})]:
        with pytest.raises(ValueError, match='space mismatch'):
            g.counts(s, picked, targets, admits)


TASK = dict(index=0, dataset='synthetic', task_id='fixture', seed=1)


def trace():
    def row(block, clock, a, b):
        return dict(block=block, clock=clock, before=a, after=b, accepted=[i != 0 for i in b])
    a = [40]*22 + [0]*10
    b = a.copy(); b[22:26] = [5, 6, 7, 8]
    c = b.copy(); c[23:27] = [10, 0, 8, 11]  # strict, revoke, unchanged, admit
    d = [v or 12 for v in c]; d[22:24] = [17, 18]  # old edit and fresh repeat
    e = d.copy(); e[22] = 19  # verification: excluded, even though strict
    x, y = [0]*32, [0]*30 + [20, 21]
    return [dict(schema=1, **TASK), dict(prompt_length=22, mask_id=0),
            row(0, 1, a, b), row(0, 2, b, c), row(0, 3, c, d), row(0, 4, d, e),
            row(32, 1, x, y), row(32, 2, y, y)]


class Spy:
    def __init__(self):
        self.calls = []

    def scores(self, visible, fresh, old, pos):
        assert set(visible) == set(fresh) | set(old)
        self.calls.append((visible, fresh, old, pos))
        return [0.]*len(old)


def test_native_pre_step_data_only_and_block_boundaries():
    spy = Spy()
    totals = g.replay(iter(trace()), TASK, spy, 0, 64)
    # First selection has only MASK + fresh, and the next block resets history.
    (visible, fresh, old, pos), = spy.calls
    assert visible == {22: 5, 25: 8, 23: 10, 26: 11}
    assert old == (22, 25) and fresh == {23, 26}
    assert pos[0][12] == 1 and pos[0][16] == 1  # near prior strict; own write age <= 2
    for c in totals.values():
        assert c['states'] == 3 and c['eligible_old'] == 2 and c['old_targets'] == 1
        assert c['mandatory_fresh'] == 8 and c['natural_mask_admissions'] == 7
    assert totals['fixed']['ranked_selected_old_hits'] == 1
    assert totals['slack_only']['ranked_selected_old_hits'] == 0


def test_observer_receives_completed_prior_transition_only():
    decisions, labels = [], []

    class Watch:
        def select(self, space, scores, picked, visible, previous_changes):
            assert len(decisions) == len(labels)
            decisions.append(previous_changes)

        def outcome(self, targets, admits):
            assert len(decisions) == len(labels) + 1
            labels.append(targets)

    g.replay(iter(trace()), TASK, Spy(), 0, 64, observer=Watch())
    assert decisions == [((22, 0, 5), (23, 0, 6), (24, 0, 7), (25, 0, 8)),
                         ((23, 6, 10), (24, 7, 0), (26, 0, 11)),
                         ((30, 0, 20), (31, 0, 21))]
    assert labels == [set(), {22}, set()]


@pytest.mark.parametrize('error', ['header', 'clock', 'width', 'acceptance', 'continuity', 'prompt', 'vocab'])
def test_trace_rejects_bad_input(error):
    rows = trace()
    if error == 'header': rows[1]['mask_id'] = 1
    if error == 'clock': rows[3]['clock'] = 3
    if error == 'width': rows[2]['before'] = [0]
    if error == 'acceptance': rows[2]['accepted'][23] = False
    if error == 'continuity': rows[3]['before'] = rows[2]['before']
    if error == 'prompt': rows[2]['after'][1] = 3
    if error == 'vocab': rows[2]['after'][22] = 64
    with pytest.raises(ValueError):
        g.replay(iter(rows), TASK, Spy(), 0, 64)


def test_no_selected_states_produces_numeric_zeros():
    full = g.replay(iter(trace()), TASK, Spy(), 0, 64)['fixed']
    empty = g.replay(iter(trace()[:3]), TASK, Spy(), 0, 64)['fixed']
    row = g.public_row('fixed', empty, full, request_index=0)
    assert full and all(row[k] == 0 for k in full)


def test_provenance_requires_pinned_fit_and_both_qualifications(tmp_path, monkeypatch):
    def store(name, data):
        path = tmp_path / name
        path.write_text(json.dumps(data))
        return path
    fit = store('fit.json', {})
    monkeypatch.setattr(g, 'FIT_SHA', g.sha(fit))
    monkeypatch.setattr(g, 'read_context_weights', lambda path: 'ok')
    meta = dict(model='llada21', batch_size=1, block=32, cap=2048, seed_offset=0,
                arms=[dict(budget=8, fresh=0, mask=7, selection='hard_context_fixed')],
                policies=['t2t_q8_f0_m7_hard_context_fixed'], context_weights_sha256=g.FIT_SHA)
    store('verified.json', dict(model='llada21', matched_requests=30))
    store('qualified.json', g.QUALIFIED)
    store('run.json', meta)
    def check():
        return g.qualify(tmp_path, tmp_path, fit)
    assert check() == 'ok'
    for file, bad in [('run.json', dict(meta, context_weights_sha256='different')),
                      ('qualified.json', dict(g.QUALIFIED, matched=29)),
                      ('verified.json', dict(model='llada21', matched_requests=29)), ('fit.json', {'stale': True})]:
        path = tmp_path / file
        orig = path.read_text()
        store(file, bad)
        with pytest.raises(ValueError, match='Unqualified'):
            check()
        path.write_text(orig)


def test_synthetic_run_writes_only_aggregate_counts_after_matching_spaces(tmp_path, monkeypatch):
    study = tmp_path / 'calibrated/embedding_context'
    study.mkdir(parents=True)
    archive = tmp_path / 'exchange/ceiling/requests.csv'
    archive.parent.mkdir(parents=True)
    manifest = tmp_path / 'tasks.json'
    manifest.write_text(json.dumps({'samples': [TASK]}))
    rows = trace()
    with gzip.open(tmp_path / '00.trace.jsonl.gz', 'wt') as f:
        f.writelines(json.dumps(r) + '\n' for r in rows)
    ref = dict(**TASK, policy='zero_reserve', q=8, states=3, physical_q=24, mandatory_fresh=8,
               old_targets=1, base_old_hits=0, natural_mask_admissions=7, base_mask_admissions=7)
    g.write_rows(archive, [ref])
    rank_ref = [dict(metric=k, seats=1, context_hits=1, targets=1) for k in ('Q8/1', 'top-1')]
    g.write_rows(study / 'comparison.csv', rank_ref)
    # Exercise the real ContextRanker with a synthetic embedding and intercept.
    class Projection:
        size = 64

        def unit(self, token):
            return np.ones(16, dtype=np.float32) / 4
    for key, val in dict(STUDY=study, MANIFEST=manifest, CALIBRATION=manifest, check_split=lambda *_: None,
                         qualify=lambda *_: np.r_[1., np.zeros(27)], mask_token=lambda _: 0,
                         input_table=lambda _: nullcontext(None), Projector=lambda _: Projection()).items():
        monkeypatch.setattr(g, key, val)
    args = (tmp_path,) * 4
    rank_ref[1]['seats'] = 2
    g.write_rows(study / 'comparison.csv', rank_ref)
    with pytest.raises(ValueError, match='top-1 ranking space mismatch'):
        g.run(*args)
    assert not (study / 'gated').exists()
    rank_ref[1]['seats'] = 1
    g.write_rows(study / 'comparison.csv', rank_ref)
    g.run(*args)
    req = g.read_csv(study / 'gated/native_q8e1_requests.csv')
    summary = g.read_csv(study / 'gated/native_q8e1_summary.csv')
    assert len(req) == len(summary) == 3
    for r, t in zip(req, summary, strict=True):
        assert set(r) - set(t) == {'request_index', 'dataset'}
        assert set(t) - set(r) == {'requests'}
        assert {k: r[k] for k in r.keys() & t.keys()} == {k: t[k] for k in r.keys() & t.keys()}
        assert int(r['physical_swaps']) == (r['mode'] != 'slack_only')

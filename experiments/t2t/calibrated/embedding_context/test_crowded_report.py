import csv
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from experiments.t2t.calibrated.embedding_context import report
from tools.t2t_results import summarize, table


DIR = report.ROOT / 'experiments/t2t'
ZERO = 't2t_q8_f0_m8_hard_noedit'
GATE = 't2t_q8_f0_m7_hard_context_gate'
BASE, RANK, RANDOM = ('t2t_q8_f0_m7_hard_context_once_' + s for s in ('base', 'rank', 'random'))
FIELDS = ('N', 'F', 'Q', 'physical_Q', 'max_q', 'correct', 'failed', 'truncated')


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def scored(out, fault=None):
    _, old = report.scored_rows(DIR / 'hard', ZERO)
    requests, summaries = [], []
    for policy in (RANDOM, BASE, RANK):
        meta = dict(model='llada21', policy=policy, nominal_q=8, fresh=0, mask=7, edit=1)
        arm = [dict(r, **meta) | {k:int(r[k]) for k in FIELDS} for r in old]
        incorrect = [r for r in arm if not r['correct'] and not r['failed']]
        if policy == RANK and fault != 'quality_tie':
            incorrect[0]['correct'] = 1
        if policy == RANDOM:
            incorrect[1]['correct'] = 1
            arm[0]['physical_Q'] += arm[0]['F']
            arm[0]['N'] += 1
        if policy == RANK and fault == 'tpf_loss':
            for k in ('N', 'F', 'Q', 'physical_Q'):
                arm[0][k] *= 1000
            arm[0]['F'] += 1
        if policy == BASE and fault == 'swapped_outcomes':
            other = next(r for r in arm if r['correct'] != arm[0]['correct'])
            arm[0]['correct'], other['correct'] = other['correct'], arm[0]['correct']
        if policy == BASE and fault in FIELDS:
            arm[0][fault] += 1
        if policy == RANDOM and fault == 'missing':
            arm.pop()
        if policy == RANK and fault == 'wrong_mask':
            arm[0]['mask'] = 8
        requests.extend(reversed(arm))
        summaries.append(dict(meta, dataset='all', **summarize(arm)))
    table(out / 'llada21-online-requests.csv', requests)
    table(out / 'llada21-online-summary.csv', summaries)


def test_crowded_cli_matched_and_delta_delta(tmp_path, capsys):
    scored(tmp_path)
    report.main(['--crowded', '--out', str(tmp_path)])
    assert 'feasibility vs one-switch base: PASS' in capsys.readouterr().out
    arms = {r['policy']: r for r in read(tmp_path / 'crowded-arms.csv')}
    assert arms.keys() == {BASE, RANK, RANDOM, ZERO, GATE}
    for a in arms.values():
        n, f, q = (int(a[k]) for k in ('N', 'F', 'physical_Q'))
        assert a['requests'] == '30'
        assert float(a['physical_efficiency']) == n/q and float(a['physical_q_per_f']) == q/f
    prs = read(tmp_path / 'crowded-matched.csv')
    pairs = {(r['policy'], r['reference']): r for r in prs}
    assert len(prs) == 3 and pairs.keys() == {(RANK, BASE), (RANK, RANDOM), (RANDOM, BASE)}
    assert [(pairs[k]['purpose'], pairs[k]['primary_feasible']) for k in
            ((RANK, BASE), (RANK, RANDOM), (RANDOM, BASE))] == [
                ('primary_feasibility', 'True'), ('same_gate_placement', ''), ('descriptive', '')]
    _, hist = report.scored_rows(DIR / 'hard', ZERO)
    keys = sorted(report.request_key(r) for r in hist)
    draws = report.stratified_draws(keys)
    matrices = {p: report.bundle(DIR / 'hard' if p == ZERO else
                DIR / 'calibrated/embedding_context/gated' if p == GATE else tmp_path, p, keys)[1] for p in arms}
    for (p, ref), row in pairs.items():
        for k, v in report.paired_uncertainty(matrices[p], matrices[ref], draws).items():
            assert float(row[k]) == pytest.approx(v)
    details = read(tmp_path / 'crowded-request-deltas.csv')
    assert len({(r['policy'], r['reference'], r['index']) for r in details}) == len(details) == 90
    dd = read(tmp_path / 'crowded-delta-delta.csv')
    assert [(r['policy'], r['base'], r['other'], r['other_base']) for r in dd] == [
        (RANK, BASE, RANDOM, BASE), (RANK, BASE, GATE, ZERO)]
    for row in dd:
        a, b, c, d = (matrices[row[k]] for k in ('policy', 'base', 'other', 'other_base'))
        for metric, get, reduce in [('correct', lambda x: x[:,0], np.sum),
                                    ('TPF', lambda x: x[:,1]/x[:,2], np.mean)]:
            diff = (get(a) - get(b)) - (get(c) - get(d))
            name = f'delta_delta_{metric}'
            assert float(row[name]) == pytest.approx(reduce(diff), abs=1e-12)
            assert [float(row[name+s]) for s in ('_lo', '_hi')] == pytest.approx(
                np.quantile(reduce(diff[draws], axis=1), [.025, .975]), abs=1e-12)
    labels = {e.text for e in ET.parse(tmp_path/'crowded.svg').iter('{http://www.w3.org/2000/svg}text')}
    assert {'One switch: ranked', 'No edit (historical)', 'Repeated gate (directed)'} <= labels
    baselines = report.read_baselines(report.ROOT/'experiments/baselines/summary.csv')
    for kind in ('ibs', 'obs'):
        p = baselines['llada21', 8, kind]
        assert {f'Optimus {kind.upper()}8 (historical)', f'{p.tpf:.3f}', f'{p.correct}/30',
                f'{p.efficiency:.2%}', f'{p.q_per_f:.2f}'} <= labels
    assert not list(tmp_path.glob('gated*')) and not (tmp_path/'matched.csv').exists()


@pytest.mark.parametrize('fault', ['swapped_outcomes', 'max_q', 'N', 'F', 'physical_Q', 'truncated',
                                   'missing', 'wrong_mask'])
def test_crowded_rejects_mismatches_before_output(tmp_path, fault):
    scored(tmp_path, fault)
    with pytest.raises(ValueError, match='base differs|mismatch|Missing'):
        report.gated_report(tmp_path, crowded=True)
    assert not list(tmp_path.glob('crowded*'))


@pytest.mark.parametrize('fault', ['quality_tie', 'tpf_loss'])
def test_primary_needs_more_correct_and_unrounded_tpf(tmp_path, fault):
    scored(tmp_path, fault)
    assert report.gated_report(tmp_path, crowded=True) == {RANK: False}
    if fault == 'tpf_loss':
        a = {r['policy']: r for r in read(tmp_path/'crowded-arms.csv')}
        assert round(float(a[RANK]['macro_tpf']), 3) == round(float(a[BASE]['macro_tpf']), 3)

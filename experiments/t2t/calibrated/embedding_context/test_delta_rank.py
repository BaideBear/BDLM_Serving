from collections import Counter
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import expit

from experiments.t2t.calibrated.embedding_context import delta_rank as d


def test_direction_mask_changes_radius_weight_and_no_neighbor():
    # ID 0 stands for MASK. 0 -> 1 at distance 8 and 1 -> 0 at distance 1.
    basis = np.eye(16)
    changes = ((0, 0, 1), (7, 1, 0), (8, 90, 91), (17, 92, 93))
    got = d.delta_features(basis.__getitem__, [8, 31], {8: 0}, changes)
    v = (basis[0]-basis[1])*7/9
    np.testing.assert_allclose(got, [[np.linalg.norm(v), basis[0] @ v], [0, 0]])
    np.testing.assert_array_equal(d.delta_features(basis.__getitem__, [8], {}, ()), [[0, 0]])


@pytest.mark.parametrize('desired,targets,orig,new', [(9,{8},1,0), (9,{9},0,1), (16,{16},0,0)])
def test_selection_before_labels_one_outside_seat_same_victim(desired, targets, orig, new):
    space = d.g.Space.build(range(18), (), range(8))
    scores = np.zeros(10)
    pick = space.plans(scores)[2]['gate5pct']
    visible = {i:int(i == desired) for i in space.old}
    # Row 10 is the change itself, so has no features. Row 16 is outside the top 8.
    obs = d.Observer(np.eye(16).__getitem__, np.array([0., 3.]))
    obs.select(space, scores, pick, visible, ((10, 0, 1),))
    chosen = obs.pending[3]
    assert chosen == (9 if desired == 9 else 10)
    obs.outcome(targets, {7})
    c = obs.totals['crowded']
    assert [c[k] for k in ('physical_q_on_swaps', 'one_seat_queries', 'lost_mask_admissions',
                           'swaps', 'list_rows', 'original_hits', 'reranked_hits')] == [8,1,1,1,8,orig,new]
    assert obs.batches is None and obs.pending is None
    before = c.copy()
    obs.select(space, scores, space.base, {}, ())
    obs.outcome({17}, set())
    assert c == before


def test_fit_has_fixed_offset_two_penalized_slopes_and_no_intercept():
    x = np.array([[1., .4], [0., .1], [2., -1.2], [.5, -.2]])
    o = np.array([-3., -1., -.5, -2.])
    y = np.array([1,0,0,1])
    w = d.fit([(x[:2], o[:2], y[:2]), (x[2:], o[2:], y[2:])])
    assert w.shape == (2,)
    np.testing.assert_allclose(x.T @ (expit(o + x@w)-y) + 2*w, 0, atol=1e-5)


def test_manifest_guards_and_holdout_identity(tmp_path, monkeypatch):
    cal = d.g.CALIBRATION.parent
    train, test = d.manifests(cal)
    assert len(train) == 24 and len(test) == 30
    assert {t['prompt_sha256'] for t in train}.isdisjoint(t['prompt_sha256'] for t in test)
    bad = tmp_path/'weights.json'
    bad.write_text('{}')
    with pytest.raises(ValueError, match='manifest'):
        d.manifests(tmp_path)
    monkeypatch.setattr(d.g, 'MANIFEST', bad)
    with pytest.raises(ValueError, match='manifest'):
        d.manifests(cal)


@pytest.mark.parametrize('hits,passed', [((18,10),False), ((19,7),False), ((19,8),True)])
def test_qualify_then_fit_only_calibration_before_native_scoring(tmp_path, monkeypatch, hits, passed):
    cal, native = d.g.CALIBRATION.parent, tmp_path/'native'
    train, test = d.manifests(cal)
    events, token, w = [], object(), np.array([.1, -.3])

    def qualify(n, s, weights):
        assert (n, s, weights) == (native, tmp_path/'src', tmp_path/'fit')
        events.append('qualify')
        return np.zeros(28)

    def scan(folder, tasks, ranker, mask, size, slopes=None, archive=None):
        assert mask == 0 and size == 64
        if folder == cal:
            assert tasks == train and slopes is archive is None
            events.append('cal')
            return [(token, None, np.array([1]))], {}
        assert folder == native and tasks == test and slopes is w and len(archive) == 30
        events.append('frozen')
        return [], {k: Counter(original_hits=o, reranked_hits=h) for k, o, h in
                    zip(('crowded', 'slack'), (16,8), hits, strict=True)}

    def fit(batches):
        assert len(batches) == 1 and batches[0][0] is token
        events.append('fit')
        return w

    monkeypatch.setattr(d.g, 'qualify', qualify)
    monkeypatch.setattr(d.g, 'mask_token', lambda _: 0)
    monkeypatch.setattr(d.g, 'input_table', lambda _: nullcontext(None))
    monkeypatch.setattr(d.g, 'Projector', lambda _: SimpleNamespace(size=64, unit=None))
    monkeypatch.setattr(d, 'scan', scan)
    monkeypatch.setattr(d, 'fit', fit)
    assert d.run(cal, native, tmp_path/'src', tmp_path/'fit', tmp_path, tmp_path) == passed
    assert events == ['qualify', 'cal', 'fit', 'frozen']
    rows = d.g.read_csv(tmp_path/'summary.csv')
    assert len(rows) == 2 and set(rows[0]) == {'requests', 'victim', *d.FIELDS}

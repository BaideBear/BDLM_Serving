"""Planner-only contracts: intercept the physical call before any model or RNG."""
from types import SimpleNamespace

import pytest

from bdlm_serving.methods.t2t.budget_family import Law
from experiments.t2t.serving_slots.gemma_fixed_dual import (
    ABLATIONS, B, FULL, AblationRequest, Canvas, StrictRows,
)


def request(arm):
    def canvas(i):
        return Canvas(i, StrictRows(48, ordinal=i), SimpleNamespace(phase='sample'))

    r = object.__new__(AblationRequest)
    h = canvas(0)
    r.rows, r.sampler = h.rows, h.sampler
    r.origin, r.ordinal, r.cursor, r.next_side = 13, 0, 0, 0
    r.law, r.ablation, r.max_new_tokens = Law(), arm, 2*B
    r.bank, r.rebuild, r.tail = SimpleNamespace(live=[]), False, None
    r._spawn = lambda: canvas(r.ordinal+1)
    r._no_accepted_eos = lambda _: True

    def planned(states, pos, sampled, prefix, cold, offer, route):
        return dict(parts=[], route=route, positions=pos, sample=sorted(sampled), prefix=sorted(prefix))
    r._apply = planned
    return r


@pytest.mark.parametrize('arm', ABLATIONS)
def test_top_opens_immediately_and_overwrites_full_debt(arm):
    r = request(arm)
    both = list(range(r.origin,r.origin+2*B))
    e = r._advance(2*B)
    assert e['route']=='dense_joint' and e['opened']==1 and list(e['positions'])==e['sample']==both
    r.rows.dirty = r.tail.rows.dirty = FULL
    same = r.tail
    e = r._advance(2*B)
    assert e['route']=='dense_joint' and e['opened'] is None and r.tail is same
    assert list(e['positions'])==e['sample']==e['dirty_before_absolute']==both


def test_downshift_retains_bank_sampler_and_parked_debt():
    r = request('00')
    r._advance(2*B)
    r.tail.rows.initialized = r.tail.rows.dirty = FULL
    tail, bank = r.tail, r.bank
    e = r._advance(32)
    assert e['route']=='drain' and not e['sample'] and list(e['positions'])==list(range(269,301))
    assert r.bank is bank and r.tail is tail and r.tail.sampler is tail.sampler


@pytest.mark.parametrize('q', [32,64,128,256])
def test_fixed_lower_tiers(q):
    out = {}
    for arm in ABLATIONS:
        r = request(arm)
        first = r._advance(q)
        assert first['route']=='sparse' and first['sample']==list(range(r.origin,r.origin+q))
        r.rows.initialized, r.rows.accepted = FULL, frozenset(range(3*B//4))
        out[arm] = r._advance(q)
    split = r.origin+B
    assert out['00']['opened'] is None and max(out['00']['positions']) < split
    cross = out['10']
    assert cross['opened']==1 and min(cross['sample']) < split <= max(cross['sample'])
    assert len(cross['positions'])<=q and all(x<r.origin+3*B//4 for x in cross['prefix'])
    build = out['11']
    assert build['route']=='build_sample' and build['opened']==1
    assert list(build['positions'])==build['sample']==list(range(split,split+B))


@pytest.mark.parametrize('arm', ['10','11'])
@pytest.mark.parametrize('q', [32,64,128,256])
def test_late_open_requires_full_head_before_tick(arm, q):
    r = request(arm)
    r.rows.accepted = frozenset(range(3*B//4))
    r.rows.initialized = FULL - {B-1}
    # Exactly at the MASK threshold, but the last head KV row is still cold.
    e = r._advance(q)
    assert e['opened'] is None and r.tail is None
    assert e['positions']==(r.origin+B-1,) and e['sample']==list(e['positions'])
    # The test intercepts projection; publish that last row as a completed tick.
    r.rows.initialized = r.rows.initialized | {B-1}
    e = r._advance(q)
    assert e['opened']==1 and e['route']==('build_sample' if arm=='11' else 'sparse')


def test_low_open_barriers_and_encoder_epoch():
    r = request('11')
    r.rows.accepted, r.rows.initialized = frozenset(range(192)), FULL
    r.rows.dirty = frozenset([4])
    e = r._advance(64)
    assert e['route']=='drain' and e['positions']==(17,) and not e['sample'] and r.tail is None
    r.rows.dirty = frozenset()
    r._no_accepted_eos = lambda _: False
    assert r._advance(64)['opened'] is None and r._advance(512)['route']=='dense_single'
    r.rows.accepted = FULL
    r._no_accepted_eos = lambda _: True
    assert r._advance(32)['route']=='verify' and r.tail is None
    r.ordinal, r.rebuild = 1, True
    e = r._advance(64)
    assert e['route']=='rebuild' and e['positions']==tuple(range(269,333)) and not e['sample']

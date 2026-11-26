from collections import Counter
from copy import deepcopy
import csv
import json
import math
import random

import pytest

from tools import context_gates_audit as a


SLACK, SRAND, GATE, GRAND = a.POLICIES
SEED = 17000


def move(fresh, masks, old, base, reason, rank=None, score=None, victim=None, entrant=None):
    return dict(fresh=list(fresh), masks=list(masks), old_pool=list(old), base=list(base),
                reason=reason, rank=rank, score=score, victim=victim, entrant=entrant)


def event(stage, b, clock, picked, fresh=0, m=None):
    absolute = [32*b+i for i in picked] if stage != 'prefill' else list(range(32*b))
    q = len(absolute)
    e = dict(stage=stage, block=b, hard_clock=clock, F=1, actual_q=q, physical_q=q, q_tokens=q,
             q_budget=8, overflow=max(0,q-8), mandatory_fresh=fresh, hard_strict_edits=0,
             absolute_positions=absolute, q_start=min(absolute), q_end=max(absolute)+1,
             layer_physical_q=[q]*20, reserved_extra_edit_positions=[], reserved_strict_edit_positions=[],
             phase=stage if stage in ('prefill','commit') else 'denoise', prefix_kv_length=0 if stage=='prefill' else b*32,
             visible_kv=(b+int(stage!='prefill'))*32, context_version=0, store_kv=stage in ('prefill','commit'),
             approximate_same_block_kv=stage == 'selected')
    if m:
        e['context_move'] = m
        if m['entrant'] is not None:
            e['reserved_extra_edit_positions'] = [32*b+m['entrant']]
    return e


def placed(m, clock=3):
    picked = set(m['base'])
    if m['victim'] is not None:
        picked.remove(m['victim'])
        picked.add(m['entrant'])
    return event('selected', 1, clock, sorted(picked), len(m['fresh']), m)


def replay(m, p):
    return a.placement(placed(m), set(m['fresh']+m['masks']+m['old_pool']), p, SEED)


@pytest.mark.parametrize('m,p', [
    (move(range(9), [9], [10], range(9), 'overflow'), GRAND),
    (move(range(8), [8], [], range(8), 'no_room'), GATE),
    (move(range(7), [7,8], [], range(8), 'no_old'), SLACK),
    (move(range(7), [7,8], [9], range(8), 'protected_mask'), GATE),
    (move([], range(8), [8], range(8), 'no_slack'), SLACK),
    (move([], range(8), [8], range(8), 'below_threshold', 8, math.nextafter(a.CUTOFF, -math.inf)), GATE),
    (move([], range(6), [6,7,8], range(8), 'already_selected', 7, -1000), SRAND),
    (move([], range(6), [6,7,8], range(8), 'swap_old', 8, -1000, 7, 8), SLACK),
    (move([], range(8), [8,9], range(8), 'swap_mask', 9, a.CUTOFF, 7, 9), GATE),
])
def test_reason_priority_and_threshold(m, p):
    replay(m, p)
    if m['entrant'] is None:
        e = placed(m)
        e['reserved_extra_edit_positions'] = e['absolute_positions'][:1]
        with pytest.raises(ValueError):
            a.placement(e, set(m['fresh']+m['masks']+m['old_pool']), p, SEED)


def test_random_draw_replayed_exactly_in_both_contexts():
    for masks, reason, p in ((range(6), 'swap_old', SRAND), (range(8), 'swap_mask', GRAND)):
        outside = list(range(8,32))
        i = random.Random('context-swap:17000:1:3').choice(outside)
        m = move([], masks, range(len(masks),32), range(8), reason, 31, 0, 7, i)
        replay(m, p)
        m['entrant'] = next(x for x in outside if x != i)
        with pytest.raises(ValueError):
            replay(m, p)


@pytest.mark.parametrize('bad', [dict(reason='no_slack'), dict(reason='already_selected'), dict(base=list(range(7))),
    dict(rank=None), dict(rank=0), dict(rank=True), dict(score=math.nan), dict(score=math.inf),
    dict(old_pool=[8,9,9]), dict(fresh=[1]), dict(victim=0), dict(victim=6), dict(entrant=8)])
def test_invalid_crowded_move(bad):
    m = move([], range(8), [8,9], range(8), 'swap_mask', 9, 0., 7, 9)
    m.update(bad)
    with pytest.raises((ValueError, KeyError)):
        replay(m, GATE)


def test_unexamined_evidence_must_be_null():
    m = move([], range(8), [8,9], range(8), 'no_slack', 9, -1.)
    with pytest.raises(ValueError):
        replay(m, SLACK)


def request(s, policy):
    # First absolute block has 26 prompt rows. No eligible old rows after bootstrap.
    m = move([26,27], range(28,32), [], range(26,32), 'no_old')
    events = [event('prefill',1,1,[]), event('bootstrap',1,2,range(32)),
              placed(m), event('verify',1,4,range(32),4), event('commit',1,5,range(32))]
    for j,(before,accepted,after) in enumerate(((6,2,4),(4,4,0),(0,0,0)),1):
        events[j].update(masked_before=before, selected_masked_before=before, accepted=accepted, edited=0,
                         masked_after=after, step_in_block=j, post_steps=int(j==3))
    return dict({k:s[k] for k in ('seed','task_id','dataset','index')}, model='llada21', policy=policy,
        status='success', events=events, generation_forwards=4, generation_query_rows=102,
        generation_physical_padded_rows=102, max_q=32, prefill_rows=32, t2t_edits=0,
        forwards_by_phase=dict(bootstrap=1, selected=1, verify=1, commit=1),
        q_by_phase=dict(bootstrap=32, selected=6, verify=32, commit=32), q_histogram={'32':3,'6':1},
        output_ids=[314], output_tokens=1, tpf=.25, token_efficiency=1/102)


S = dict(index=0, seed=SEED, dataset='gsm8k', task_id='gsm8k/167')


@pytest.mark.parametrize('mutate', [lambda r: r.update(seed=17001), lambda r: r.update(t2t_edits=2),
    lambda r:r['events'][2].pop('context_move'), lambda r:r['events'][3].update(context_move=None),
    lambda r:r['events'][2]['layer_physical_q'].pop(), lambda r:r['events'][2].update(hard_clock=2),
    lambda r:r['events'][2].update(reserved_strict_edit_positions=[58]),
    lambda r:r['events'][2].update(mandatory_fresh=3), lambda r:r['events'][2].update(accepted=3),
    lambda r:r['events'][2]['context_move'].update(base=list(range(6))),
    lambda r:r['q_histogram'].update({'32':4}), lambda r:r.update(generation_physical_padded_rows=106),
    lambda r:r['events'].pop(), lambda r:r.update(prefill_rows=31)])
def test_physical_request_and_accounting(mutate):
    r = request(S, SLACK)
    c = a.audit_request(r,S,SLACK)
    assert (c['selected_Q'], c['no_old'], c['swaps'], c['admissions'], c['F'],c['Q']) == (6,1,0,6,4,102)
    mutate(r)
    with pytest.raises(ValueError):
        a.audit_request(r,S,SLACK)


def test_history_identifies_strict_rows_from_next_selected_step():
    # Q={0,1,3,4}; MASK 0 admitted, row 1 revoked, row 3 strictly edited.
    prev = {0,2}, {0,1,3,4}, {3}, {3}, 1, 1, 1
    a.transition(prev, {1,2}, {0,3})
    for masks, fresh in [({0,1},{2,3}), ({1,2},{0,4}), ({1,2},{0,5}), ({1,2},{1,3})]:
        with pytest.raises(ValueError):
            a.transition(prev, masks, fresh)


def rollout(policy):
    rows = set(range(32))
    path = [event('bootstrap',0,1,range(32))]
    path[0].update(masked_before=32, masked_after=23, accepted=9, edited=0, selected_masked_before=32,
                   step_in_block=1, post_steps=0)
    masks, fresh = set(range(9,32)), set(range(9))
    while masks:
        old = rows-fresh-masks
        free = max(0, 8-len(fresh))
        base = fresh | set(sorted(masks)[:free])
        base.update(sorted(old)[:max(0,free-len(masks))])
        clock = len(path)+1
        rank = score = victim = entrant = None
        if len(fresh)>8:
            reason = 'overflow'
        elif free == 0:
            reason = 'no_room'
        elif clock == 3:
            reason, rank, score, victim = 'swap_mask', 8, 0., 16
            entrant = rank if policy == GATE else random.Random('context-swap:17000:0:3').choice(sorted(old))
        elif len(masks) >= free:
            reason, rank, score = 'below_threshold', min(old), -100.
        else:
            reason, rank, score = 'already_selected', min(old), -100.
        m = move(sorted(fresh),sorted(masks),sorted(old),sorted(base),reason,rank,score,victim,entrant)
        picked = base if entrant is None else (base-{victim})|{entrant}
        e = event('selected',0,clock,sorted(picked),len(fresh),m)
        admissions = masks & picked
        if entrant is not None:
            admissions = {9}
            e.update(hard_strict_edits=1, reserved_strict_edit_positions=[entrant])
        e.update(masked_before=len(masks),selected_masked_before=len(masks & picked), accepted=len(admissions),
                 edited=int(entrant is not None),masked_after=len(masks-admissions),step_in_block=clock,post_steps=0)
        path.append(e)
        fresh = admissions if entrant is None else admissions|{entrant}
        masks = masks-admissions
    e = event('verify',0,len(path)+1,range(32),len(fresh))
    e.update(masked_before=0,masked_after=0,accepted=0,edited=0,selected_masked_before=0,step_in_block=len(path)+1,post_steps=1)
    path.extend([e,event('commit',0,len(path)+2,range(32))])
    qs = Counter()
    for e in path:
        qs[e['stage']] += e['physical_q']
    q = sum(qs.values())
    return dict(request(S,policy),events=path,prefill_rows=0,generation_forwards=len(path), generation_query_rows=q,
                generation_physical_padded_rows=q, tpf=1/len(path),token_efficiency=1/q, t2t_edits=1,
                q_by_phase=dict(qs),forwards_by_phase=dict(Counter(e['stage'] for e in path)),
                q_histogram=dict(Counter(str(e['physical_q']) for e in path)))


def test_end_to_end_overflow_then_rank_random_fork():
    x, y = (rollout(p) for p in (GATE, GRAND))
    for r in (x,y):
        c = a.audit_request(r,S,r['policy'])
        assert (c['overflow'], c['fresh_overflow_rows'], c['reserved_hits'], c['swaps'], c['admissions']) == (1,1,1,1,32)
        assert c['followup_mask_slot'] == 1
    c, clock, block = a.audit_pair(x,y)
    assert (clock, block, c['shared_generation_Q'], c['fork_Q_per_arm'], c['fork_left_admissions'],
            c['fork_right_hits']) == (3,0,41,8,1,1)
    y['events'][2]['reserved_strict_edit_positions'] = []
    with pytest.raises(ValueError):
        a.audit_request(y,S,GRAND)


@pytest.mark.parametrize('stage,fresh,masks,name', [
    ('verify', 1, 0, 'dense_verify'), ('selected', 9, 12, 'overflow'),
    ('selected', 8, 1, 'mask_slot'), ('selected', 6, 2, 'neutral_slot'),
    ('selected', 1, 8, 'mask_slot')])
def test_hit_next_mandatory_slot(stage, fresh, masks, name):
    assert a.followup(stage, fresh, masks) == name


def test_nonfork_checks_all_measured_events_and_output():
    r, t = request(S,SLACK), request(S,GATE)
    c, clock, block = a.audit_pair(r,t)
    assert (c['shared_noops'], c['shared_events'], c['no_fork'], clock, block)==(1,5,1,None,None)
    for mutate in [lambda x:x['events'][2].update(edited=1), lambda x:x['output_ids'].append(2),
                   lambda x:x['events'].pop()]:
        bad = deepcopy(t)
        mutate(bad)
        with pytest.raises(ValueError):
            a.audit_pair(r,bad)


def test_pair_same_swap_collision_and_first_fork():
    r, t = request(S,GATE), request(S,GRAND)
    i = random.Random('context-swap:17000:1:3').choice(list(range(8,32)))
    m = move([],range(8),range(8,32),range(8),'swap_mask',i,0.,7,i)
    def insert(raw, move):
        e = placed(move)
        raw['events'][2].update(e)
    insert(r,m)
    insert(t,deepcopy(m))
    c,_,_ = a.audit_pair(r,t)
    assert c['rank_random_collisions'] == c['shared_swaps'] == 1 and c['shared_noops'] == 0
    for raw in (r,t):
        raw['events'][2]['context_move']['rank'] = 8 if i != 8 else 9
    rank = r['events'][2]['context_move']['rank']
    insert(r,dict(m,rank=rank,entrant=rank))
    t['events'] = t['events'][:3]  # Outcomes and all later history are unconstrained between arms.
    t['events'][2]['edited'] = 1
    c,clock,block = a.audit_pair(r,t)
    assert (c['fork_two_swaps'], c['shared_events'], clock, block) == (1,2,3,1)
    for key,new in [('layer_physical_q',[8]*19), ('masked_before',9)]:
        bad = deepcopy(t)
        bad['events'][2][key] = new
        with pytest.raises(ValueError):
            a.audit_pair(r,bad)
    for key,new in [('score',-1.), ('rank',10), ('masks',[7])]:
        bad = deepcopy(t)
        bad['events'][2]['context_move'][key] = new
        with pytest.raises(ValueError):
            a.audit_pair(r,bad)


def write(path, data):
    path.write_text(json.dumps(data))


def cohort(tmp):
    dirs = [tmp/'slack',tmp/'gate']
    for d, group in zip(dirs, [a.POLICIES[:2],a.POLICIES[2:]]):
        d.mkdir()
        specs = [dict(budget=8,fresh=0,mask=7,selection=p.split('m7_')[1]) for p in group]
        write(d/'run.json',dict(model='llada21',batch_size=1,cap=2048,block=32,seed_offset=0,
              context_weights_sha256=a.WEIGHTS, policies=group, arms=specs))
        write(d/'qualified.json',dict(model='llada21',full_width=32,matched=30))
        for p in group:
            write(d/f'complete-{p}.json',dict(policy=p,requests=30))
            for s in a.load(a.MANIFEST)['samples']:
                write(d/f"{s['index']:02d}-{p}.json",request(s,p))
    return dirs


def test_complete_disjoint_cohort_csv(tmp_path):
    dirs = cohort(tmp_path)
    out = tmp_path/'out'
    a.audit(dirs, out)
    for name, size in [('requests',300),('summary',10)]:
        p = out/f'context-gates-{name}.csv'
        assert not any(s in p.read_bytes() for s in [b'\r', b'/data1', b'n164', b'output_ids', b'314'])
        with p.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == size
        assert len([r for r in rows if r['comparison']]) == 6*size//10
        if name == 'summary':
            assert all(r['requests']=='30' and (r['no_fork']=='30' if r['comparison'] else r['noops']=='30') for r in rows)


@pytest.mark.parametrize('issue', ['missing-group','duplicate','unqualified','incomplete','29','same-wrong-seed',
                                   'unrelated','offset','weights'])
def test_reject_bad_runs_without_publication(tmp_path,issue):
    dirs = cohort(tmp_path)
    d = dirs[0]
    if issue == 'missing-group':
        dirs.pop()
    elif issue == 'duplicate':
        dirs.append(d)
    elif issue == 'unqualified':
        write(d/'qualified.json',{})
    elif issue == 'incomplete':
        write(d/f'complete-{SRAND}.json',dict(policy=SRAND,requests=29))
    elif issue in ('29','same-wrong-seed'):
        for folder,p in zip([d,d,dirs[1],dirs[1]], a.POLICIES):
            f = folder/f'00-{p}.json'
            if issue=='29':
                f.unlink()
            else:
                write(f,dict(a.load(f),seed=1))
    else:
        meta = a.load(d/'run.json')
        if issue=='unrelated':
            meta['policies'][0]='hard_context_fixed'
        else:
            meta['seed_offset' if issue=='offset' else 'context_weights_sha256']=1
        write(d/'run.json',meta)
    with pytest.raises(ValueError):
        a.audit(dirs,tmp_path/'out')
    assert not (tmp_path/'out').exists()

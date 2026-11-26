from collections import Counter
from copy import deepcopy
import csv
from itertools import product
import random

import pytest

from tools import context_gates_audit as a
from test_context_gates_audit import event, move, request, S, write


BASE, RANK, RAND = a.ONCE
SHA = 'a'*64


def path(policy, s=S, *, trigger=True, collision=False):
    # Absolute block 1. First overflow, then a below-threshold check, then
    # an exact-cutoff trigger. The control admits its victim; treatments edit.
    allrows = set(range(32))
    ev = [event('prefill', 1, 1, []), event('bootstrap', 1, 2, range(32))]
    ev[-1].update(masked_before=32, selected_masked_before=32, masked_after=23, accepted=9,
                  edited=0, step_in_block=1, post_steps=0)
    used = False
    masks, fresh = set(range(9,32)), set(range(9))
    while masks:
        tick = len(ev)+1
        free = max(0,8-len(fresh))
        old = allrows-masks-fresh
        base = fresh | set((sorted(masks)+sorted(old))[:free])
        picked, extra, changed = base, set(), set()
        rank = victim = entrant = score = None
        anchor = trigger and tick == 5
        if anchor:
            other = random.Random(f"context-swap:{s['seed']}:1:{tick}").choice(sorted(old-base))
            rank = other if collision else min(old-{other})
            victim, entrant, score, reason = max(base&masks), rank, a.CUTOFF, 'swap_mask'
            targeted, placebo = ((base-{victim})|{i} for i in (rank,other))
            picked = {BASE:base, RANK:targeted, RAND:placebo}[policy]
            extra = changed = picked-base
            admitted = {victim} if policy==BASE else {min(picked&masks)}
            watched = sorted({rank,other,victim})
            plan = dict(base=sorted(base), targeted=sorted(targeted), random=sorted(placebo), executed=sorted(picked),
                        victim=victim, candidate=rank, random_candidate=other, clock=tick, risk_score=score,
                        fresh=sorted(fresh), masks=sorted(masks), stratum='mask', pre_sha256=SHA, post_rng_sha256=SHA,
                        after_sha256=SHA, after_admissions=1, victim_queried=policy==BASE, victim_accepted=policy==BASE,
                        victim_unresolved_before=True, strict_rows=sorted(extra), revoked_rows=[],
                        resident_l15=[dict(row=i,dK=float(i in picked),dV=0.) for i in watched])
        else:
            admitted = ({min(base&masks)} if tick==4 else base&masks)
            if len(fresh)>8:
                reason = 'overflow'
            elif not free:
                reason = 'no_room'
            elif free < 2 and len(masks)>=free:
                reason = 'protected_mask'
            elif free <= len(masks):
                reason, score, rank = 'below_threshold', a.CUTOFF-1., min(old)
            else:
                reason, score, rank, victim = 'swap_old', -100., max(old-base), max(old&base)
                entrant = rank
        e = event('selected',1,tick, sorted(picked), len(fresh))
        if not used:
            e['crowded_screen'] = move(sorted(fresh),sorted(masks),sorted(old),sorted(base),reason,rank,score,victim,entrant)
        if anchor:
            used = True
            e['intervention'] = plan
        e.update(masked_before=len(masks), selected_masked_before=len(picked&masks), masked_after=len(masks-admitted),
                 accepted=len(admitted), edited=len(changed), hard_strict_edits=len(changed),
                 reserved_extra_edit_positions=[i+32 for i in extra], reserved_strict_edit_positions=[i+32 for i in extra],
                 step_in_block=tick-1, post_steps=0)
        ev.append(e)
        masks, fresh = masks-admitted, admitted|changed
    v = event('verify',1,len(ev)+1,range(32),len(fresh))
    v.update(masked_before=0,selected_masked_before=0,masked_after=0,accepted=0,edited=0,
             step_in_block=len(ev),post_steps=1)
    ev.extend([v,event('commit',1,len(ev)+2,range(32))])
    for e in ev[1:-1]:
        e.update(trajectory_sha256=SHA,accepted_sha256=SHA)
    fs, qs, hist = Counter(), Counter(), Counter()
    for e in ev[1:]:
        fs[e['stage']] += 1
        qs[e['stage']] += e['physical_q']
        hist[str(e['physical_q'])] += 1
    f, q = sum(fs.values()), sum(qs.values())
    return dict(request(s,policy), events=ev, generation_forwards=f, generation_query_rows=q,
                generation_physical_padded_rows=q, tpf=1/f,token_efficiency=1/q, t2t_edits=int(used and policy!=BASE),
                forwards_by_phase=dict(fs),q_by_phase=dict(qs),q_histogram=dict(hist))


@pytest.mark.parametrize('trigger,collision', [(False,False),(True,False),(True,True)])
def test_three_arms(trigger, collision):
    raws = [path(p,trigger=trigger,collision=collision) for p in a.ONCE]
    metrics, anchors = zip(*(a.audit_crowded_request(r,S,r['policy']) for r in raws))
    a.audit_crowded_match(raws,list(anchors))
    assert anchors == ((5,1) if trigger else None,)*3
    for j,c in enumerate(metrics):
        assert c['screen_overflow'] and c['screen_below_threshold'] and c['anchors']==int(trigger)
        assert c['candidate_collisions']==int(collision) and c['screen_no_slack']==0
        if trigger:
            assert c['anchor_admissions']==1 and c['screened']==3 and c['threshold_checks']==2
            assert c['victim_accepted']==int(j==0) and c['rank_direct_hits']==int(j==1)
            assert c['random_direct_hits']==int(j==2) and c['followup_mask_slot']==int(j!=0)
        else:
            assert c['screen_swap_old'] and c['swaps']==c['reserved_hits']==c['post_anchor_selected']==0


@pytest.mark.parametrize('policy,idx,key,value', [
    (BASE,2,'crowded_screen',None), (RANK,4,'intervention',None), (RAND,5,'crowded_screen',{}),
    (BASE,4,'random_candidate',-1), (BASE,4,'random',[]), (RANK,4,'risk_score',0.),
    (RAND,4,'victim_accepted',True), (BASE,4,'after_admissions',2), (BASE,4,'strict_rows',[0]),
    (RANK,4,'reserved_strict_edit_positions',[]), (RANK,4,'revoked_rows',[0]),
    (RANK,5,'reserved_extra_edit_positions',[33]), (BASE,5,'selected_masked_before',7+1),
    (BASE,5,'layer_physical_q',[8]*19), (BASE,5,'masked_before',3), (BASE,0,'intervention',{}),
    (BASE,4,'executed',[]), (BASE,4,'pre_sha256','x'*64), (BASE,4,'victim_queried',1),
])
def test_reject_corrupt_ledgers(policy,idx,key,value):
    raw = path(policy)
    e = raw['events'][idx]
    target = e if key in e or key in ('crowded_screen','intervention') else e['intervention']
    if value is None:
        del target[key]
    else:
        target[key] = value
    with pytest.raises((ValueError,KeyError)):
        a.audit_crowded_request(raw,S,policy)


def test_suffix_requires_accepted_victim_and_independent_noedit_order():
    raw = path(BASE)
    e = raw['events'][5]
    victim = raw['events'][4]['intervention']['victim']+32
    for dropped in [victim,min(e['absolute_positions'])]:
        bad = deepcopy(raw)
        x = bad['events'][5]
        x['absolute_positions'] = sorted(set(x['absolute_positions'])-{dropped}|{63})
        x.update(q_start=x['absolute_positions'][0], q_end=64)
        with pytest.raises(ValueError,match='post-anchor'):
            a.audit_crowded_request(bad,S,BASE)


@pytest.mark.parametrize('policy',[BASE,RANK,RAND])
def test_anchor_mask_witness(policy):
    raw = path(policy)
    e = raw['events'][4]
    p = e['intervention']
    rows = sorted(set(p['base']) & set(p['masks']))
    victim = p['victim']
    accepted = victim if policy==BASE else min(rows)
    ids = [77 if i==accepted else None for i in rows]
    e['beneficiary'] = dict(rows=rows,ids=ids,output_offsets=rows)
    a.audit_crowded_request(raw,S,policy)
    ids.reverse()
    with pytest.raises(ValueError,match='anchor MASK panel'):
        a.audit_crowded_request(raw,S,policy)


def test_partial_suffix_partitions_against_exhaustive_noedit():
    # N=10 crosses Q8, with five unresolved label pairs and five known rows.
    opts = ('MF','OF','OM','MF','MO','M','O','M','O','F')
    possible = {i:set(o) for i,o in enumerate(opts)}
    oracle = {}
    for tags in product(*opts):
        fresh,masks,old = ([i for i,s in enumerate(tags) if s==k] for k in 'FMO')
        f,m = len(fresh),len(masks)
        q = frozenset(fresh+(masks+old)[:8-f])
        union = oracle.setdefault((f,m,q), {i:set() for i in possible})
        for i,s in enumerate(tags):
            union[i].add(s)
    for (f,m,q),labels in oracle.items():
        e = dict(mandatory_fresh=f,masked_before=m,selected_masked_before=min(m,8-f))
        assert a.suffix_noedit(e,possible,q) == labels


@pytest.mark.parametrize('trigger,collision,key', [(False,False,'output_ids'),(True,True,'output_ids'),
    (True,False,'pre_sha256'), (True,False,'post_rng_sha256'), (True,False,'prefix'), (True,False,'screen')])
def test_exact_matched_history_and_collisions(trigger,collision,key):
    raws = [path(p,trigger=trigger,collision=collision) for p in a.ONCE]
    anchors = [a.audit_crowded_request(r,S,r['policy'])[1] for r in raws]
    r = raws[-1]
    if key=='output_ids':
        r[key].append(9)
    elif key=='prefix':
        r['events'][3]['trajectory_sha256'] = 'b'*64
    elif key=='screen':
        r['events'][4]['crowded_screen']['score'] += 1
    else:
        r['events'][4]['intervention'][key] = 'b'*64
    with pytest.raises(ValueError):
        a.audit_crowded_match(raws,anchors)


def test_complete_group_public_csv(tmp_path):
    dirs = [tmp_path/'pair',tmp_path/'random']
    samples = a.load(a.MANIFEST)['samples']
    for d,group in zip(dirs,[a.ONCE[:2],a.ONCE[2:]]):
        d.mkdir()
        write(d/'qualified.json',dict(model='llada21',full_width=32,matched=30))
        write(d/'run.json',dict(model='llada21',batch_size=1,cap=2048,block=32,seed_offset=0,
              context_weights_sha256=a.WEIGHTS, policies=group,
              arms=[dict(budget=8,fresh=0,mask=7,selection=p.split('m7_')[1]) for p in group]))
        for p in group:
            write(d/f'complete-{p}.json',dict(policy=p,requests=30))
            for s in samples:
                write(d/f"{s['index']:02d}-{p}.json",path(p,s,trigger=s['index']%2==0))
    out = tmp_path/'out'
    a.audit_crowded(dirs,out)
    for label,n in [('requests',90),('summary',3)]:
        f = out/f'context-crowded-{label}.csv'
        assert all(s not in f.read_bytes() for s in (b'\r', b'/data1', b'n164', b'output_ids', SHA.encode()))
        with f.open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows)==n and (label!='summary' or all(r['anchors']=='15' for r in rows))
    for runs in [dirs[:1],dirs+dirs]:
        with pytest.raises(ValueError):
            a.audit_crowded(runs,tmp_path/'bad')
    for p,d in zip(a.ONCE,[dirs[0],dirs[0],dirs[1]]):
        (d/f'00-{p}.json').unlink()
    with pytest.raises(ValueError):
        a.audit_crowded(dirs,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()

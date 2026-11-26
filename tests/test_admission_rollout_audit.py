import csv
from copy import deepcopy

import pytest

from bdlm_serving.methods.t2t import admission_gate as gate
from test_context_gates_audit import S, write
from test_crowded_audit import path
from tools import admission_rollout_audit as a


LOW, COIN = a.POLICIES
BASE, RANK, RAND = a.cg.ONCE


def offer(e):
    m = e['crowded_screen']
    assert m['reason'] == 'below_threshold'
    v = max(set(m['base']) & set(m['masks']))
    m.update(reason='swap_mask',victim=v,entrant=m['rank'],score=a.cg.CUTOFF)


def synthetic(policy, *, early=False, fire=True, s=S, decide=None):
    r = path(RANK,s,trigger=fire)
    r['policy'] = policy
    if early:
        offer(r['events'][3])
    for e in r['events']:
        m = e.get('crowded_screen',{})
        if m.get('reason') == 'swap_mask':
            keep,risk = (decide(m,None,policy.removeprefix(a.PREFIX),s['seed'],e['block'],e['hard_clock'])
                         if decide else ('intervention' in e,.5))
            e[a.DECISION] = dict(keep=keep,risk_probability=risk)
    return r


def replay(raw,fire):
    return a.audit_arm(raw,S,raw['policy'],None,lambda m,f,mode,s,b,t: (fire and t==5,.5))


@pytest.mark.parametrize('policy',[LOW,COIN])
@pytest.mark.parametrize('early,fire',[(True,True),(False,False),(True,False),(False,True)])
def test_first_approval_defer_and_physical_charges(policy,early,fire):
    r = synthetic(policy,early=early,fire=fire)
    c,clock,first = replay(r,fire)
    assert c['opportunities']==int(early)+int(fire) and c['skipped']==int(early)
    assert clock==(5 if fire else None) and first==(4 if early else 5 if fire else None)
    assert c['triggered']==c['reserved_hits']==c['followup_mask_slot']==int(fire)
    assert c['delayed_approved']==int(fire and early) and c['first_approved']==int(fire and not early)
    assert c['Q']==sum(c[s+'_Q'] for s in a.cg.STAGES if s!='prefill')==r['generation_query_rows']
    assert c['fresh_overflow_rows']==1 and (fire or c['screen_swap_old'])


@pytest.mark.parametrize('case',['wrong-risk','wrong-keep','non-bool','extra-key','absent-decision','defer-fired',
    'missed-fire','second-screen','decision-after','slack-decision','wrong-entrant','overflow','layers','phase-Q'])
def test_reject_bad_request(case):
    r = synthetic(COIN,early=True,fire=True)
    early, fire, after = r['events'][3:6]
    if case=='wrong-risk': early[a.DECISION]['risk_probability']=.501
    if case=='wrong-keep': early[a.DECISION]['keep']=True
    if case=='non-bool': early[a.DECISION]['keep']=0
    if case=='extra-key': early[a.DECISION]['unlogged']=1
    if case=='absent-decision': del early[a.DECISION]
    if case=='defer-fired': early['intervention']=fire['intervention']
    if case=='missed-fire': del fire['intervention']
    if case=='second-screen': after.update(crowded_screen=early['crowded_screen'],admission_decision=early[a.DECISION])
    if case=='decision-after': after[a.DECISION]=fire[a.DECISION]
    if case=='slack-decision':
        r = synthetic(COIN,fire=False)
        next(e for e in r['events'] if e.get('crowded_screen',{}).get('reason')=='swap_old')[a.DECISION] = early[a.DECISION]
    if case=='wrong-entrant': fire['intervention']['executed']=fire['intervention']['random']
    if case=='overflow': r['events'][2]['mandatory_fresh'] -= 1
    if case=='layers': fire['layer_physical_q'][0] -= 1
    if case=='phase-Q': r['q_by_phase']['selected'] -= 1
    with pytest.raises(ValueError):
        replay(r,case!='slack-decision')


def simple_ref(base):
    return dict(base, policy=a.BASELINE, events=[a.without(e,a.SIDE|a.DIGESTS) for e in base['events']])


@pytest.mark.parametrize('collision',[False,True])
def test_measured_first_match_and_corruption(collision):
    hist = [path(p,collision=collision) for p in a.cg.ONCE]
    first = a.audit_references(simple_ref(hist[0]),hist,S)
    raw = dict(deepcopy(hist[1]),policy=COIN)
    raw['events'][first-1][a.DECISION] = dict(keep=True,risk_probability=.5)
    _,anchor,f = replay(raw,True)
    a.compare_arm(raw,anchor,f,hist,first)
    for mutate in [lambda r:r['output_ids'].append(1),lambda r:r['events'][-2].update(accepted_sha256='b'*64),
                   lambda r:r['events'][first-1]['crowded_screen'].update(score=7.)]:
        bad=deepcopy(raw)
        mutate(bad)
        with pytest.raises(ValueError):
            a.compare_arm(bad,anchor,f,hist,first)


def test_compare_only_shared_paths_for_late_and_never_fire():
    # Compare a later fork against an earlier independently measured fork. On
    # the vetoed prefix the old base can have a panel/screen the new arm lacks.
    r = synthetic(LOW,early=True)
    hist = [path(p) for p in a.cg.ONCE]
    early = r['events'][3]
    for h in hist:
        h['events'][3]['crowded_screen']=early['crowded_screen']
        h['events'][4].pop('crowded_screen')
    for h in hist[1:]:
        h['events'][3]['absolute_positions'] = [32+i for i in sorted(set(early['crowded_screen']['base'])-
                 {early['crowded_screen']['victim']}|{early['crowded_screen']['entrant']})]
    a.compare_arm(r,5,4,hist,4)
    # Endpoint and anchor outcomes may diverge on the late path, not the common prefix.
    r['output_ids'].append(88888)
    a.compare_arm(r,5,4,hist,4)
    r['events'][3]['accepted_sha256']='b'*64
    with pytest.raises(ValueError):
        a.compare_arm(r,5,4,hist,4)
    never = deepcopy(hist[0]); never['policy']=LOW
    a.compare_arm(never,None,4,hist,4)
    never['output_ids'].append(1)
    with pytest.raises(ValueError):
        a.compare_arm(never,None,4,hist,4)


def fit_spec():
    return dict(schema=gate.SCHEMA,calibration_sha256=gate.CAL_SHA,ranker_sha256=gate.RANK_SHA,
                features=list(gate.FEATURES),weights=[0.]*len(gate.FEATURES),cutoff=.6,
                keep_probability=.999999999,fit_requests=24)


def test_complete_cli_inputs_fit_and_sanitized_csv(tmp_path):
    fit = tmp_path/'fit.json'; write(fit,fit_spec())
    model=gate.read_admission_fit(fit)
    def decision(m,_,mode,s,b,t):
        return gate.admission_decision(m,model,mode,s,b,t)
    def directory(name,group):
        d=tmp_path/name; d.mkdir()
        specs=[dict(budget=8,fresh=0,mask=8 if p==a.BASELINE else 7,selection=p.split('_',4)[4]) for p in group]
        meta=dict(model='llada21',batch_size=1,cap=2048,block=32,policies=group,arms=specs)
        if a.BASELINE not in group: meta.update(seed_offset=0,context_weights_sha256=a.cg.WEIGHTS)
        if group[0] in a.POLICIES: meta['admission_fit_sha256']=a.sha(fit)
        write(d/'run.json',meta)
        write(d/'qualified.json',dict(model='llada21',matched=30,full_width=32))
        for p in group: write(d/f'complete-{p}.json',dict(policy=p,requests=30))
        return d
    baseline=directory('base',[a.BASELINE]); history=directory('hist',a.cg.ONCE)
    runs=[directory(str(j),[p]) for j,p in enumerate(a.POLICIES)]
    for s in a.load(a.MANIFEST)['samples']:
        i=s['index']; fire=i%2==0
        hist=[path(p,s,trigger=fire) for p in a.cg.ONCE]
        for d,r in [(baseline,simple_ref(hist[0])),*((history,r) for r in hist),
                    *((d,synthetic(p,fire=fire,s=s,decide=decision)) for d,p in zip(runs,a.POLICIES))]:
            write(d/f"{i:02d}-{r['policy']}.json",r)
    def run(ds,out): return a.audit(ds,baseline,[history],fit,tmp_path/out)
    run(runs,'out')
    for kind,n in [('requests',60),('summary',2)]:
        f=tmp_path/'out'/f'admission-rollout-{kind}.csv'
        with f.open() as stream: rows=list(csv.DictReader(stream))
        assert len(rows)==n and all(k not in f.read_bytes() for k in (b'\r',b'/data1',b'n164',b'output_ids',b'88888'))
        if n==2: assert all(r['triggered']==r['reserved_hits']=='15' and r['requests']=='30' for r in rows)
    for bad in (runs[:1],runs+runs[:1]):
        with pytest.raises(ValueError): run(bad,'bad')
    meta=a.load(runs[0]/'run.json'); meta['admission_fit_sha256']='c'*64; write(runs[0]/'run.json',meta)
    with pytest.raises(ValueError,match='digest'): run(runs,'bad')
    meta['admission_fit_sha256']=a.sha(fit); write(runs[0]/'run.json',meta)
    write(runs[1]/'qualified.json',{})
    with pytest.raises(ValueError,match='unqualified'): run(runs,'bad')
    write(runs[1]/'qualified.json',dict(model='llada21',matched=30,full_width=32))
    for d,p in zip(runs,a.POLICIES): (d/f'00-{p}.json').unlink()
    with pytest.raises(ValueError,match='indices'): run(runs,'bad')
    assert not (tmp_path/'bad').exists()

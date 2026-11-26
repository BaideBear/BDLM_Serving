from fractions import Fraction
from itertools import combinations, product
from statistics import mean

import pytest

from tools import admission_forks as forks


def test_first_decision_uses_only_first_prestate_once():
    calls = []
    first = {'reason':'swap_mask', 'victim':3}
    later = {'reason':'swap_mask', 'victim':7}
    raw = {'seed':13, 'events':[{}, {'crowded_screen':first, 'intervention':{'victim_accepted':True}},
                                {'crowded_screen':later}]}
    def decide(move, fit, mode, seed, block, clock):
        calls.append((move, fit, mode, seed, block, clock))
        return move is later, .9
    assert forks.first_decision(raw,None,{},decide) == (False,None)
    assert forks.first_decision(raw,(2,1),{},decide) == (False,.9)
    assert calls == [(first,{},'hard_admit_low',13,1,2)]
    with pytest.raises(ValueError, match='decision'):
        forks.first_decision(raw,(2,1),{},lambda *args: (False,float('nan')))


def test_branches_and_exact_stratified_uniform_expectation():
    E, C = forks.Endpoint, forks.Case
    base = E(0,2,3,5)
    cases = [C(0,'A',base,E(1,3,4,7),True,True,.1,True,False),
             C(1,'A',base,E(0,7,4,11),True,False,.8,False,True),
             C(2,'A',base,base,False,False,None,False,False),
             C(3,'B',base,E(1,5,2,11),True,False,.9,True,True),
             C(4,'B',base,E(1,9,3,13),True,True,.2,False,False),
             C(5,'B',base,E(0,10,4,25),True,True,.3,True,True),
             C(6,'C',base,base,False,False,None,False,False)]
    rows, summary = forks.evaluate(cases)
    s = {(r['policy'],r['dataset']):r for r in summary}
    learned = s['hard_admit_low','all']
    assert learned['requests']==7 and learned['anchors']==5 and learned['approved']==3
    assert learned['N']==3+2+2+2+9+10+2 and learned['correct']==2
    assert learned['retained_direct_edits']==2 and learned['paid_admissions']==1
    assert [learned[k] for k in forks.CELLS.values()] == [1,0,1,1]
    assert [learned[k.replace('swaps_', 'vetoes_')] for k in forks.CELLS.values()] == [0,1,0,1]
    assert s['oracle_diagnostic','all']['approved']==1 and s['oracle_diagnostic','all']['paid_admissions']==0
    assert s['always_rank','all']['approved']==5 and s['base','all']['approved']==0
    # Enumerate every matched-count draw: one of two eligible A, two of three B, zero C.
    draws = []
    for a,b in product(combinations([0,1],1),combinations([3,4,5],2)):
        chosen = [c.rank if c.index in a+b else c.base for c in cases]
        draws.append(dict(N=sum(x.N for x in chosen), physical_Q=sum(x.Q for x in chosen), F=sum(x.F for x in chosen),
                          correct=sum(x.correct for x in chosen), macro_tpf=mean(Fraction(x.N,x.F) for x in chosen)))
    u = s['uniform_timing_expectation','all']
    for key in draws[0]:
        assert u[key] == mean(Fraction(d[key]) for d in draws)
    assert u['pooled_efficiency'] is None
    assert u['ratio_of_expected_N_to_expected_physical_Q']==u['N']/u['physical_Q']
    assert u['ratio_of_expected_N_to_expected_physical_Q']!=mean(Fraction(d['N'],d['physical_Q']) for d in draws)
    assert all(sum(r[k] for k in forks.CELLS.values()) == r['approved'] for r in rows)
    assert all(r['endpoint_source']=='base' and r['base_victim_admit'] is None for r in rows if not r['anchors'])


def test_fails_on_unmatched_noop_and_endpoint_corruption():
    e = forks.Endpoint(1,5,3,11)
    bad = forks.Case(1,'A',e,forks.Endpoint(1,5,4,11),False,False,None,False,False)
    with pytest.raises(ValueError,match='no-anchor'):
        forks.evaluate([bad])
    raw = dict(index=1,policy='p',model='llada21',dataset='math500',task_id='fixture',truncated=False,
               output_tokens=5,generation_forwards=3,generation_query_rows=11,generation_physical_padded_rows=11,max_q=8)
    ref = {**{k:str(raw[k]) for k in ('index','policy','model','dataset','task_id','max_q')},'N':'5','F':'3',
           'Q':'11','physical_Q':'11','correct':'1','failed':'0','truncated':'0'}
    assert forks.endpoint(raw,True,ref)==e
    with pytest.raises(ValueError, match='published endpoint'):
        forks.endpoint(raw,True,dict(ref,physical_Q='12'))

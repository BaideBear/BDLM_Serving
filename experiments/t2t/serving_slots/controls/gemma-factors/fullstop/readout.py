"""Read-only receipt audit; no model, prompt, completions or device access."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dense-stop', action='store_true')
parser.add_argument('--complete', action='store_true')
parser.add_argument('--pairs-csv', action='store_true')
parser.add_argument('--cohort', choices=('original30','new60','frozen90'), default='frozen90')
args=parser.parse_args()
if args.pairs_csv and (not args.complete or args.cohort!='frozen90'):
    parser.error('The per-request archive requires the full completed study')
dense = args.dense_stop
run = 'dense-stop-1128' if dense else 'fullstop-1128'
root = Path('/gs/results')/run
qual = {0,10,20}
target = {256:set(range(90))} if dense else {128:qual,256:set(range(90)),512:set(range(90))}
scope = {'original30':range(30),'new60':range(30,90),'frozen90':range(90)}[args.cohort]
target = {k:v.intersection(scope) for k,v in target.items() if v.intersection(scope)}
rows = {}
codes = {'partial':'P','drain':'D','verify':'V','absent':'A','rebuild':'R'}
source = None
pair_rows = []
manifest = json.loads((Path('/gs/runtime')/run/'repo/benchmarks/frozen90/manifest.json').read_text())
for path in root.glob('C*.json'):
    r = json.loads(path.read_text())
    c,i,on = r['budget'],r['index'],r['fullstop']
    if i not in scope:
        continue
    gate = 'half' if c==512 else 'off'
    assert path.name == (f'C256-dense-stop{int(on)}-{i:02d}.json' if dense
                         else f'C{c}-gate{gate}-fullstop{int(on)}-{i:02d}.json')
    assert i in target[c] and type(on) is bool
    assert r['schema']==('dense-fused-stop-v1' if dense else 'head-fullstop-v1')
    assert r['fuse'] is dense and not (r['reuse'] or r['slack'])
    assert r['rank']=='right' and r['prefix']==[1,8] and r['gate']==gate
    assert r['max_decoder_Q']<=c and (dense or not r['route_F'].get('verify_paid',0))
    if dense:
        assert r['full_head_at_b'] and c==256 and r['max_decoder_Q']==c and not r['joint_F']
        assert set(r['route_F'])<={'dense_single','verify','verify_paid','drain'}
        assert r['history_query_rows']==r['decoder_Q'] and (on or r['shadow_replay_sha256'])
    assert sum(r['route_F'].values())==r['decoder_F'] and r['F']==r['decoder_F']+r['encoder_F']
    assert r['samples_sha256']==manifest['samples_sha256']
    assert all(r[k]==manifest['samples'][i][k] for k in ('index','seed','dataset','task_id'))
    provenance = [r[k] for k in ('source_sha256','scorer_sha256','native_sha256','transformers','torch','checkpoint')]
    if source is None:
        source = provenance
    assert source==provenance
    s = r['observer']
    tape = s['head_tick_tape']
    assert len(tape)==r['decoder_F']==s['decoder_ticks']
    assert tape.count('F')==s['qualified_full_ticks'] and s['resets']=={k:tape.count(v) for k,v in codes.items()}
    assert sum(s['resets'].values())+tape.count('F')==len(tape)
    assert sum(a==b=='F' for a,b in zip(tape,tape[1:]))==s['consecutive_full_pairs']
    assert s['positive_ticks']==len(s['hits']) and s['first_hit']==next(iter(s['hits']),None)
    assert s['applied_hits']==(len(s['hits']) if on else 0)
    assert s['actionable_hits']==sum(h['otherwise_can_sample'] for h in s['hits'])
    assert tape.count('V')==r['route_F'].get('verify',0)+r['route_F'].get('verify_paid',0)
    assert all(tape[h['tick']-1:h['tick']+1]=='FF' and
               h['otherwise_can_sample']==bool(h['remaining_masks'] and h['age_after']<48) and
               h['applied']==on for h in s['hits'])
    if c==128:
        assert tape.count('F')==0
    assert (c,i,on) not in rows
    rows[c,i,on]=r

def ci(pairs, diffs):
    # Equal dataset weighting in both frozen cohorts. Preserve the observed
    # dataset sizes for provisional readouts as well, always resample whole pairs.
    strata = defaultdict(list)
    for (a,_),d in zip(pairs,diffs):
        strata[a['dataset']].append(d)
    rng = random.Random(1128)
    n=len(diffs)
    m=sorted(sum(sum(rng.choices(x,k=len(x))) for x in strata.values())/n for _ in range(3000))
    return [round(m[k],4) for k in (74,2924)]

for c,indices in target.items():
    pairs=[]
    for i in sorted(indices):
        if (c,i,False) not in rows or (c,i,True) not in rows:
            continue
        a,b=rows[c,i,False],rows[c,i,True]
        assert all(a[k]==b[k] for k in ('seed','input_sha256','gpu_visible','host','source_sha256','scorer_sha256'))
        first=lambda r: next((h for h in r['observer']['hits'] if h['otherwise_can_sample']),None)
        ah,bh=first(a),first(b)
        # Same inputs, sampler and gate all the way to the first effective intervention.
        assert bool(ah)==bool(bh) and (not ah or {**ah,'applied':True}==bh), (c,i,'first actionable mismatch')
        if not ah:
            assert all(a[k]==b[k] for k in ('F','N','output_ids','route_F','decoder_Q')), (c,i,'no-action drift')
        pairs.append((a,b))
        p=dict(budget=c,index=i,dataset=a['dataset'],seed=a['seed'],gate=a['gate'],gpu=a['gpu_visible'],
               ids_changed=a['output_ids']!=b['output_ids'],delta_F=b['F']-a['F'],
               delta_tpf=b['N']/b['F']-a['N']/a['F'])
        for label,r in (('off',a),('on',b)):
            s=r['observer']
            d={k:r.get(k) for k in ('F','N','decoder_Q','encoder_F','correct')}
            d.update({k:s[k] for k in ('decoder_ticks','qualified_full_ticks','consecutive_full_pairs',
                      'positive_ticks','actionable_hits','applied_hits')})
            d.update({f'{k}_resets':v for k,v in s['resets'].items()})
            d.update(first_hit_tick=(s['first_hit'] or {}).get('tick'),
                     first_actionable_tick=(first(r) or {}).get('tick'))
            p.update({f'{label}_{k}':v for k,v in d.items()})
        pair_rows.append(p)
    n=len(pairs)
    out=dict(tier=c,matched=n,expected=len(indices),sealed=[sum((c,i,h) in rows for i in indices) for h in (0,1)])
    if dense or args.cohort!='frozen90':
        out['cohort']=args.cohort
    if n:
        replay_n, replay_different = 0, []
        for a,_ in pairs:
            i=a['index']
            prior = (root.parent/'cross-race-1128'/f'C512-gatehalf-slack0-4-pref1of8-{i:02d}.json'
                     if c==512 else root.parent/'cross-race-1128'/f'C128-gateoff-slack0-4-pref1of8-{i:02d}.json'
                     if c==128 else root.parent/'paid-close-1128'/f'C256-fuse0-{i:02d}.json')
            if dense:
                prior=root.parent/'capacity-head-1128'/f'C256-capacity1-{i:02d}.json'
                assert prior.exists()
            if not prior.exists() and c==256:
                prior=root.parent/'prefix-rank-1128'/f'C256-rankright-reuse0-{i:02d}.json'
            if not prior.exists():
                continue
            hist=json.loads(prior.read_text())
            assert all(a[k]==hist[k] for k in ('index','seed','task_id','input_sha256','samples_sha256'))
            replay_n+=1
            if any(a[k]!=hist[k] for k in ('F','decoder_F','decoder_Q','encoder_F','encoder_Q','N','output_ids','route_F')):
                replay_different.append(i)
        out.update(prior_control_available=replay_n,prior_control_different=replay_different)
        for j,label in enumerate(('off','on')):
            arm=[p[j] for p in pairs]
            obs=[r['observer'] for r in arm]
            out[label]=dict(F=sum(r['F'] for r in arm),decoder_Q=sum(r['decoder_Q'] for r in arm),
                            N=sum(r['N'] for r in arm),correct=sum(r.get('correct') is True for r in arm),
                            scored=sum(type(r.get('correct')) is bool for r in arm),
                            macro_TPF=round(statistics.mean(r['N']/r['F'] for r in arm),4),
                            ticks=sum(s['decoder_ticks'] for s in obs),full=sum(s['qualified_full_ticks'] for s in obs),
                            consecutive=sum(s['consecutive_full_pairs'] for s in obs),
                            hits=sum(s['positive_ticks'] for s in obs),hit_cases=sum(bool(s['hits']) for s in obs),
                            actionable=sum(s['actionable_hits'] for s in obs),
                            actionable_cases=sum(bool(s['actionable_hits']) for s in obs),
                            resets={k:sum(s['resets'][k] for s in obs) for k in codes})
            if dense:
                out[label]['token_efficiency']=out[label]['N']/out[label]['decoder_Q']
        ds=[b['N']/b['F']-a['N']/a['F'] for a,b in pairs]
        fs=[b['F']-a['F'] for a,b in pairs]
        out.update(delta_macro_TPF=round(statistics.mean(ds),4), delta_TPF_ci95=ci(pairs,ds),
                   delta_F_ci95=ci(pairs,fs), F_down_same_up=[sum(f<0 for f in fs),fs.count(0),sum(f>0 for f in fs)],
                   ids_changed=sum(a['output_ids']!=b['output_ids'] for a,b in pairs),
                   score_flip=[[a['index'],a['correct'],b['correct']] for a,b in pairs
                               if 'correct' in a and 'correct' in b and a['correct']!=b['correct']])
    print(json.dumps(out),flush=True)
if args.complete:
    assert len(rows)==2*sum(map(len,target.values())) and all(type(r.get('correct')) is bool for r in rows.values()), 'incomplete'
    if args.pairs_csv:
        target_file=Path(__file__).with_name('dense-pairs.csv' if dense else 'pairs.csv')
        with target_file.with_suffix('.pending').open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=pair_rows[0],lineterminator="\n")
            w.writeheader()
            w.writerows(pair_rows)
        target_file.with_suffix('.pending').replace(target_file)

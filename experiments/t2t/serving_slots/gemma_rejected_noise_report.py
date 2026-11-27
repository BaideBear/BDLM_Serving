"""Read sealed pairs only. The compact CSV contains no prompts, tokens or generations."""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .gemma_rejected_noise_eval import QUAL, TIERS, check, pair, path, qualify


def interval(pairs, metric):
    groups=defaultdict(list)
    for a,b in pairs:
        groups[a['dataset']].append(metric(b)-metric(a))
    if len(groups)<3 or min(map(len,groups.values()))<2:
        return None
    rng=random.Random(1128)
    diffs=sorted(sum(sum(rng.choices(v,k=len(v))) for v in groups.values())/len(pairs)
                 for _ in range(4000))
    return [round(diffs[k],4) for k in (99,3899)]


def report(root, indices, complete):
    out,compact={},[]
    common=None
    for c in TIERS:
        pairs=[]
        missing={on:[] for on in (False,True)}
        for i in indices:
            for on in missing:
                if not path(root,c,i,on).is_file():
                    missing[on].append(i)
            if any(i in m for m in missing.values()):
                continue
            a,b=pair(root,c,i)
            signature=[a[k] for k in ('source_sha256','scorer_sha256','samples_sha256','checkpoint',
                                       'torch','transformers','policy')]
            if common is None:
                common=signature
            check(common==signature,'Mixed registration')
            pairs.append((a,b))
            row=dict(budget=c,index=i,dataset=a['dataset'],seed=a['seed'],gpu=a['gpu_visible'],
                     same_ids=a['output_ids']==b['output_ids'])
            for label,r in (('off',a),('on',b)):
                for k in ('N','F','decoder_F','decoder_Q','encoder_F','encoder_Q','correct'):
                    row[label+'_'+k]=r[k]
                for k,v in r['noise'].items():
                    if k!='first_exposure':
                        row[label+'_'+k]=v
                row[label+'_first_exposure_tick']=(r['noise']['first_exposure'] or {}).get('tick')
            compact.append(row)
        check(not complete or len(pairs)==len(indices), f'C{c} missing off={missing[False]} on={missing[True]}')
        tier=out[c]=dict(matched=len(pairs),expected=len(indices),missing_off=missing[False],missing_on=missing[True])
        if not pairs:
            continue
        tpf=lambda r:r['N']/r['F']
        for j,label in enumerate(('off','on')):
            arm=[p[j] for p in pairs]
            summary=tier[label]={k:sum(r[k] for r in arm) for k in
                ('N','F','decoder_F','decoder_Q','encoder_F','encoder_Q','correct')}
            summary.update({k:sum(r['noise'][k] for r in arm) for k in arm[0]['noise'] if k!='first_exposure'})
            summary.update(macro_TPF=round(mean(tpf(r) for r in arm),4),
                           exposed_cases=sum(bool(r['noise']['would_change']) for r in arm))
        tier['delta_macro_TPF']=round(mean(tpf(b)-tpf(a) for a,b in pairs),4)
        tier['TPF_CI95']=interval(pairs,tpf)
        tier['F_delta_CI95']=interval(pairs,lambda r:r['F'])
        tier['ids_changed']=sum(a['output_ids']!=b['output_ids'] for a,b in pairs)
        tier['rescues']=[a['index'] for a,b in pairs if b['correct'] and not a['correct']]
        tier['losses']=[a['index'] for a,b in pairs if a['correct'] and not b['correct']]
    return out,compact


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path)
    ap.add_argument('--history',type=Path,required=True)
    ap.add_argument('--qualification',action='store_true')
    ap.add_argument('--complete',action='store_true')
    ap.add_argument('--csv',type=Path)
    a=ap.parse_args()
    q=qualify(a.root,a.history)
    out,compact=report(a.root,QUAL if a.qualification else range(90),a.complete)
    if a.csv:
        check(a.complete,'Only export complete matched cohorts')
        tmp=a.csv.with_suffix('.tmp')
        with tmp.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=compact[0])
            writer.writeheader()
            writer.writerows(compact)
        tmp.replace(a.csv)
    print(json.dumps(dict(qualification=q,tiers=out),sort_keys=True))


if __name__=='__main__':
    main()

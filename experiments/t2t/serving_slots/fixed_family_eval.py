"""One persistent worker, independent fixed-budget requests; no within-request Q switching."""
import argparse
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
from itertools import groupby
import json
import os
from pathlib import Path

from bdlm_serving.evaluation import scoring
from bdlm_serving.evaluation.frozen import ROOT, sha
from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.methods.t2t.budget_family import Law
from bdlm_serving.runtime.diffusiongemma import check_snapshot, observe
from .gemma_budget_family import FamilyRequest
from .gemma_fixed_caps import DATA, SAMPLES_SHA
from .gemma_strict_close_pilot import CKPT
from .gemma_strict_eval import B, CAP, SCORER_SHA, encoder_tape, require, write
from .gemma_unified_audit import projectors
from .mask_pair_gemma import input_digest

TIERS = (32, 64, 128, 256)
ORDER = (0, 10, 20, *(i for i in range(90) if i not in (0, 10, 20)))
OUT = DATA/'gemma-fixed-family-v1'


def fixed(model, proc, task, q, law):
    import torch
    require(q in TIERS, 'Not a supported fixed tier')
    x = proc.apply_chat_template([{'role':'user','content':task['prompt']}], tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors='pt').to(model.device)
    prompt = x['input_ids'][0].tolist()
    origin = len(prompt)
    trace, h, tops = [], [], []
    with torch.inference_mode(), ExitStack() as ctx:
        ev = ctx.enter_context(observe(model))
        projections, pos = ctx.enter_context(projectors(model))
        enc = ctx.enter_context(encoder_tape(model))
        ctx.enter_context(model.lm_head.register_forward_pre_hook(lambda _,args: h.append(args[0].shape[1])))
        ctx.enter_context(model.register_forward_hook(lambda _,args,result:
            tops.append(result.logits[0].argmax(-1).tolist())))
        req = FamilyRequest(model,x,seed=task['seed'],max_new_tokens=CAP,law=law)
        while req.status == 'ready':
            require(len(trace) < (CAP//B)*(48*(B//q+1)+B//q+1),'Runaway request')
            trace.append(req.step(q))
    ids = req.tokens
    expected = [('encoder',origin)]
    widths = [e['q'] for e in trace]
    require(len(projections)==85 and all(w == h == widths for w in projections.values())
            and len(tops)==len(pos)==len(widths), 'Physical projections or LM head mismatch')
    require(enc[0] == (prompt,list(range(origin))), 'Prompt encoder mismatch')
    draft, stats = [], Counter()
    for ordinal, (b, it) in enumerate(groupby(trace, key=lambda e:e['ordinal'])):
        require(b == ordinal, 'Ordinal gap')
        sealed, seen, accepted = {}, set(), set()
        block = list(it)
        for e in block:
            rows, sampled, pref, dirty, add, edit, revoke = map(set, (e[k] for k in
                ('rows','sampled','prefix','dirty_before','admitted','edited','revoked')))
            w,t=e['q'],e['tick']
            require(e['offer']==q and 0 < len(rows)==w<=q and
                    pos[t]==[origin+B*b+i for i in sorted(rows)] and
                    pref <= sampled <= rows and add|edit|revoke <= sampled and edit|revoke <= accepted
                    and not add&accepted and not edit&revoke, 'Incorrect row selection or fixed offer')
            require(set(sorted(dirty)[:q]) <= rows, 'Unpaid mandatory debt')
            if pref:
                end=min(set(range(B))-accepted,default=B)
                require(pref <= (seen & set(range(end)))-dirty and
                        len(pref) <= min(q-len(rows & dirty),B)*law.num//law.den,
                        'Prefix outside current block or quota')
            if e['phase']=='verify':
                require(len(seen)==B and not dirty and not sealed.keys()&rows and not sampled,
                        'Invalid completion')
                sealed.update(zip(e['rows'],tops[t],strict=True))
            else:
                require(not sealed,'Sample/drain after verify')
            require(e['encoded']==(len(sealed)==B and t+1<len(trace)),'Invalid boundary encoder')
            expected.append(('decoder',w))
            if e['encoded']:
                expected.append(('encoder',B))
            accepted=(accepted|add)-revoke
            seen.update(sampled)
            stats.update(dirty=len(rows&dirty),prefix=len(pref),prefix_edits=len(pref&edit),
                         edited=len(edit),m2t=len(add),revoked=len(revoke))
        require(len(sealed)==B,'Incomplete verification')
        sealed=[sealed[i] for i in range(B)]
        if block[-1]['encoded']:
            require(enc[b+1]==(sealed,list(range(origin+b*B,origin+(b+1)*B))),
                    'Generation encoder did not read verified block')
        draft.extend(sealed)
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos,int) else set(eos or ())
    stop=next((i+1 for i,v in enumerate(draft) if v in eos),len(draft))
    require(draft[:stop]==ids and (ids[-1] in eos or len(ids)==CAP) and expected==ev,
            'Wrong EOS, output IDs, or measured forward order')
    n=sum(i not in set(proc.tokenizer.all_special_ids)|eos|{model.generation_config.pad_token_id} for i in ids)
    f,paid=len(ev)-1,sum(w for _,w in ev[1:])
    text=proc.tokenizer.decode(ids,skip_special_tokens=True)
    score=scoring.score(text,task)
    return dict(index=task['index'],dataset=task['dataset'],task_id=task['task_id'],seed=task['seed'],
                q=q,law=[law.num,law.den],input_sha256=input_digest(x),output_ids=ids,
                score=score,correct=bool(score['correct']),N=n,F=f,Q=paid,TPF=n/f,efficiency=n/paid,
                max_decoder_Q=max(widths),projection_modules=len(projections),decoder_Q=sum(widths),
                encoder_Q=B*(len(enc)-1),phase_F=dict(Counter(e['phase'] for e in trace)),
                row_visits=dict(stats),trace=trace)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    ap.add_argument('--rank',type=int,default=0)
    ap.add_argument('--shards',type=int,default=1)
    ap.add_argument('--tiers',type=int,nargs='+',choices=TIERS,default=TIERS)
    ap.add_argument('--num',type=int,choices=(0,1),default=1)
    ap.add_argument('--den',type=int,choices=(8,),default=8)
    a=ap.parse_args()
    require(0<=a.rank<a.shards<=90 and len(set(a.tiers))==len(a.tiers),'Invalid shard/tiers')
    if a.device=='cpu':
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and len(os.sched_getaffinity(0))==4,'Pin four CPU cores')
    else:
        require(os.environ.get('CUDA_VISIBLE_DEVICES') and ',' not in os.environ['CUDA_VISIBLE_DEVICES'],
                'Protected worker must expose one GPU')
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    import torch, transformers
    require(transformers.__version__=='5.12.0' and str(torch.__version__)=='2.9.0+cu128'
            and sha(DATA/'samples.json')==SAMPLES_SHA and sha(Path(scoring.__file__))==SCORER_SHA,
            'Different runtime/suite/scorer')
    os.umask(0o077)
    law=Law(a.num,a.den)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    tasks=load_tasks(DATA/'samples.json')
    require(len(tasks)==90,'Incomplete suite')
    root=OUT/a.device/f'law-{a.num}-{a.den}'
    root.mkdir(exist_ok=True,parents=True)
    meta=root/f'shard-{a.rank}-of-{a.shards}-tiers-{"-".join(map(str,a.tiers))}.json'
    require(not meta.exists(),'Shard already registered; inspect it before resuming')
    files=[Path(__file__),ROOT/'bdlm_serving/methods/t2t/budget_family.py',*(Path(__file__).with_name(s+'.py')
           for s in ('gemma_budget_family','gemma_strict_request','gemma_strict_rows','gemma_strict_sampler','gemma_strict_project','gemma_wave_kv'))]
    hashes={p.name:sha(p) for p in files}
    write(meta,dict(tiers=a.tiers,indices=ORDER[a.rank::a.shards],samples=SAMPLES_SHA,scoring=SCORER_SHA,
                    files=hashes,checkpoint=str(CKPT),device=a.device,torch=torch.__version__,transformers=transformers.__version__))
    print('Verifying weights',flush=True)
    check_snapshot(CKPT)
    proc=AutoProcessor.from_pretrained(CKPT,local_files_only=True)
    model=DiffusionGemmaForBlockDiffusion.from_pretrained(CKPT,local_files_only=True,dtype=torch.bfloat16,
            device_map={'':a.device},attn_implementation='sdpa',experts_implementation='grouped_mm').eval()
    for i in ORDER[a.rank::a.shards]:
        for q in a.tiers:
            dst=root/f'{i:02}-q{q}.json'
            require(not dst.exists() and all(sha(p)==hashes[p.name] for p in files),'Existing case or changed source')
            print(f'Running fixed Q{q} index={i}',flush=True)
            row=fixed(model,proc,tasks[i],q,law)
            row['registration_sha256']=sha(meta)
            write(dst,row)
            print(f'Sealed {i:02} Q{q}: correct={row["correct"]}, N/F={row["N"]}/{row["F"]}, paid={row["Q"]}',flush=True)


if __name__=='__main__':
    main()

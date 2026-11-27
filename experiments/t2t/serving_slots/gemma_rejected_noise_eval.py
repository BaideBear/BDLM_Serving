"""Single-factor low-tier GPU evaluation. Qualify against unfused paid-close controls."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

from .layout_sweep import write

QUAL = (0,10,20)
TIERS = (32,64,128)
ORDER = [j+k for j in range(10) for k in QUAL] + list(range(30,90))
CONTROL = ('index','dataset','task_id','seed','input_sha256','samples_sha256','scorer_sha256',
           'N','output_ids','correct','F','decoder_F','encoder_F','decoder_Q','encoder_Q',
           'route_F','route_Q','max_decoder_Q')


def check(test, message):
    if not test:
        raise ValueError(message)


def path(root,c,i,on):
    return root/f'C{c}-keep{int(on)}-{i:02d}.json'


def pair(root,c,i):
    a,b = [json.loads(path(root,c,i,on).read_text()) for on in (False,True)]
    for on,r in ((False,a),(True,b)):
        check(r['schema']=='reject-noise-v1' and r['index']==i and r['budget']==c and r['keep'] is on
              and type(r['correct']) is bool, 'Incomplete or wrong case')
        n=r['noise']
        check(n['avoided']==(n['would_change'] if on else 0) and
              n['token_changes']+n['avoided']==n['baseline_token_changes'] and
              0<=n['would_change']<=n['eligible']<=n['warm_rows']<=n['sampled_rows'] and
              (n['first_exposure'] is not None)==bool(n['would_change']) and
              (n['drain_F'],n['drain_Q'])==(r['route_F'].get('drain',0),r['route_Q'].get('drain',0)),
              'Unreconciled exposure or physical drain')
    check(all(a[k]==b[k] for k in (*CONTROL[:7],'source_sha256','gpu_visible','policy','checkpoint'))
          and a['noise']['first_exposure']==b['noise']['first_exposure'], 'Unmatched pair or first exposure')
    if not b['noise']['avoided']:
        check(all(a[k]==b[k] for k in CONTROL), 'Trajectories diverged without an intervention')
    return a,b


def qualify(root, history):
    report={}
    for c in TIERS:
        effective=0
        for i in QUAL:
            a,b=pair(root,c,i)
            hist=json.loads((history/f'C{c}-fuse0-{i:02d}.json').read_text())
            check(hist['schema']=='paid-close-v1' and hist['fuse'] is False
                  and hist['gate']=='off' and tuple(hist['prefix'])==(1,8) and not hist['slack']
                  and hist['budget']==c, 'Wrong historical control')
            diff=[k for k in CONTROL if a[k]!=hist[k]]
            check(not diff, f'Historical off mismatch C{c} #{i}: {diff}')
            effective+=b['noise']['avoided']
        check(effective>0, f'C{c}: no effective exposure in qualification')
        report[str(c)]=dict(matched=len(QUAL), avoided=effective)
    return report


def main():
    import torch
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from transformers.models.diffusion_gemma import generation_diffusion_gemma as native
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.evaluation.frozen import sha
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.methods.t2t.budget_family import Law
    from .gemma_fixed_dual_eval import audited
    from .gemma_rejected_noise import noise_receipt
    from .mask_pair_gemma import input_digest, SAMPLES_SHA
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('samples','checkpoint','output','history'):
        ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--deadline',type=int,required=True)
    ap.add_argument('--qualify-only',action='store_true')
    args=ap.parse_args()
    check(time.time()<args.deadline,'Past deadline')
    dev=os.environ.get('CUDA_VISIBLE_DEVICES')
    check(torch.cuda.device_count()==1 and dev and dev.startswith('GPU-') and ',' not in dev,'Bind one GPU UUID')
    check(transformers.__version__=='5.12.0' and sha(Path(native.__file__))==
          'cded57b7e365f0265b1c088e993c358b5f70acdf48055fce396540b4d9d691bc','Wrong pinned native source')
    tasks=load_tasks(args.samples)
    check(sha(args.samples)==SAMPLES_SHA,'Wrong task snapshot')
    package=Path(__file__).parent
    hashes={k:sha(package/(k+'.py')) for k in ('gemma_rejected_noise','gemma_rejected_noise_eval',
        'gemma_strict_sampler','gemma_fixed_dual','gemma_fixed_dual_eval','gemma_strict_rows',
        'gemma_strict_request','gemma_wave_kv','gemma_budget_family')}
    scorer=sha(Path(scoring.__file__))
    root=args.output
    root.mkdir(parents=True,exist_ok=True)
    proc=AutoProcessor.from_pretrained(args.checkpoint,local_files_only=True)
    model=DiffusionGemmaForBlockDiffusion.from_pretrained(args.checkpoint,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa',experts_implementation='grouped_mm',
        device_map={'':'cuda:0'}).eval()
    eos=model.generation_config.eos_token_id
    special=set(proc.tokenizer.all_special_ids)|set([eos] if isinstance(eos,int) else eos or [])
    policy=dict(top_gate='off',head_first=False,prefix_order='right',reuse_paid_masks=False,
                fuse_first_verify=False,fullstop=None)
    for j,i in enumerate(ORDER):
        if j==len(QUAL):
            q=qualify(root,args.history)
            write(root/'qualification.json',q)
            print('QUALIFIED',q,flush=True)
            if args.qualify_only:
                break
        for c in TIERS:
            with (root/f'C{c}-pair-{i:02d}.lock').open('a') as lock:
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                saved=[p for on in (False,True) if (p:=path(root,c,i,on)).exists()]
                old=[json.loads(p.read_text()) for p in saved]
                check(all(r['source_sha256']==hashes for r in old),'Cannot resume changed source')
                if len(old)==1 and old[0]['gpu_visible']!=dev:
                    continue  # The original device owns its unfinished pair.
                for on in ((False,True) if j%2==0 else (True,False)):
                    dst=path(root,c,i,on)
                    if time.time()>=args.deadline:
                        return
                    if dst.exists():
                        continue
                    print('START',dst.stem,flush=True)
                    t=tasks[i]
                    d=proc.apply_chat_template([{'role':'user','content':t['prompt']}], tokenize=True,
                        add_generation_prompt=True,return_dict=True,return_tensors='pt').to(model.device)
                    r=audited(model,d,q=c,seed=t['seed'],law=Law(1,8),ablation='00',
                              policy=dict(**policy,keep_rejected_noise=on))
                    check(r['projection_modules']==85 and r['max_decoder_Q']<=c and r['joint_F']==0,
                          'Wrong model, width or head-only scope')
                    receipt=noise_receipt(r['trace'],on)
                    ids=r['output_ids']
                    correct=bool(scoring.score(proc.tokenizer.decode(ids,skip_special_tokens=True),t)['correct'])
                    row={k:t[k] for k in ('index','dataset','task_id','seed')}
                    row.update({k:r[k] for k in ('F','decoder_F','encoder_F','decoder_Q','encoder_Q',
                               'route_F','route_Q','max_decoder_Q','output_ids')})
                    row.update(schema='reject-noise-v1',keep=on,budget=c,policy={**policy,'prefix':[1,8]},
                        N=sum(v not in special for v in ids),correct=correct,input_sha256=input_digest(d),
                        scorer_sha256=scorer,samples_sha256=SAMPLES_SHA,source_sha256=hashes,noise=receipt,
                        gpu_visible=dev,torch=str(torch.__version__),transformers=transformers.__version__,
                        checkpoint=str(args.checkpoint),run_pid=os.getpid())
                    write(dst,row)
                    print('DONE',dst.stem,row['F'],receipt['avoided'],flush=True)
                pair(root,c,i)


if __name__=='__main__':
    main()

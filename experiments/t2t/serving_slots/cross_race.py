"""Single-factor frozen90 Gemma screen. No scheduler, holders or GPU selection."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import socket
import time

from .layout_sweep import write


def grid():
    # Gate candidates share the same sampler; sequential control is a counterfactual,
    # not a genuine 512-row operation. Prefix dose and slack gate do not change together.
    for i in [j+k for j in range(10) for k in (0,10,20)] + list(range(30,90)):
        for gate in ('off','eager','half','quarter'):
            yield i, 512, gate, False, 4, (1,8)
        for c in (128,64):
            for dose in ((1,8),(0,8),(1,4),(1,2)):
                yield i, c, 'off', False, 4, dose
            for den in (4,2,8):
                yield i, c, 'off', True, den, (1,8)


def cases(suite):
    if suite == 'cross':
        for item in grid():
            yield (*item, 'right', False)
        return
    if suite == 'close':
        for i in [j+k for j in range(10) for k in (0,10,20)] + list(range(30,90)):
            for c in (256,128,32,64,512):
                yield i,c,'off',False,4,(1,8),'right',False
        return
    # One feature at a time. Include both missing endpoint tiers and a right-edge
    # replay in the same software build; comparisons never change the fraction.
    for i in [j+k for j in range(10) for k in (0,10,20)] + list(range(30,90)):
        for c in (32,256,64,128):
            for rank, reuse in (('right',False),('oldest',False),('uniform',False),('right',True)):
                yield i,c,'off',False,4,(1,8),rank,reuse


def fullstop_cases(*, lane=None, budget=None, arm=None, qualify=False):
    order = [j+k for j in range(10) for k in (0,10,20)] + list(range(30,90))
    for j,i in enumerate(order):
        if (lane is not None and j%2 != lane) or (qualify and j>=3):
            continue
        for c in ((512,256,128) if j<3 else (512,256)):
            if budget is not None and c!=budget:
                continue
            for on in ((False,True) if j%2 == 0 else (True,False)):
                if arm is None or on == (arm=='on'):
                    yield i,c,'half' if c==512 else 'off',False,4,(1,8),'right',False,on


def receipt(r, special):
    ids = r['output_ids']
    last, ever, costs = {}, set(), Counter()
    opened, discarded = set(), set()
    accepted, initialized = set(), set()
    exposure = Counter()
    for e in r['trace']:
        if e['opened'] is not None:
            opened.add(e['opened'])
        discarded.update(e['discarded'])
        for p in e['parts']:
            o = p['ordinal']
            tail = o > e['ordinal']
            dirty, pref = set(p['dirty_before']), set(p['prefix'])
            if pref:
                edge = next((j for j in range(256) if (o,j) not in accepted), 256)
                pool = [j for j in reversed(range(edge)) if (o,j) in initialized and j not in dirty]
                assert pref <= set(pool)
                exposure['prefix_seats'] += len(pref)
                exposure['differing_prefix_seats'] += len(pref-set(pool[:len(pref)]))
                exposure['prefix_distance_sum'] += sum(edge-j for j in pref)
                exposure['prefix_edits'] += len(pref.intersection(p['edited']))
                exposure['prefix_revoked'] += len(pref.intersection(p['revoked']))
            for j in p['rows']:
                k = (o,j)
                cat = (p['phase'] if p['phase'] in ('rebuild','verify') else
                       'mandatory' if j in dirty else 'prefix' if j in pref else
                       'unresolved' if k not in accepted else 'clean-other')
                costs[cat] += 1
            accepted.update((o,j) for j in p['admitted'])
            accepted.difference_update((o,j) for j in p['revoked'])
            initialized.update((o,j) for j in p['rows'])
            if tail:
                costs[f'tail-Q-block-{o}'] += len(p['rows'])
                for j,t in p.get('tail_updates', ()):
                    last[o,j] = t
                    ever.add((o,j))
                for j in p['revoked']:
                    last.pop((o,j), None)
    retained = sum(256*o+j < len(ids) and ids[256*o+j] == t and t not in special
                   for (o,j),t in last.items())
    cats = ('rebuild','verify','mandatory','prefix','unresolved','clean-other')
    assert sum(costs[k] for k in cats) == r['decoder_Q']
    return dict(categories={k:costs[k] for k in cats},
                prefix_exposure={k:exposure[k] for k in ('prefix_seats','differing_prefix_seats',
                    'prefix_distance_sum','prefix_edits','prefix_revoked')}, tail_open=len(opened),
                tail_discard=len(discarded), tail_touched_positions=len(ever),
                tail_last_candidate=len(last), tail_final_matches=retained,
                discarded_tail_Q=sum(costs[f'tail-Q-block-{o}'] for o in discarded),
                tail_Q=sum(v for k,v in costs.items() if k.startswith('tail-Q-block-')))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('samples','checkpoint','output'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--deadline', type=int, required=True)
    ap.add_argument('--suite', choices=('cross','rank','close','fullstop'), default='cross')
    ap.add_argument('--fuse-first-verify', action='store_true')
    ap.add_argument('--fullstop-lane', type=int, choices=(0,1), help='disjoint same-device pairs')
    ap.add_argument('--fullstop-budget', type=int, choices=(128,256,512))
    ap.add_argument('--fullstop-arm', choices=('off','on'), help='off is shadow-observed')
    ap.add_argument('--fullstop-qualify', action='store_true', help='one frozen item per dataset')
    a = ap.parse_args()
    if a.fuse_first_verify and a.suite != 'close':
        ap.error('Terminal fusion is an independent close-suite intervention')
    if a.suite != 'fullstop' and (a.fullstop_qualify or any(x is not None for x in
            (a.fullstop_lane,a.fullstop_budget,a.fullstop_arm))):
        ap.error('fullstop selectors require --suite fullstop')
    from bdlm_serving.evaluation.frozen import sha
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.evaluation.scoring import score
    from bdlm_serving.methods.t2t.budget_family import Law
    from .gemma_fixed_dual_eval import audited
    from .mask_pair_gemma import input_digest, SAMPLES_SHA
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    import transformers
    import torch
    assert torch.cuda.device_count() == 1 and sha(a.samples) == SAMPLES_SHA
    tasks = load_tasks(a.samples)
    a.output.mkdir(exist_ok=True, parents=True)
    proc = AutoProcessor.from_pretrained(a.checkpoint, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(a.checkpoint,local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa',experts_implementation='grouped_mm',
        device_map={'':'cuda:0'}).eval()
    eos = model.generation_config.eos_token_id
    special = set(proc.tokenizer.all_special_ids) | set([eos] if isinstance(eos,int) else eos or [])
    package = Path(__file__).resolve().parent
    sources = {p:sha(package/p) for p in ('cross_race.py','gemma_fixed_dual.py','gemma_fixed_dual_eval.py')}
    scorer_sha = sha(package.parents[2]/'bdlm_serving/evaluation/scoring.py')
    if a.suite == 'fullstop':
        from transformers.models.diffusion_gemma import generation_diffusion_gemma as upstream
        native_sha = sha(Path(upstream.__file__))
        if (transformers.__version__ != '5.12.0' or
                native_sha != 'cded57b7e365f0265b1c088e993c358b5f70acdf48055fce396540b4d9d691bc'):
            raise RuntimeError('Unregistered native Gemma stopping implementation')
        sources.update({p:sha(package/p) for p in ('gemma_strict_sampler.py','gemma_strict_rows.py',
                        'gemma_fullstop.py','gemma_strict_request.py','gemma_budget_family.py','gemma_wave_kv.py')})
        work = fullstop_cases(lane=a.fullstop_lane,budget=a.fullstop_budget,arm=a.fullstop_arm,
                              qualify=a.fullstop_qualify)
    else:
        work = ((*case,None) for case in cases(a.suite))
    for i,c,gate,slack,den,dose,rank,reuse,on in work:
        if time.time() >= a.deadline:
            return
        key = f"C{c}-gate{gate}-slack{int(slack)}-{den}-pref{dose[0]}of{dose[1]}-{i:02d}"
        if a.suite == 'rank':
            key = f'C{c}-rank{rank}-reuse{int(reuse)}-{i:02d}'
        if a.suite == 'close':
            key = f'C{c}-fuse{int(a.fuse_first_verify)}-{i:02d}'
        if on is not None:
            key = f'C{c}-gate{gate}-fullstop{int(on)}-{i:02d}'
        out = a.output / (key+'.json')
        if out.exists():
            continue
        with out.with_suffix('.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if out.exists():
                continue
            print('START', key, flush=True)
            t = tasks[i]
            d = proc.apply_chat_template([{'role':'user','content':t['prompt']}], tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors='pt').to(model.device)
            r = audited(model,d,seed=t['seed'],q=c,ablation='10' if slack else '00',law=Law(*dose),
                policy=dict(top_gate=gate,head_first=slack,late_den=den,receipt_tail_ids=True,
                            prefix_order=rank,reuse_paid_masks=reuse,
                            fuse_first_verify=a.fuse_first_verify, fullstop=on))
            assert r['projection_modules']==85 and r['max_decoder_Q']<=c
            ids = r['output_ids']
            n = sum(tid not in special for tid in ids)
            text = proc.tokenizer.decode(ids,skip_special_tokens=True)
            row = {k:t[k] for k in ('index','dataset','task_id','seed')}
            row.update({k:r[k] for k in ('F','decoder_F','encoder_F','decoder_Q','encoder_Q',
                                         'max_decoder_Q','joint_F','route_F','route_Q','output_ids')})
            row.update(receipt(r,special), schema='single-factor-cross-v1', budget=c, **{
                'gate':gate,'slack':slack,'late_den':den,'prefix':dose}, N=n, history_query_rows=r['decoder_Q'],
                truncated=len(ids)==2048 and ids[-1] not in special, output_text=text, samples_sha256=SAMPLES_SHA,
                input_sha256=input_digest(d), gpu=torch.cuda.get_device_name(), host=socket.gethostname(),
                torch=str(torch.__version__), run_pid=os.getpid())
            if a.suite == 'rank':
                row.update(schema='prefix-rank-v1', rank=rank, reuse=reuse,
                           source_sha256=sources, scorer_sha256=scorer_sha,
                           transformers=str(transformers.__version__),
                           resample_paid_F=r['route_F'].get('resample_paid',0))
                if rank == 'right':
                    assert row['prefix_exposure']['differing_prefix_seats'] == 0
            if a.suite == 'close':
                row.update(schema='paid-close-v1', fuse=a.fuse_first_verify,
                           source_sha256=sources, scorer_sha256=scorer_sha,
                           transformers=str(transformers.__version__),
                           fused_F=r['route_F'].get('verify_paid',0),
                           fused_dirty_Q=sum(len(p['dirty_before']) for e in r['trace']
                                if e['route']=='verify_paid' for p in e['parts']))
            if on is not None:
                row.update(schema='head-fullstop-v1', fullstop=on, fuse=False, rank=rank, reuse=reuse,
                           observer=r['fullstop_receipt'], source_sha256=sources, scorer_sha256=scorer_sha,
                           transformers=str(transformers.__version__), native_sha256=native_sha,
                           checkpoint=str(a.checkpoint), gpu_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
                           stability_threshold=1, confidence_threshold=.005)
            try:
                row['correct'] = bool(score(text,t)['correct'])
            except (FileNotFoundError, RuntimeError) as err:
                row['score_pending'] = str(err)
            write(out,row)
            print('DONE', key, row.get('correct'), n, r['F'], flush=True)


if __name__ == '__main__':
    main()

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


def receipt(r, special):
    ids = r['output_ids']
    last, ever, costs = {}, set(), Counter()
    opened, discarded = set(), set()
    accepted = set()
    for e in r['trace']:
        if e['opened'] is not None:
            opened.add(e['opened'])
        discarded.update(e['discarded'])
        for p in e['parts']:
            o = p['ordinal']
            tail = o > e['ordinal']
            dirty, pref = set(p['dirty_before']), set(p['prefix'])
            for j in p['rows']:
                k = (o,j)
                cat = (p['phase'] if p['phase'] in ('rebuild','verify') else
                       'mandatory' if j in dirty else 'prefix' if j in pref else
                       'unresolved' if k not in accepted else 'clean-other')
                costs[cat] += 1
            accepted.update((o,j) for j in p['admitted'])
            accepted.difference_update((o,j) for j in p['revoked'])
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
    return dict(categories={k:costs[k] for k in cats}, tail_open=len(opened),
                tail_discard=len(discarded), tail_touched_positions=len(ever),
                tail_last_candidate=len(last), tail_final_matches=retained,
                discarded_tail_Q=sum(costs[f'tail-Q-block-{o}'] for o in discarded),
                tail_Q=sum(v for k,v in costs.items() if k.startswith('tail-Q-block-')))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('samples','checkpoint','output'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--deadline', type=int, required=True)
    a = ap.parse_args()
    from bdlm_serving.evaluation.frozen import sha
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.evaluation.scoring import score
    from bdlm_serving.methods.t2t.budget_family import Law
    from .gemma_fixed_dual_eval import audited
    from .mask_pair_gemma import input_digest, SAMPLES_SHA
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
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
    for i,c,gate,slack,den,dose in grid():
        if time.time() >= a.deadline:
            return
        key = f"C{c}-gate{gate}-slack{int(slack)}-{den}-pref{dose[0]}of{dose[1]}-{i:02d}"
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
                policy=dict(top_gate=gate,head_first=slack,late_den=den,receipt_tail_ids=True))
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
            try:
                row['correct'] = bool(score(text,t)['correct'])
            except (FileNotFoundError, RuntimeError) as err:
                row['score_pending'] = str(err)
            write(out,row)
            print('DONE', key, row.get('correct'), n, r['F'], flush=True)


if __name__ == '__main__':
    main()

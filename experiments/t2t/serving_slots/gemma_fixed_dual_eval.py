"""Loaded-worker adapter: fixed(model, proc, task, q, *, ablation='00')."""
from collections import Counter
from contextlib import ExitStack

import torch

from bdlm_serving.methods.t2t.budget_family import Law
from bdlm_serving.runtime.diffusiongemma import observe
from experiments.t2t.serving_slots.gemma_strict_eval import B, CAP, encoder_tape, require
from experiments.t2t.serving_slots.gemma_unified_audit import projectors
from experiments.t2t.serving_slots.mask_pair_gemma import input_digest

from .gemma_fixed_dual import ABLATIONS, TIERS, AblationRequest, FixedFamilyRequest


@torch.inference_mode()
def audited(model, data, *, seed, max_new_tokens=CAP, law=Law(), q=512, ablation=None,
            policy=None):
    # None preserves the pre-ablation fixture / archived Q512 registration.
    if type(q) is not int or q not in TIERS or (ablation is None and q != 512) or (
            ablation is not None and ablation not in ABLATIONS):
        raise ValueError('Invalid Gemma audit configuration')
    prompt = data['input_ids'][0].tolist()
    origin = len(prompt)
    trace, heads, top = [], [], []
    with ExitStack() as stack:
        events = stack.enter_context(observe(model))
        proj, pos = stack.enter_context(projectors(model))
        enc = stack.enter_context(encoder_tape(model))
        stack.enter_context(model.lm_head.register_forward_pre_hook(lambda _, a: heads.append(a[0].shape[1])))
        stack.enter_context(model.register_forward_hook(lambda _, a, out: top.append(out.logits[0].argmax(-1).tolist())))
        cls = FixedFamilyRequest if ablation is None else AblationRequest
        opts = {} if ablation is None else {'ablation': ablation, **(policy or {})}
        if policy and ablation is None:
            raise ValueError('Opening policy requires the ablation executor')
        req = cls(model, data, q=q, seed=seed, max_new_tokens=max_new_tokens, law=law, **opts)
        while req.status == 'ready':
            require(len(trace) < (max_new_tokens // B) * (2*48*(B//min(q,B)+2)+2*B//min(q,B)+4),
                    'No bounded decoder progress')
            trace.append(req.step())
    widths = [e['q'] for e in trace]
    require(heads == widths and proj and all(t == heads for t in proj.values()) and
            pos == [e['positions'] for e in trace] and len(top) == len(trace), 'Physical decoder mismatch')
    ledger, snapshots = [('encoder', origin)], [(prompt, list(range(origin)))]
    # Observed position lists, in physical forward order. The prompt is visible
    # in this tape but excluded from registered post-prefill N/F and N/Q.
    physical = [dict(kind='encoder', phase='prefill', tick=None, paid=False, positions=enc[0][1])]
    sealed, partial, debt, accepted, ages = [], [], set(), set(), {}
    epoch_rows = set()
    phases, visits, route_F, route_Q = Counter(), Counter(), Counter(), Counter()
    for t, (e, argmax) in enumerate(zip(trace, top, strict=True)):
        positions = []
        verified = None
        route = e['route']
        build = route == 'build_sample'
        limit = B if build else q
        require(e['tick'] == t and e['offer'] == q and 0 < widths[t] <= limit and
                len(argmax) == widths[t] and e['ordinal'] == len(sealed), 'Bad decoder receipt')
        if ablation is not None:
            require(e['dirty_before_absolute'] == sorted(debt) and
                    set(sorted(debt)[:q]) <= set(pos[t]), 'Missing global refresh')
            if route == 'resample_paid':
                base = origin+B*len(sealed)
                require(req.reuse_paid_masks and q<2*B and e['head_initialized_before']==B
                        and epoch_rows.issuperset(range(base,base+B))
                        and set(pos[t])==debt and len(e['parts'])==1 and
                        len(e['sampled_absolute'])>0 and
                        set(e['sampled_absolute'])==debt-accepted,
                        'Paid MASK sampling without full warm head / captured physical write')
            if q == 2*B:
                require(not build and route != 'sparse', 'Top tier took a low-tier policy')
                if route in ('dense_joint','dense_single','dense_head_debt'):
                    start = origin+B*len(sealed)
                    count = 2*B if route=='dense_joint' else B
                    require(pos[t] == sorted(set(range(start,start+count)) | debt),
                            'Not a dense top-tier call')
                    if route=='dense_head_debt':
                        require(all(i<start+B for i in e['sampled_absolute']), 'Sampled a parked tail')
            if q < 2*B and ablation == '00':
                require(all(p['ordinal']==e['ordinal'] for p in e['parts']), 'Tail on fixed 00')
            if e['opened'] is not None and q < 2*B:
                base = origin+B*len(sealed)
                require(e['head_initialized_before']==B and epoch_rows.issuperset(range(base,base+B)),
                        'Late opening before full head KV in this encoder epoch')
                require(e['opened']==len(sealed)+1 and e['head_mask_before']<=B//req.late_den and
                        ablation in ('10','11') and build==(ablation=='11'), 'Wrong low-tier opening')
            if build:
                require(not debt and len(e['parts'])==1 and e['q']==B and
                        e['parts'][0]['ordinal']==e['opened'] and
                        e['parts'][0]['sampled']==list(range(B)) and
                        e['parts'][0]['age_before']==0, 'Unpaid or unsampled whole-canvas build')
        ledger.append(('decoder', widths[t]))
        epoch_rows.update(pos[t])
        physical.append(dict(kind='decoder', phase=route, tick=t, paid=True, positions=pos[t]))
        e['encoder_positions'] = []
        route_F[route] += 1
        route_Q[route] += widths[t]
        phases['+'.join(p['phase'] for p in e['parts'])] += 1
        for p in e['parts']:
            o, rows = p['ordinal'], p['rows']
            base = origin + B * o
            require(o in (len(sealed),len(sealed)+1) and rows == sorted(set(rows)) and
                    0 <= rows[0] <= rows[-1] < B, 'Bad block part')
            start = len(positions)
            positions.extend(base + i for i in rows)
            absrows = {base+i for i in rows}
            sampled, pref, adds, edits, revokes, changed = (
                {base+i for i in p[k]} for k in ('sampled','prefix','admitted','edited','revoked','input_changed'))
            domain = range(base,base+B)
            old_dirty = debt.intersection(domain)
            require(old_dirty=={base+i for i in p['dirty_before']} and
                    pref <= sampled <= absrows and (adds|edits|revokes|changed) <= sampled and
                    not adds&accepted and (edits|revokes) <= accepted and not edits&revokes and
                    p['age_before']==ages.get(o,0) and
                    p['age_after']==p['age_before']+bool(sampled), 'Unmeasured row or sampler clock')
            if pref:
                edge = next((a for a in domain if a not in accepted),base+B)
                require(pref <= accepted-old_dirty and max(pref)<edge,'Prefix crosses a block frontier')
            accepted = (accepted|adds)-revokes
            debt = (debt-absrows)|changed|adds|revokes
            require(debt.intersection(domain)=={base+i for i in p['dirty_after']}, 'Lost token/acceptance debt')
            ages[o] = p['age_after']
            visits.update(prefix=len(p['prefix']), edited=len(p['edited']),
                          revoked=len(p['revoked']), m2t=len(p['admitted']),
                          prefix_edits=len(set(p['prefix']) & set(p['edited'])),
                          dirty=len(old_dirty & absrows), rebuild=len(rows) if p['phase']=='rebuild' else 0)
            if p['phase'] == 'verify':
                require(o == len(sealed) and rows == list(range(len(partial),len(partial)+len(rows))) and not p['sampled']
                        and not p['dirty_before'] and verified is None, 'Invalid head verify')
                partial.extend(argmax[start:start+len(rows)])
                if len(partial)==B:
                    verified, partial = partial, []
        require(positions == sorted(set(positions)) == pos[t], 'Wrong block offsets')
        if ablation is not None:
            require(debt==set(e['dirty_after_absolute']) and
                    debt==(set(e['dirty_before_absolute'])-set(pos[t])) | set(e['written_absolute']),
                    'Absolute debt does not reconcile')
        require(e['encoded'] == (verified is not None and t < len(trace)-1), 'Wrong encoder flag')
        if verified is not None:
            base = origin + B * len(sealed)
            sealed.append(verified)
            if e['encoded']:
                epoch_rows.clear()
                ledger.append(('encoder', B))
                e['encoder_positions'] = enc[len(snapshots)][1]
                physical.append(dict(kind='encoder', phase='encoder_commit', tick=t, paid=True,
                                     positions=e['encoder_positions']))
                snapshots.append((verified, list(range(base, base+B))))
    draft = [v for block in sealed for v in block]
    end = next((i+1 for i,v in enumerate(draft) if v in req.eos), len(draft))
    require(not partial and not debt and events == ledger and enc == snapshots and req.tokens == draft[:end] and
            (len(req.tokens)==max_new_tokens or req.tokens[-1] in req.eos), 'Commit/EOS/billing mismatch')
    require([(e['kind'],len(e['positions'])) for e in physical]==events, 'Incomplete physical tape')
    paid = physical[1:]
    encoder = [e for e in paid if e['kind']=='encoder']
    return dict(output_ids=req.tokens, decoder_F=len(widths), decoder_Q=sum(widths),
                encoder_F=len(encoder), encoder_Q=sum(len(e['positions']) for e in encoder),
                F=len(paid), Q=sum(len(e['positions']) for e in paid),
                max_decoder_Q=max(widths), joint_F=sum(len(e['parts'])==2 for e in trace),
                route_F=dict(route_F), route_Q=dict(route_Q), over_offer_F=sum(w>q for w in widths),
                prefill_Q=origin, projection_modules=len(proj), phase_F=dict(phases),row_visits=dict(visits),
                trace=trace, physical_trace=physical)


def _scored(model, proc, task, q, ablation):
    from experiments.t2t.serving_slots import fixed_family_eval as registered
    data = proc.apply_chat_template([{'role':'user','content':task['prompt']}], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors='pt').to(model.device)
    law = Law()
    row = audited(model, data, seed=task['seed'], law=law, q=q, ablation=ablation)
    require(row['projection_modules']==85, 'Not the pinned physical Gemma model')
    ids, tok = row['output_ids'], proc.tokenizer
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos,int) else set(eos or ())
    special = set(tok.all_special_ids) | eos | {model.generation_config.pad_token_id}
    n = sum(i not in special for i in ids)
    score = registered.scoring.score(tok.decode(ids, skip_special_tokens=True),task)
    return dict(**row, **{k:task[k] for k in ('index','dataset','task_id','seed')}, q=q,
        law=[law.num,law.den], input_sha256=input_digest(data), correct=bool(score['correct']),
        score=score, N=n, TPF=n/row['F'], efficiency=n/row['Q'],
        **({'schema':'fixed-dual-v1'} if ablation is None else
           {'schema':'gemma-fixed-ablation-v1','ablation':ablation}))


def fixed(model, proc, task, q, *, ablation='00'):
    if type(q) is not int or q not in TIERS or ablation not in ABLATIONS:
        raise ValueError('Expected Q32/64/128/256/512 and ablation 00/10/11')
    return _scored(model, proc, task, q, ablation)


def main():
    """Append a disjoint CPU Q512 registration. Never change the active Q<=256 evaluator."""
    import os
    from pathlib import Path
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from . import fixed_family_eval as ev
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and not torch.cuda.is_available() and
            os.sched_getaffinity(0)==set(range(32,36)), "Expected the registered Gemma CPU lane")
    require(transformers.__version__=="5.12.0" and str(torch.__version__)=="2.9.0+cu128" and
            ev.sha(ev.DATA/"samples.json")==ev.SAMPLES_SHA and ev.sha(Path(ev.scoring.__file__))==ev.SCORER_SHA,
            "Runtime, suite or scorer changed")
    os.umask(0o077)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    tasks=ev.load_tasks(ev.DATA/"samples.json")
    root=ev.OUT/"cpu/law-1-8"
    root.mkdir(parents=True,exist_ok=True)
    meta=root/"shard-0-of-1-tiers-512.json"
    require(not meta.exists() and not list(root.glob("*-q512.json")),"Q512 already registered")
    files=[Path(__file__), Path(ev.__file__), ev.ROOT/"bdlm_serving/methods/t2t/budget_family.py",
           *(Path(__file__).with_name(k+".py") for k in ("gemma_fixed_dual","gemma_budget_family",
             "gemma_strict_request","gemma_strict_rows","gemma_strict_sampler","gemma_strict_project","gemma_wave_kv"))]
    hashes={p.name:ev.sha(p) for p in files}
    ev.write(meta,dict(indices=ev.ORDER,tiers=[512],law=[1,8],samples=ev.SAMPLES_SHA,scoring=ev.SCORER_SHA,
             checkpoint=str(ev.CKPT),files=hashes,device="cpu",torch=torch.__version__,transformers=transformers.__version__))
    print("Verifying weights for fixed Q512",flush=True)
    ev.check_snapshot(ev.CKPT)
    proc=AutoProcessor.from_pretrained(ev.CKPT,local_files_only=True)
    model=DiffusionGemmaForBlockDiffusion.from_pretrained(ev.CKPT,local_files_only=True,dtype=torch.bfloat16,
              device_map={"":"cpu"},attn_implementation="sdpa",experts_implementation="grouped_mm").eval()
    for i in ev.ORDER:
        dst=root/f"{i:02}-q512.json"
        require(not dst.exists() and all(ev.sha(p)==hashes[p.name] for p in files),"Duplicate or changed source")
        print(f"Running fixed Q512 index={i}",flush=True)
        r=_scored(model,proc,tasks[i],512,None)
        r["registration_sha256"]=ev.sha(meta)
        ev.write(dst,r)
        print(f"Sealed {i:02} Q512: correct={r['correct']}, N/F={r['N']}/{r['F']}, paid={r['Q']}, dual calls={r['joint_F']}",flush=True)


if __name__=="__main__":
    main()

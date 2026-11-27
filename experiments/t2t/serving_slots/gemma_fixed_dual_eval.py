"""Loaded-worker adapter: fixed(model, proc, task, q, law), no loader or writes."""
from collections import Counter
from contextlib import ExitStack

import torch

from bdlm_serving.methods.t2t.budget_family import Law
from bdlm_serving.runtime.diffusiongemma import observe
from experiments.t2t.serving_slots.gemma_strict_eval import B, CAP, encoder_tape, require
from experiments.t2t.serving_slots.gemma_unified_audit import projectors
from experiments.t2t.serving_slots.mask_pair_gemma import input_digest

from .gemma_fixed_dual import FixedFamilyRequest


@torch.inference_mode()
def audited(model, data, *, seed, max_new_tokens=CAP, law=Law()):
    prompt = data['input_ids'][0].tolist()
    origin = len(prompt)
    trace, heads, top = [], [], []
    with ExitStack() as stack:
        events = stack.enter_context(observe(model))
        proj, pos = stack.enter_context(projectors(model))
        enc = stack.enter_context(encoder_tape(model))
        stack.enter_context(model.lm_head.register_forward_pre_hook(lambda _, a: heads.append(a[0].shape[1])))
        stack.enter_context(model.register_forward_hook(lambda _, a, out: top.append(out.logits[0].argmax(-1).tolist())))
        req = FixedFamilyRequest(model, data, q=512, seed=seed, max_new_tokens=max_new_tokens, law=law)
        while req.status == 'ready':
            require(len(trace) < (max_new_tokens // B) * 200, 'No bounded decoder progress')
            trace.append(req.step())
    widths = [e['q'] for e in trace]
    require(heads == widths and proj and all(t == heads for t in proj.values()) and
            pos == [e['positions'] for e in trace] and len(top) == len(trace), 'Physical decoder mismatch')
    ledger, snapshots = [('encoder', origin)], [(prompt, list(range(origin)))]
    sealed, phases, visits = [], Counter(), Counter()
    for t, (e, argmax) in enumerate(zip(trace, top, strict=True)):
        positions = []
        verified = None
        require(e['tick'] == t and e['offer'] == 512 and 0 < widths[t] <= 512 and
                len(argmax) == widths[t] and e['ordinal'] == len(sealed), 'Bad decoder receipt')
        ledger.append(('decoder', widths[t]))
        phases['+'.join(p['phase'] for p in e['parts'])] += 1  # One category per D.
        for p in e['parts']:
            o, rows = p['ordinal'], p['rows']
            base = origin + B * o
            require(o in (len(sealed),len(sealed)+1) and rows == sorted(set(rows)) and
                    0 <= rows[0] <= rows[-1] < B and set(p['dirty_before']) <= set(rows), 'Bad block part')
            start = len(positions)
            positions.extend(base + i for i in rows)
            visits.update(prefix=len(p['prefix']), edited=len(p['edited']),
                          revoked=len(p['revoked']), m2t=len(p['admitted']),
                          prefix_edits=len(set(p['prefix']) & set(p['edited'])),
                          dirty=len(p['dirty_before']), rebuild=len(rows) if p['phase']=='rebuild' else 0)
            if p['phase'] == 'verify':
                require(o == len(sealed) and rows == list(range(B)) and not p['sampled']
                        and not p['dirty_before'] and verified is None, 'Invalid head verify')
                verified = argmax[start:start+B]
        require(positions == sorted(set(positions)) == pos[t], 'Wrong block offsets')
        require(e['encoded'] == (verified is not None and t < len(trace)-1), 'Wrong encoder flag')
        if verified is not None:
            base = origin + B * len(sealed)
            sealed.append(verified)
            if e['encoded']:
                ledger.append(('encoder', B))
                snapshots.append((verified, list(range(base, base+B))))
    draft = [v for block in sealed for v in block]
    end = next((i+1 for i,v in enumerate(draft) if v in req.eos), len(draft))
    require(events == ledger and enc == snapshots and req.tokens == draft[:end] and
            (len(req.tokens)==max_new_tokens or req.tokens[-1] in req.eos), 'Commit/EOS/billing mismatch')
    df, dq, ef = len(widths), sum(widths), len(enc)-1
    return dict(output_ids=req.tokens, decoder_F=df, decoder_Q=dq, encoder_F=ef, encoder_Q=B*ef,
                F=df+ef, Q=dq+B*ef, max_decoder_Q=max(widths), joint_F=sum(len(e['parts'])==2 for e in trace),
                prefill_Q=origin, projection_modules=len(proj), phase_F=dict(phases),row_visits=dict(visits),trace=trace)


def fixed(model, proc, task, q, law=Law()):
    from experiments.t2t.serving_slots import fixed_family_eval as registered
    if q != 512:
        return registered.fixed(model, proc, task, q, law)
    data = proc.apply_chat_template([{'role':'user','content':task['prompt']}], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors='pt').to(model.device)
    row = audited(model, data, seed=task['seed'], law=law)
    require(row['projection_modules']==85 and row['joint_F'] and row['max_decoder_Q']==512,
            'Not the pinned physical Q512 model/run')
    ids, tok = row['output_ids'], proc.tokenizer
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos,int) else set(eos or ())
    special = set(tok.all_special_ids) | eos | {model.generation_config.pad_token_id}
    n = sum(i not in special for i in ids)
    score = registered.scoring.score(tok.decode(ids, skip_special_tokens=True),task)
    return dict(**row, **{k:task[k] for k in ('index','dataset','task_id','seed')}, q=512,
        law=[law.num,law.den], input_sha256=input_digest(data), correct=bool(score['correct']),
        score=score, N=n, TPF=n/row['F'], efficiency=n/row['Q'], schema='fixed-dual-v1')


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
        r=fixed(model,proc,tasks[i],512)
        r["registration_sha256"]=ev.sha(meta)
        ev.write(dst,r)
        print(f"Sealed {i:02} Q512: correct={r['correct']}, N/F={r['N']}/{r['F']}, paid={r['Q']}, dual calls={r['joint_F']}",flush=True)


if __name__=="__main__":
    main()

"""Frozen CPU Q128 K32: three new KV-only requests, six read-only archived controls."""
import argparse
from collections import Counter
from contextlib import ExitStack
from importlib.metadata import version
import os
from pathlib import Path
from unittest.mock import patch

from bdlm_serving.evaluation import scoring
from bdlm_serving.evaluation.frozen import ROOT, sha
from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.runtime.diffusiongemma import STUDY, check_snapshot, observe
from .gemma_fixed_caps import DATA, SAMPLES_SHA
from .gemma_strict_close_pilot import BASE, CKPT, FIELDS, HERE, HISTORY, REV, read
from .gemma_strict_eval import B, CAP, CASES, SCORER_SHA, audit, encoder_tape, require, write
from .gemma_unified_audit import projectors
from .mask_pair_gemma import generation_settings, input_digest

Q, K = 128, 32
PINS = {0: HISTORY, K: {
    0: ('c10f738d65c9202863e97eeb8f6b423c879d05dcddaac5bde35592f0e4e61b66',
        'aa1f7c41362752f677e6cff5fcf91111d11ac23075a4bae4b866ebceff1ddd44'),
    10: ('52924dff64d04fdc464725a35a841b2e68178bc768908c778610d567b8c49e00',
         '0c89492fd632f983192821b7636db031a09fe158aaea0a0132d8ac35082f2e39'),
    20: ('0130b15aa8d2b59c7f533ab8958add6bfdcd85fffa394a1e7596f57a90e91adc',
         '90263d8d475ca0faf3f61b4eb2e59d59a9f281150a1c7635afff0e31b7e0690f')}}
OUT = DATA/'gemma-strict-prefix-action-pilot-v1'
ARCHIVE = DATA/'gemma-strict-eval-v1'
FULL = set(range(B))


def control(task, k, native):
    i = task['index']
    tag = f'cpu-q{Q}-k{k}-{i:02}'
    cp, raw = ARCHIVE/'compact'/(tag+'.json'), ARCHIVE/'raw'/tag
    ep, rp = raw/'evidence.json', raw/'run.json'
    require((sha(cp), sha(ep)) == PINS[k][i], f'Changed control {tag}')
    c, e, r = read(cp), read(ep), read(rp)
    require(sha(rp) == c['run_sha256'] and r['native'] == native
            and r['checkpoint'] == str(CKPT) and c['revision'] == REV
            and (r['index'], r['case'], r['seed']) == (i, task['task_id'], task['seed'])
            and all(task[key] == c[key] for key in ('index', 'dataset', 'task_id', 'seed'))
            and c['prefix_mode'] == 'right' and c['cap'] == CAP and r['affinity'] == list(range(32,36))
            and (r['torch'], r['transformers']) == ('2.9.0+cu128', '5.12.0'), tag)
    for d in (r, c):
        require((d['q'], d['k'], d['device'], d['samples_sha256'], d['scorer_sha256'])
                == (Q, k, 'cpu', SAMPLES_SHA, SCORER_SHA), tag)
    require(all(sha(HERE/(n+'.py')) == digest for n, digest in r['sources'].items()),
            f'Archived strict protocol changed: {tag}')
    return c, e


def paired_controls(controls):
    c0, e0 = controls[0]
    c1, e1 = controls[K]
    a, b = e0['trace'], e1['trace']
    first = next((i for i, e in enumerate(b) if e['prefix']), None)
    require(c0['input_sha256'] == c1['input_sha256'] and
            e0['encoder_inputs'][0] == e1['encoder_inputs'][0] and
            e0['generation_config'] == e1['generation_config'] and first is not None and
            first < len(a) and a[:first] == b[:first] and
            e0['decoder_position_ids'][:first] == e1['decoder_position_ids'][:first],
            'Controls unmatched before first prefix')
    x, y = a[first], b[first]
    require(all(x[k] == y[k] for k in ('tick','ordinal','phase','dirty_before','offer'))
            and x['phase'] == 'sample' and not x['prefix']
            and set(x['rows']) == set(y['rows']) - set(y['prefix'])
            and set(x['sampled']) == set(y['sampled']) - set(y['prefix']), 'Control fork geometry')
    return first


def preflight():
    import torch
    from transformers import AutoProcessor, __version__ as transformers_version
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.sched_getaffinity(0) == set(range(36,40)),
            'Require hidden GPUs and taskset -c 36-39')
    require(ROOT == BASE/'runtime/t2t-exchange-suite-1127' and
            DATA.resolve() == BASE/'runtime/t2t-native-1125/prefix90', 'Wrong checkout or data root')
    require(CASES == (0,10,20) and (B,CAP) == (256,2048), 'Pilot registration changed')
    require(torch.__version__ == '2.9.0+cu128' and transformers_version == '5.12.0'
            and version('math-verify') == '0.8.0' and sha(DATA/'samples.json') == SAMPLES_SHA
            and sha(Path(scoring.__file__)) == SCORER_SHA and
            sha(ROOT/'benchmarks/checkpoints/diffusiongemma.json') ==
            'ceb951734eb354728d7ba984093739119addbeda981b554af46d5dc105071f53', 'Frozen runtime')
    native = read(STUDY/'config.json')
    require((native['revision'], native['canvas'], native['cap'], native['dtype'], native['attention'],
             native['experts']) == (REV, B, CAP, 'bfloat16', 'sdpa', 'grouped_mm'), 'Native config')
    proc = AutoProcessor.from_pretrained(CKPT, local_files_only=True)
    cases = []
    tasks = load_tasks(DATA/'samples.json')
    for i in CASES:
        task = tasks[i]
        data = proc.apply_chat_template([{'role':'user','content':task['prompt']}], tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors='pt')
        controls = {k: control(task, k, native) for k in (0, K)}
        first = paired_controls(controls)
        for k, (c, e) in controls.items():
            cfg = e['generation_config']
            require(input_digest(data) == c['input_sha256'] and c['layers'] == 30 and
                    len(e['projection_widths']) == c['projection_modules'] == 85, 'Control inputs/layers')
            m = audit(data['input_ids'][0].tolist(), e['output_ids'], e['trace'],
                      list(map(tuple,e['events'])), e['decoder_position_ids'], e['projection_widths'],
                      list(map(tuple,e['encoder_inputs'])), q=Q, k=k, eos=set(cfg['eos_token_id']),
                      special=set(proc.tokenizer.all_special_ids), pad=cfg['pad_token_id'])
            require(all(c[key] == v for key, v in m.items()) and c['correct'] == e['score']['correct']
                    and c['metric'] == e['score']['metric'], 'Archived physical evidence/score mismatch')
        check_control_selection(controls[K][1]['trace'])
        cases.append((task, data, controls, first))
    return native, proc, cases


def selection(seen, dirty, accepted, age, cursor):
    """Independent B256 control law. Only warm, clean, contiguous prefix can use slack."""
    closing = age == 48 or accepted == FULL
    debt = sorted(dirty)
    if closing and not debt:
        require(seen == FULL and cursor < B, 'Incomplete terminal sweep')
        return 'verify', list(range(cursor, min(cursor+Q, B))), [], []
    if closing or len(debt) >= Q:
        return 'drain', debt[:Q], [], []
    if seen != FULL:
        cold = sorted(FULL-seen)[:Q-len(debt)]
        return 'sample', sorted(dirty.union(cold)), cold, []
    masks = FULL-accepted
    ordinary = sorted(dirty.union(sorted(masks-dirty)[:Q-len(debt)]))
    pool = sorted(set(range(min(masks))) - dirty)
    n = min(K, Q-len(ordinary), len(pool))
    prefix = pool[len(pool)-n:]
    return 'sample', sorted(ordinary+prefix), ordinary, prefix


def check_control_selection(trace):
    """Validate the independent policy on every tick of the archived editable arm."""
    seen, accepted, age, cursor = set(), set(), 0, 0
    for e in trace:
        phase, rows, ordinary, prefix = selection(seen, set(e['dirty_before']), accepted, age, cursor)
        require((phase, rows, prefix, sorted(ordinary+prefix)) ==
                (e['phase'], e['rows'], e['prefix'], e['sampled']), 'Control selection replay')
        seen.update(e['sampled'])
        accepted = (accepted | set(e['admitted'])) - set(e['revoked'])
        age += phase == 'sample'
        cursor += len(rows) if phase == 'verify' else 0
        if e['encoded']:
            seen, accepted, age, cursor = set(), set(), 0, 0


def run_case(cls, model, proc, case):
    import torch
    task, data, controls, first = case
    editable = controls[K][1]
    cfg = generation_settings(model.generation_config)
    require(cfg == editable['generation_config'], 'Live generation configuration changed')
    eos = set(cfg['eos_token_id'])
    prompt = data['input_ids'][0].tolist()
    trace, head, sampled_tape, sealed, emitted = [], [], [], [], []
    seen, dirty = set(), set()
    fphase, qphase, changes = Counter(), Counter(), Counter()
    block = 0
    with torch.inference_mode(), ExitStack() as stack:
        events = stack.enter_context(observe(model))
        proj, pos = stack.enter_context(projectors(model))
        enc = stack.enter_context(encoder_tape(model))
        logits = []
        stack.enter_context(model.lm_head.register_forward_pre_hook(lambda _, a: head.append(a[0].shape[1])))
        stack.enter_context(model.register_forward_hook(lambda _, a, out: logits.append(out.logits)))
        req = cls(model, data, seed=task['seed'], prefix_quota=K, prefix_mode='right',
                  prefix_action='kv_only', max_new_tokens=CAP)
        require(events == [('encoder', len(prompt))] and enc == [(prompt,list(range(len(prompt))))],
                'Expected one real encoder prefill')
        while req.status == 'ready':
            tick = len(trace)
            require(tick < (CAP//B)*(3*48+4) and req.ordinal == block, 'Clock/block mismatch')
            s = req.sampler
            before = s.ids[0].tolist()
            accepted = set(s.accepted[0].nonzero().flatten().tolist())
            age, rng = s.age, s.rng.get_state()
            phase, rows, sampled, pref = selection(seen, dirty, accepted, age, len(sealed))
            keep_sc = {r: s.sc[r].clone() for r in pref}
            calls = []
            old_consume = s.consume

            def decoder_input(_, args, kw):
                require(kw['decoder_input_ids'].tolist() == [[before[r] for r in rows]],
                        'Projected token IDs do not match the frozen pre-step inputs')
                if pref:
                    sc = kw['self_conditioning_logits']
                    require(sc is not None and all(torch.equal(sc[0,rows.index(r)], keep_sc[r]) for r in pref),
                            'Prefix projection did not receive its prior self-conditioning')

            def consume(rr, measured):
                rr = list(rr)
                require(rr == sampled and len(logits) == 1, 'Sampler saw nonordinary rows')
                raw = logits[0]
                require(measured.shape == (1, len(rr), raw.shape[-1]) and
                        (measured is raw and rr == rows or all(torch.equal(measured[:,j], raw[:,rows.index(r)])
                            for j,r in enumerate(rr))), 'Sampler logits not gathered from projected rows')
                calls.append(rr)
                return old_consume(rr, measured)

            start = len(events)
            with model.register_forward_pre_hook(decoder_input, with_kwargs=True), patch.object(s, 'consume', consume):
                e = req.step(Q)
            require((e['tick'],e['ordinal'],e['offer'],e['phase'],e['q'],e['rows'],
                     e['sampled'],e['prefix'],e['dirty_before']) ==
                    (tick,block,Q,phase,len(rows),rows,sampled,pref,sorted(dirty)) and
                    type(e['encoded']) is bool and 0 < len(rows) <= Q and
                    calls == ([sampled] if sampled else []), f'Row/consume contract at {tick}')
            width = len(rows)
            require(len(head) == len(pos) == tick+1 and head[-1] == width and len(proj) == 85 and
                    all(len(ws) == tick+1 and ws[-1] == width for ws in proj.values()) and
                    pos[-1] == [len(prompt)+B*block+r for r in rows] and len(logits) == 1
                    and tuple(logits[0].shape) == (1,width,model.config.text_config.vocab_size)
                    and events[start:] == [('decoder',width)]+([('encoder',B)] if e['encoded'] else []),
                    f'Physical work mismatch at {tick}')
            if tick <= first:
                y = editable['trace'][tick]
                keys = FIELDS if tick < first else ('tick','ordinal','offer','phase','dirty_before','rows','q','prefix')
                require(all(e[key] == y[key] for key in keys) and pos[-1] == editable['decoder_position_ids'][tick],
                        f'Divergence before matched action at tick {tick}')
            now = s.ids[0].tolist()
            acc = set(s.accepted[0].nonzero().flatten().tolist())
            changed = {r for r in FULL if before[r] != now[r]}
            require(changed | (accepted ^ acc) <= set(sampled) and s.age == age+bool(sampled) and
                    (sampled or torch.equal(rng, s.rng.get_state())) and
                    all(torch.equal(s.sc[r], v) for r,v in keep_sc.items()) and
                    e['admitted'] == sorted(acc-accepted) and e['revoked'] == sorted(accepted-acc) and
                    e['edited'] == sorted(acc & accepted & changed), f'State mutated off sample at {tick}')
            if phase == 'verify':
                sealed.extend(logits[0][0].argmax(-1).tolist())
            logits.clear()
            seen.update(sampled)
            dirty = (dirty-set(rows)) | changed | (accepted ^ acc)
            more = False
            if len(sealed) == B:
                end = next((j+1 for j,t in enumerate(sealed) if t in eos), None)
                emitted.extend(sealed[:end])
                more = end is None and len(emitted) < CAP
                if more:
                    base = len(prompt)+block*B
                    require(enc[-1] == (sealed,list(range(base,base+B))), 'Charged encoder != verified IDs')
                require(req.status == ('ready' if more else 'done'), 'Premature stop or extra block')
                block += 1
                seen, dirty, sealed = set(), set(), []
            else:
                require(req.status == 'ready', 'Unverified output')
            require(req.tokens == emitted and e['encoded'] == more, 'EOS/cap or encoder accounting')
            trace.append(e)
            sampled_tape.append(calls)
            qphase[phase] += width
            fphase[phase] += 1
            for key in ('admitted','edited','revoked','prefix','sampled'):
                changes[key] += len(e[key])
    require(len(trace) > first and block == len(enc) and 0 < len(emitted) <= CAP, 'Incomplete pair')
    n = sum(t not in eos | set(proc.tokenizer.all_special_ids) | {cfg['pad_token_id']} for t in emitted)
    text = proc.tokenizer.decode(emitted, skip_special_tokens=True)
    score = scoring.score(text, task)
    f, paid = len(events)-1, sum(w for _, w in events[1:])
    m = dict(N=n,F=f,Q=paid,N_per_F=n/f,N_per_Q=n/paid,correct=bool(score['correct']),metric=score['metric'],
             decoder_F=len(trace),decoder_Q=sum(head),encoder_F=len(enc)-1,encoder_Q=B*(len(enc)-1),
             phase_F=dict(fphase),phase_Q=dict(qphase), prefix_calls=sum(bool(e['prefix']) for e in trace),
             prefix_rows=changes['prefix'],prefix_edited=0,prefix_revoked=0,sampled_rows=changes['sampled'],
             edited=changes['edited'],revoked=changes['revoked'],admitted=changes['admitted'],
             truncated=emitted[-1] not in eos)
    require(f == m['encoder_F']+m['decoder_F'] and paid == m['encoder_Q']+m['decoder_Q'], 'F/Q accounting')
    raw = dict(metrics=m,output_ids=emitted,output_text=text,score=score,trace=trace,events=events,
               encoder_inputs=enc,decoder_position_ids=pos,head_widths=head,consume_calls=sampled_tape,
               projection_widths={f'{i}.{name}':v for (i,name),v in proj.items()})
    return m, raw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--check-controls', action='store_true', help='Read archives only; no treatment import')
    group.add_argument('--run', action='store_true', help='Explicitly run three new CPU KV-only requests')
    args = ap.parse_args()
    if args.run:
        require(not os.path.lexists(OUT), 'Never resume or overwrite an experiment directory')
    native, proc, cases = preflight()
    print('Six pinned controls passed; first prefix ticks:', [c[3] for c in cases], flush=True)
    if args.check_controls:
        return
    from .gemma_prefix_action import PrefixActionRequest
    if not args.run:
        print('Treatment import OK; no model loaded or output created.')
        return
    import torch
    from transformers import DiffusionGemmaForBlockDiffusion
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    os.umask(0o077)
    OUT.mkdir(mode=0o700, exist_ok=False)
    names = set(read(ARCHIVE/'raw/cpu-q128-k0-00/run.json')['sources'])
    names.update((Path(__file__).stem, 'gemma_prefix_action', 'gemma_strict_close_pilot',
                  'mask_pair_gemma', 'gemma_fixed_caps'))
    sources = {n+'.py':sha(HERE/(n+'.py')) for n in sorted(names)}
    write(OUT/'registration.json', dict(cases=CASES,q=Q,k=K,prefix_mode='right',prefix_action='kv_only',
         cap=CAP,device='cpu',affinity=sorted(os.sched_getaffinity(0)),control_affinity=[32,33,34,35],
         checkpoint=str(CKPT),native=native,archives=str(ARCHIVE),control_pins=PINS,sources=sources,
         samples_sha256=SAMPLES_SHA,scorer_sha256=SCORER_SHA,torch=str(torch.__version__)))
    check_snapshot(CKPT)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(CKPT,local_files_only=True,dtype=torch.bfloat16,
                device_map={'':'cpu'},attn_implementation='sdpa',experts_implementation='grouped_mm').eval()
    require(all(p.device.type=='cpu' for p in model.parameters()) and model.config.canvas_length == B
            and len(model.model.decoder.layers)==30, 'Not the frozen CPU model')
    pairs = []
    for case in cases:
        task,data,controls,first = case
        i = task['index']
        m,raw = run_case(PrefixActionRequest,model,proc,case)
        keys = ('N','F','Q','N_per_F','N_per_Q','correct','encoder_F','encoder_Q','decoder_F','decoder_Q',
                'prefix_rows','prefix_edited','prefix_revoked','edited','revoked')
        arms = {name:{key:v[key] for key in keys} for name,v in
                [('k0',controls[0][0]),('editable',controls[K][0]),('kv_only',m)]}
        identity = dict(index=i,seed=task['seed'],task_id=task['task_id'],dataset=task['dataset'],
                        input_sha256=input_digest(data))
        write(OUT/f'{i:02}-kv-only.json',dict(**identity,**raw))
        pair = dict(**identity,first_prefix_tick=first,arms=arms,
                    kv_only_minus={a:{k:m[k]-v[k] for k in keys} for a,v in arms.items() if a!='kv_only'},
                    same_output_ids={a:raw['output_ids']==controls[k][1]['output_ids']
                                     for a,k in [('k0',0),('editable',K)]})
        pairs.append(pair)
        print(i, pair['kv_only_minus'], flush=True)
    totals = {a:{k:sum(p['arms'][a][k] for p in pairs) for k in
                 ('N','F','Q','correct','encoder_Q','decoder_Q','prefix_rows','prefix_edited','prefix_revoked')}
              for a in ('k0','editable','kv_only')}
    for v in totals.values():
        v.update(pooled_N_per_F=v['N']/v['F'],pooled_N_per_Q=v['N']/v['Q'])
    require(all(sha(HERE/n)==h for n,h in sources.items()), 'Sources changed during run')
    write(OUT/'summary.json',dict(pairs=pairs,totals=totals,
         limitation='Same geometry through first action only; sampler support, RNG and later paths may differ. '
                    'Three paired CPU tasks are not a quality or latency benchmark.'))
    artifacts = ['registration.json','summary.json',*(f'{i:02}-kv-only.json' for i in CASES)]
    write(OUT/'sealed.json',dict(complete=True, generated_requests=3,reused_controls=6,
                                sha256={n:sha(OUT/n) for n in artifacts}))
    print('Sealed:',OUT/'sealed.json',flush=True)


if __name__ == '__main__':
    main()

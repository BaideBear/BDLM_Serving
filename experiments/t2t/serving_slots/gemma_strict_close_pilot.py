"""Paired real-CPU qualification of dirty-first close, not a speed or quality benchmark.

Default is read-only preflight. --run requires hidden GPUs and CPU affinity 36-39.
Evidence stays on n164. Only summary.json marks a complete three-pair run.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path

from bdlm_serving.evaluation import scoring
from bdlm_serving.evaluation.frozen import ROOT, sha
from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.runtime.diffusiongemma import STUDY, check_snapshot, observe
from .gemma_fixed_caps import DATA, SAMPLES_SHA
from .gemma_strict_eval import B, CAP, CASES, SCORER_SHA, encoder_tape, require, write
from .gemma_unified_audit import projectors
from .mask_pair_gemma import add_tensor, generation_settings, input_digest

Q = 128
FULL = set(range(B))
HERE = Path(__file__).resolve().parent
BASE = Path('/data1/project_data/gx-erqw-20261123')
CKPT = BASE/'models/diffusiongemma-cd40195'
OUTPUT = DATA/'gemma-strict-close-pilot-v1'
REV = 'cd40195554d3cfcb56f34b9adfb66efc790ed5b6'
# Pins are for the actual CPU K0 compact and raw evidence, not a pooled control.
HISTORY = {
    0: ('eb94d8f7e8fc6d5d76a04a13e494a98b1acf7067e8fe92f6969c3b29994311c5',
        '1537242b478072e79ae9e1fdfbda8477653103b0071e5073fbab0ad3fa55edaf'),
    10: ('88833ecff026abd519d649cb8290397ad564011af9e9cecbf3e1ff1020835fcb',
         '582fbd38a41473d5bf0dc06e87e0651648cd70f9255abf9dc0d30c1c4cebe344'),
    20: ('d78ddc6ea9f369fd4c6fb9b27c4e27fdccc922dee2f2e6ac486b604287344af1',
         '6f86d94cc83147afd179ea6da7e168048e81dcd474c0f10cba1d6376f5567c59'),
}
FIELDS = ('tick', 'ordinal', 'phase', 'offer', 'q', 'rows', 'sampled', 'prefix',
          'dirty_before', 'admitted', 'edited', 'revoked', 'encoded')
LIMITATION = ('Dirty-first verify seals each row once. Other resident KV rows, and deeper-layer '
              'KV even for unchanged token IDs, can differ when chunks refresh in a different '
              'order. Argmax, EOS, later blocks, charged encoder work and scores may change. '
              'Three fixed cases are not frozen90 quality estimates; F and Q are not latency.')


def read(p):
    return json.loads(p.read_bytes())


def archive(task, native):
    i = task['index']
    root = DATA/'gemma-strict-eval-v1'
    tag = f'cpu-q128-k0-{i:02}'
    cp, raw = root/'compact'/(tag+'.json'), root/'raw'/tag
    ep, rp = raw/'evidence.json', raw/'run.json'
    require((sha(cp), sha(ep)) == HISTORY[i], f'Historical CPU evidence changed: {i}')
    c, e = read(cp), read(ep)
    require(sha(rp) == c['run_sha256'], f'Historical metadata changed: {i}')
    r = read(rp)
    require(all(task[k] == c[k] for k in ('index','seed','task_id','dataset')) and
            r['native'] == native and r['checkpoint'] == str(CKPT) and r['device'] == 'cpu' and
            (r['q'], r['k'], r['seed']) == (Q, 0, task['seed']), 'Historical identity mismatch')
    for name, digest in r['sources'].items():
        require(sha(HERE/(name+'.py')) == digest, f'Control path changed: {name}')
    return c, e


def preflight():
    import torch
    import transformers
    require(ROOT == BASE/'runtime/t2t-exchange-suite-1127' and
            DATA.resolve() == BASE/'runtime/t2t-native-1125/prefix90', 'Wrong checkout/data root')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Set CUDA_VISIBLE_DEVICES to empty')
    require(CASES == (0, 10, 20) and (B, CAP) == (256, 2048), 'Pilot registration changed')
    require(sha(DATA/'samples.json') == SAMPLES_SHA and sha(Path(scoring.__file__)) == SCORER_SHA
            and version('math-verify') == '0.8.0', 'Frozen data or scorer mismatch')
    require(sha(ROOT/'benchmarks/checkpoints/diffusiongemma.json') ==
            'ceb951734eb354728d7ba984093739119addbeda981b554af46d5dc105071f53', 'Checkpoint manifest')
    native = read(STUDY/'config.json')
    require(native['revision'] == REV and transformers.__version__ == native['transformers'] == '5.12.0'
            and torch.__version__ == '2.9.0+cu128', 'Wrong checkpoint/runtime')
    tasks = load_tasks(DATA/'samples.json')
    controls = {i: archive(tasks[i], native) for i in CASES}
    from .gemma_strict_request import StrictRequest
    from .gemma_strict_close import ClosingRequest
    require(ClosingRequest is not StrictRequest and issubclass(ClosingRequest, StrictRequest),
            'Wrong treatment class; no control fallback')
    return tasks, controls, native, (StrictRequest, ClosingRequest)


def tensor_hash(name, value):
    h = sha256()
    add_tensor(h, name, value)
    return h.hexdigest()


def arm(model, data, seed, cls, fused, tokenizer):
    """Only observer hooks, never replace projector/sampler outputs. No stored full logits."""
    import torch
    events_out, decoder_ids, head_q, trace, witness = [], [], [], [], []
    prompt = data['input_ids'][0].tolist()
    origin = len(prompt)
    phase_f, phase_q = Counter(), Counter()
    seen, debt, sealed = set(), set(), {}
    emitted, blocks = [], []
    terminal_drain_f = terminal_drain_q = 0

    def decoder(_, args, kw):
        ids = kw['decoder_input_ids']
        require(ids.device.type == 'cpu' and ids.shape == kw['decoder_position_ids'].shape,
                'Non-CPU decoder/input geometry')
        decoder_ids.append(ids[0].tolist())

    def head(_, args):
        x, = args
        require(x.device.type == 'cpu' and x.ndim == 3 and x.shape[0] == 1, 'Head geometry')
        head_q.append(x.shape[1])

    def output(_, args, kw, result):
        logits = result.logits
        require(logits.device.type == 'cpu' and logits.dtype == torch.float32 and
                logits.shape == (1, head_q[-1], model.config.text_config.vocab_size), 'Returned logits')
        events_out.append(dict(argmax=logits[0].argmax(-1).tolist(), logits=tensor_hash('logits', logits),
                               sc=tensor_hash('sc', kw.get('self_conditioning_logits'))))

    with torch.inference_mode(), ExitStack() as stack:
        events = stack.enter_context(observe(model))
        proj, pos = stack.enter_context(projectors(model))
        enc = stack.enter_context(encoder_tape(model))
        stack.callback(model.model.decoder.register_forward_pre_hook(decoder, with_kwargs=True).remove)
        stack.callback(model.lm_head.register_forward_pre_hook(head).remove)
        stack.callback(model.register_forward_hook(output, with_kwargs=True).remove)
        req = cls(model, data, seed=seed, max_new_tokens=CAP, prefix_mode='right', prefix_quota=0)
        require(events == [('encoder', origin)] and enc == [(prompt, list(range(origin)))],
                'First encoder must be exactly the pure prompt')
        while req.status == 'ready':
            tick, ordinal = len(trace), len(blocks)
            require(ordinal < CAP//B and tick < (CAP//B)*(3*48+4), 'Runaway request')
            s = req.sampler  # keep the old block alive across a commit
            old = s.ids[0].tolist()
            accepted = set(s.accepted[0].nonzero().flatten().tolist())
            age, rng = s.age, s.rng.get_state()
            terminal = seen == FULL and (age == 48 or accepted == FULL)
            before = dict(canvas=old, accepted=sorted(accepted), age=age, rng=tensor_hash('rng', rng),
                          debt=sorted(debt), seen=sorted(seen), ordinal=ordinal)
            event_start = len(events)
            e = req.step(Q)
            p, rows, w = e['phase'], e['rows'], e['q']
            take = set(rows)
            require(e['tick'] == tick and e['ordinal'] == ordinal and e['offer'] == Q and
                    type(e['encoded']) is bool and 0 < w == len(rows) <= Q and rows == sorted(take) and
                    take <= FULL and not e['prefix'] and e['dirty_before'] == sorted(debt),
                    f'Bad physical record or unreported debt at {tick}')
            expected_events = [('decoder', w)] + ([('encoder', B)] if e['encoded'] else [])
            require(events[event_start:] == expected_events and len(events_out) == len(pos) ==
                    len(head_q) == len(decoder_ids) == tick+1 and head_q[-1] == w and
                    all(len(t) == tick+1 and t[-1] == w for t in proj.values()) and proj and
                    pos[-1] == [origin+B*ordinal+r for r in rows] and
                    decoder_ids[-1] == [old[r] for r in rows], f'Hook geometry/input mismatch at {tick}')
            sampled = set(e['sampled'])
            now = s.ids[0].tolist()
            acc = set(s.accepted[0].nonzero().flatten().tolist())
            flips, changed = accepted ^ acc, {i for i in FULL if now[i] != old[i]}
            require(sampled <= take and changed | flips <= sampled and
                    set(e['admitted']) == acc-accepted and set(e['revoked']) == accepted-acc and
                    set(e['edited']) == acc & accepted & changed, 'Sampler state differs from ledger')
            if p == 'sample':
                require(not terminal and not sealed and sampled and s.age == age+1 and
                        set(sorted(debt)[:Q]) <= take, 'Invalid sample/clock/debt')
            else:
                require(not sampled and s.age == age and torch.equal(rng, s.rng.get_state()),
                        'Refresh/verify advanced sampler state')
                if p == 'drain':
                    require(not sealed and take == set(sorted(debt)[:Q]) and not (terminal and fused),
                            'Unexpected drain or treatment failed to fuse terminal debt')
                    if terminal:
                        terminal_drain_f += 1
                        terminal_drain_q += w
                else:
                    require(p == 'verify' and terminal and not take.intersection(sealed) and
                            w == min(Q, B-len(sealed)), 'Premature/repeated/underfilled verify')
                    require(len(take & debt) == min(w, len(debt)) if fused else not debt,
                            'Skipped outstanding dirty rows during terminal sweep')
                    sealed.update(zip(rows, events_out[-1]['argmax'], strict=True))
            seen.update(sampled)
            debt = (debt-take) | changed | flips
            if len(sealed) == B:
                block = [sealed[r] for r in range(B)]
                stop = next((j+1 for j, t in enumerate(block) if t in eos_ids(model)), None)
                emitted.extend(block[:stop])
                more = stop is None and len(emitted) < CAP
                require(not debt and req.tokens == emitted and e['encoded'] == more and
                        req.status == ('ready' if more else 'done'), 'Hook argmax != commit/EOS/encoder')
                if more:
                    base = origin+B*ordinal
                    require(enc[-1] == (block, list(range(base, base+B))), 'Encoder != sealed B256')
                blocks.append(block)
                seen, debt, sealed = set(), set(), {}
            else:
                require(not e['encoded'] and req.status == 'ready' and req.tokens == emitted,
                        'Early commit/encode/end')
            phase_f[p] += 1
            phase_q[p] += w
            trace.append(e)
            witness.append(dict(before=before, **events_out[-1]))

    require(blocks and req.status == 'done' and len(enc) == len(blocks) and 0 < len(emitted) <= CAP,
            'Incomplete physical ledger')
    eos = eos_ids(model)
    excluded = eos | set(tokenizer.all_special_ids) | {model.generation_config.pad_token_id}
    n = sum(t not in excluded for t in emitted)
    f, q = len(events)-1, sum(w for _, w in events[1:])
    m = dict(N=n, F=f, Q=q, N_per_F=n/f, N_per_Q=n/q, prefill_rows=origin,
             decoder_F=len(pos), decoder_Q=sum(head_q), decoder_q_max=max(head_q), encoder_F=len(enc)-1,
             encoder_Q=sum(len(ids) for ids, _ in enc[1:]), phase_F=dict(phase_f), phase_Q=dict(phase_q),
             eos=emitted[-1] in eos, truncated=emitted[-1] not in eos, output_ids_count=len(emitted),
             layers=len(model.model.decoder.layers), projection_modules=len(proj))
    require(f == m['decoder_F']+m['encoder_F'] and q == m['decoder_Q']+m['encoder_Q'], 'Work mismatch')
    evidence = dict(output_ids=emitted, trace=trace, witness=witness, sealed_blocks=blocks,
                    decoder_position_ids=pos, decoder_ids=decoder_ids, events=events, encoder_inputs=enc,
                    head_widths=head_q, projection_widths={f'{i}.{k}': v for (i,k), v in proj.items()},
                    terminal_drain_F=terminal_drain_f, terminal_drain_Q=terminal_drain_q)
    return m, evidence


def eos_ids(model):
    eos = model.generation_config.eos_token_id
    return {eos} if isinstance(eos, int) else set(eos or ())


def first_diff(a, b):
    return next((i for i, (x,y) in enumerate(zip(a,b)) if x != y),
                None if len(a) == len(b) else min(len(a),len(b)))


def compare(old, new):
    def tape(d):
        return [([e[k] for k in FIELDS], w) for e,w in zip(d['trace'], d['witness'], strict=True)]
    a, b = tape(old), tape(new)
    i = first_diff(a, b)
    fork = None
    if i is not None:
        require(i < min(len(a),len(b)), 'One arm ended before a treatment difference')
        x, y = old['trace'][i], new['trace'][i]
        require(a[i][1]['before'] == b[i][1]['before'] and x['dirty_before'] and
                x['phase'] == 'drain' and y['phase'] == 'verify',
                f'Unexplained pre-treatment difference at tick {i}; do not attribute savings')
        fork = dict(tick=i, ordinal=x['ordinal'], before_equal=True, dirty=x['dirty_before'],
                    control_rows=x['rows'], dirty_first_rows=y['rows'])
    return dict(equal_leading_decoder_calls=len(a) if i is None else i, first_intervention=fork,
                first_output_token_diff=first_diff(old['output_ids'], new['output_ids']),
                output_ids_equal=old['output_ids']==new['output_ids'],
                first_full_block_diff=first_diff(old['sealed_blocks'], new['sealed_blocks']),
                archived_path_terminal_drain_F=old['terminal_drain_F'],
                archived_path_terminal_drain_Q=old['terminal_drain_Q'])


def paired(model, processor, task, controls, classes, root):
    i, seed = task['index'], task['seed']
    c, historical = controls
    cfg = generation_settings(model.generation_config)
    require(cfg == historical['generation_config'], 'Historical generation settings differ')
    data = processor.apply_chat_template([{'role':'user', 'content':task['prompt']}], tokenize=True,
               add_generation_prompt=True, return_dict=True, return_tensors='pt')
    digest = input_digest(data)
    require(digest == c['input_sha256'], 'Historical tokenized prompt mismatch')
    metrics, evidence = {}, {}
    for name, cls in zip(('control','dirty_first'), classes, strict=True):
        print(f'{i:02} {name}: starting', flush=True)
        m, e = arm(model, {k:v.clone() for k,v in data.items()}, seed, cls, name=='dirty_first',
                   processor.tokenizer)
        text = processor.tokenizer.decode(e['output_ids'], skip_special_tokens=True)
        score = scoring.score(text, task)
        e.update(output_text=text, score=score)
        m.update(correct=bool(score['correct']), metric=score['metric'])
        write(root/f'{i:02}-{name}.json', dict(metrics=m, evidence=e, input_sha256=digest, seed=seed))
        if name == 'control':
            for k, v in m.items():
                require(c[k] == v, f'Control disagrees with archived {i}:{k}')
            for k in ('output_ids','output_text','score','trace','events','decoder_position_ids',
                      'encoder_inputs','projection_widths'):
                require(historical[k] == json.loads(json.dumps(e[k])), f'Control replay mismatch {i}:{k}')
        metrics[name], evidence[name] = m, e
    result = dict(index=i, task_id=task['task_id'], seed=seed, input_sha256=digest, arms=metrics,
                  **compare(evidence['control'], evidence['dirty_first']))
    result['dirty_first_minus_control'] = {k:metrics['dirty_first'][k]-metrics['control'][k] for k in
        ('N','F','Q','N_per_F','N_per_Q','decoder_F','decoder_Q','encoder_F','encoder_Q','correct')}
    # Not an extra measured arm: zero residual would be consistent with simply dropping the drains.
    result['net_delta_beyond_removing_control_terminal_drains'] = {
        k:result['dirty_first_minus_control'][k]+evidence['control']['terminal_drain_'+k]
        for k in ('F','Q')}
    write(root/f'{i:02}-pair.json', result)
    print(f"{i:02}: audited delta {result['dirty_first_minus_control']}", flush=True)
    return result


def summarize(pairs):
    sums = {}
    for name in ('control', 'dirty_first'):
        ms = [p['arms'][name] for p in pairs]
        t = {k:sum(m[k] for m in ms) for k in ('N','F','Q','decoder_Q','encoder_Q','correct')}
        t.update(pooled_N_per_F=t['N']/t['F'], pooled_N_per_Q=t['N']/t['Q'],
                 mean_request_N_per_F=sum(m['N_per_F'] for m in ms)/len(ms))
        sums[name] = t
    x, y = sums['control'], sums['dirty_first']
    regressions = [p['index'] for p in pairs if p['arms']['control']['correct']
                   and not p['arms']['dirty_first']['correct']]
    return dict(pairs=pairs, totals=sums, correct_to_incorrect=regressions,
                measured_net_change={k:y[k]-x[k] for k in ('N','F','Q','decoder_Q','encoder_Q')},
                baseline_terminal_drains={k:sum(p['archived_path_terminal_drain_'+k] for p in pairs)
                                          for k in ('F','Q')},
                same_output_pairs=sum(p['output_ids_equal'] for p in pairs),
                pilot_signal=('lower physical F and Q with no observed score loss; replicate' if
                    y['F'] < x['F'] and y['Q'] < x['Q'] and not regressions else
                    'no joint physical-cost and score-preservation signal'), limitation=LIMITATION)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', action='store_true', help='Explicitly load real weights and run six requests')
    args = ap.parse_args()
    if args.run:
        require(os.sched_getaffinity(0) == set(range(36,40)), 'Use taskset -c 36-39')
        require(not os.path.lexists(OUTPUT), 'Output already exists; never resume/overwrite implicitly')
    tasks, controls, native, classes = preflight()
    if not args.run:
        print('Preflight OK: classes, scorer, frozen inputs and all three CPU control pins; no weights loaded.')
        return
    import torch
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    os.umask(0o077)
    OUTPUT.mkdir(mode=0o700, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    meta = dict(cases=CASES, q=Q, k=0, cap=CAP, device='cpu', affinity=sorted(os.sched_getaffinity(0)),
                torch=torch.__version__, transformers=version('transformers'), native=native,
                checkpoint=str(CKPT), samples_sha256=SAMPLES_SHA, scorer_sha256=SCORER_SHA,
                control_pins=HISTORY, sources={n:sha(HERE/(n+'.py')) for n in
                ('gemma_strict_close_pilot','gemma_strict_close','gemma_strict_request')}, limitation=LIMITATION)
    write(OUTPUT/'registration.json', meta)
    check_snapshot(CKPT)
    proc = AutoProcessor.from_pretrained(CKPT, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(CKPT, local_files_only=True,
                dtype=torch.bfloat16, device_map={'':'cpu'}, attn_implementation='sdpa',
                experts_implementation='grouped_mm').eval()
    require(all(p.device.type == 'cpu' for p in model.parameters()) and model.config.canvas_length == B
            and model.generation_config.max_denoising_steps == 48, 'Wrong model geometry/device')
    pairs = [paired(model, proc, tasks[i], controls[i], classes, OUTPUT) for i in CASES]
    write(OUTPUT/'summary.json', summarize(pairs))
    print(f'Complete: {OUTPUT}/summary.json', flush=True)


if __name__ == '__main__':
    main()

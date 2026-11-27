"""Read-only CPU arithmetic on sealed strict Gemma Q128 request pairs (stdlib only).

The compact record is the writer's last atomic write, after raw/evidence.json.
Missing seals exclude the whole pair; a present but inconsistent seal is fatal.
Never import a runner, model or scorer. Only counts/fingerprints leave the reader.
"""
import argparse
from collections import Counter
from hashlib import sha256
from itertools import groupby
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3] / 't2t-native-1125/prefix90/gemma-strict-eval-v1'
OUTPUT = HERE / 'controls/gemma-strict-q128-close-cost-cpu.json'
TOKENIZER = HERE.parents[3].parent / 'models/diffusiongemma-cd40195/tokenizer.json'
SAMPLES_SHA = 'bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed'
TOKENIZER_SHA = 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f'
SOURCE_SET_SHA = '5b85d6591f40bac0db578acf0300aea8ff94517e966e1fefee69e057bbbc78cb'
NATIVE_SHA = '226fcf0cc8bf0e17039d0676141136fdbc169f15414a7fe6c102089353ae1782'
CFG_SHA = '134cc67165891edb51515a0cac88e61ee0b95d28a524eafaa6d027015bfefc02'
SCORER_SHA = 'a72cec90206f69616b9de436d052f6df54cf075ba9abeac4a75a11e46587d692'
B, CAP, Q = 256, 2048, 128
CASES, ARMS = (0, 10, 20), (0, 32)
PROJECTIONS = {f'{i}.{p}_proj' for i in range(30) for p in ('q', 'k', 'v')
               if p != 'v' or i % 6 != 5}


def need(condition, msg):
    if not condition:
        raise ValueError(msg)


def read(path, expect=None):
    data = path.read_bytes()
    digest = sha256(data).hexdigest()
    need(expect is None or digest == expect, f'wrong fingerprint: {path.name}')
    return json.loads(data), digest


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def positions(e, key):
    xs = e[key]
    need(isinstance(xs, list) and all(type(i) is int and 0 <= i < B for i in xs)
         and len(set(xs)) == len(xs), f'invalid block-local {key}')
    return set(xs)


def drain_composition(block):
    """Attribute measured pure-drain Q, not a simulated saving on a new trajectory."""
    priority, totals = set(), Counter()
    for e in block:
        dirty, rows = (set(e[k]) for k in ('dirty_before','rows'))
        need(priority <= dirty, 'unpaid acceptance or edit transition')
        if e['phase'] == 'drain':
            need(rows <= dirty, 'drain includes clean rows')
            totals.update(drain_write_or_flip_Q=len(rows & priority),
                          drain_unaccepted_noise_Q=len(rows-priority), pure_drains=1)
        priority = (priority - rows) | set(e['admitted']) | set(e['edited']) | set(e['revoked'])
    need(not priority, 'terminal token write missed')
    return {key:totals[key] for key in DRAIN_FIELDS}


DRAIN_FIELDS = ('drain_write_or_flip_Q','drain_unaccepted_noise_Q','pure_drains')


def close_geometry(block, q=Q):
    """Original sequential verify partition. No inference of KV/logit equality."""
    start_v = next((i for i, e in enumerate(block) if e['phase'] == 'verify'), len(block))
    verifies = block[start_v:]
    chunks = [e['rows'] for e in verifies]
    need(chunks == [list(range(i, min(i + q, B))) for i in range(0, B, q)] and
         all(e['phase'] == 'verify' and not e['dirty_before'] for e in verifies), 'wrong verify sweep')
    start_d = start_v
    while start_d and block[start_d-1]['phase'] == 'drain':
        start_d -= 1
    need(start_d > 0 and block[start_d-1]['phase'] == 'sample', 'missing last sample')
    drains = block[start_d:start_v]
    debt = set(drains[0]['dirty_before']) if drains else set()
    paid = set()
    for e in drains:
        rows = positions(e, 'rows')
        need(debt - paid == positions(e, 'dirty_before') and e['rows'] == sorted(debt - paid)[:q]
             and not e['sampled'] and not e['encoded'], 'wrong terminal dirty drain')
        paid |= rows
    need(paid == debt, 'terminal debt not flushed')
    first = set(chunks[0])
    # Only a contiguous suffix of pure drains wholly overwritten by verify[0]
    # can be discarded by this argument. Do not erase a preceding drain whose
    # writes may have fed a retained later drain at a deeper layer.
    removable = []
    for d in reversed(drains):
        if not set(d['rows']) <= first:
            break
        removable.append(d)
    hits = [len(debt.intersection(chunk)) for chunk in chunks]
    return dict(ordinal=block[0]['ordinal'], stop_reason=verifies[0]['stop_reason'],
                last_sample_tick=block[start_d-1]['tick'], verify_ticks=[e['tick'] for e in verifies],
                terminal_drain_ticks=[e['tick'] for e in drains], terminal_drain_widths=[e['q'] for e in drains],
                other_drain_F=sum(e['phase'] == 'drain' for e in block[:start_d]),
                other_drain_Q=sum(e['q'] for e in block[:start_d] if e['phase'] == 'drain'),
                verify_F=len(verifies), verify_Q=B, dirty_by_verify_chunk=hits,
                first_verify_plus_all_dirty_Q=len(first | debt), fits_original_first=len(first | debt) <= q,
                dirty_fits_one_original_chunk=bool(debt) and max(hits) == len(debt),
                original_order_save_F=len(removable), original_order_save_Q=sum(e['q'] for e in removable),
                repacked_ceiling_F=len(drains), repacked_ceiling_Q=len(paid))


def audit_tape(evidence, compact, special, q=Q):
    tr = evidence['trace']
    tokens, enc = evidence['output_ids'], evidence['encoder_inputs']
    cfg = evidence['generation_config']
    need(fingerprint(cfg) == CFG_SHA, 'wrong generation config')
    eos = set(cfg['eos_token_id'])
    excluded = special | eos | {cfg['pad_token_id']}
    n = sum(t not in excluded for t in tokens)
    stop = next((i for i, t in enumerate(tokens) if t in eos), None)
    need(0 < len(tokens) <= CAP and (stop == len(tokens)-1 if stop is not None else len(tokens) == CAP),
         'not first EOS or cap')
    origin = len(enc[0][0])
    expected = [['encoder', origin]]
    need(origin > 0 and enc[0][1] == list(range(origin)), 'bad prefill positions')
    widths = [s['q'] for s in tr]
    projs = evidence['projection_widths']
    need(tr and set(projs) == PROJECTIONS and all(w == widths for w in projs.values()), 'projection tape mismatch')
    pos = evidence['decoder_position_ids']
    need(len(tr) == len(pos), 'decoder position tape mismatch')
    counts, sums = Counter(), Counter()
    blocks = []
    prefix_rows = prefix_edits = prefix_revokes = 0
    for ordinal, (b, group) in enumerate(groupby(tr, key=lambda e: e['ordinal'])):
        block = list(group)
        need(b == ordinal, 'unexpected block order')
        base = origin + b * B
        accepted, seen, debt_min, debt_max = set(), set(), set(), set()
        reasons = set()
        for e in block:
            tick, w, phase = e['tick'], e['q'], e['phase']
            need(tick == sum(counts.values()) and phase in ('sample', 'drain', 'verify')
                 and e['offer'] == q and 0 < w <= q, 'invalid decoder call')
            rows, dirty, sampled, pref, add, edit, rev = (positions(e, k) for k in
                ('rows', 'dirty_before', 'sampled', 'prefix', 'admitted', 'edited', 'revoked'))
            need(e['rows'] == sorted(rows) and len(rows) == w and pos[tick] == [base+r for r in sorted(rows)],
                 'physical positions disagree')
            need(debt_min <= dirty <= debt_max and set(sorted(dirty)[:q]) <= rows, 'dirty ledger mismatch')
            need(pref <= sampled <= rows and (add | edit | rev) <= sampled and not add & accepted
                 and (edit | rev) <= accepted and not edit & rev and bool(sampled) == (phase == 'sample'),
                 'sample ledger mismatch')
            if pref:
                frontier = min(set(range(B)) - accepted, default=B)
                need(len(seen) == B and len(pref) <= compact['k'] and pref <= set(range(frontier))-dirty,
                     'not a stable prefix')
            if phase == 'verify':
                reasons.add(e['stop_reason'])
                need(len(seen) == B and not dirty and e['stop_reason'] in ('limit', 'resolved', 'native_stop')
                     and (e['stop_reason'] != 'resolved' or len(accepted) == B), 'bad terminal state')
            if phase == 'drain':
                need(rows == set(sorted(dirty)[:q]), 'not a pure drain')
            # Renoise changes are not listed individually. Bound the next debt
            # after a sample; on a drain the transition is known exactly.
            debt_min, debt_max = (dirty-rows) | add | edit | rev, (dirty-rows) | sampled
            accepted, seen = (accepted | add)-rev, seen | sampled
            counts[phase] += 1
            sums[phase] += w
            prefix_rows += len(pref)
            prefix_edits += len(edit & pref)
            prefix_revokes += len(rev & pref)
            expected.append(['decoder', w])
            encoded = tick < len(tr)-1 and e is block[-1]
            need(e['encoded'] is encoded, 'wrong encoder flag')
            if encoded:
                expected.append(['encoder', B])
                need(enc[b+1] == [tokens[b*B:(b+1)*B], list(range(base, base+B))], 'wrong generation encoder tape')
        need(len(reasons) == 1, 'inconsistent close reason')
        blocks.append(close_geometry(block, q) | drain_composition(block))
    need(expected == evidence['events'] and len(enc) == len(blocks) and (len(blocks)-1)*B < len(tokens) <= len(blocks)*B,
         'physical event order or block count mismatch')
    metrics = dict(N=n, F=len(expected)-1, Q=sum(w for _, w in expected[1:]), decoder_F=len(tr),
                   decoder_Q=sum(widths), encoder_F=len(blocks)-1, encoder_Q=B*(len(blocks)-1),
                   prefill_rows=origin, decoder_q_max=max(widths), layers=30, projection_modules=len(projs),
                   phase_F=dict(counts), phase_Q=dict(sums), prefix_rows=prefix_rows, prefix_edited=prefix_edits,
                   prefix_revoked=prefix_revokes, eos=stop is not None, truncated=stop is None,
                   output_ids_count=len(tokens), correct=evidence['score']['correct'])
    need(all(compact[k] == v for k, v in metrics.items()), 'sealed metrics disagree with tape')
    return metrics, blocks


def case_arm(root, task, k, special, q=Q):
    idx = task['index']
    tag = f'cpu-q{q}-k{k}-{idx:02d}'
    folder = root/'raw'/tag
    compact, seal_sha = read(root/'compact'/(tag+'.json'))
    run, run_sha = read(folder/'run.json', compact['run_sha256'])
    ev, ev_sha = read(folder/'evidence.json')
    identity = dict(device='cpu', q=q, k=k, index=idx, seed=task['seed'], samples_sha256=SAMPLES_SHA,
                    scorer_sha256=SCORER_SHA)
    need(all(run[key] == value == compact[key] for key, value in identity.items()) and
         run['case'] == task['task_id'] == compact['task_id'] and compact['dataset'] == task['dataset']
         and fingerprint(run['native']) == NATIVE_SHA and fingerprint(run['sources']) == SOURCE_SET_SHA
         and compact['revision'] == run['native']['revision'] and compact['prefix_mode'] == 'right'
         and compact['cap'] == CAP, 'wrong sealed run provenance')
    metrics, blocks = audit_tape(ev, compact, special, q)
    need(k or not metrics['prefix_rows'], 'K0 used prefix seats')
    bounds = {key: sum(b[key] for b in blocks) for key in (*BOUNDS,*DRAIN_FIELDS)}
    need(bounds['drain_write_or_flip_Q'] + bounds['drain_unaccepted_noise_Q']
         == metrics['phase_Q'].get('drain',0), 'drain partition lost a paid row')
    return (compact['input_sha256'], fingerprint(ev['encoder_inputs'][0])), dict(
        index=idx, dataset=task['dataset'], k=k, **metrics, **bounds, blocks=blocks,
        sha256=dict(compact=seal_sha, run=run_sha, evidence=ev_sha))


BOUNDS = ('original_order_save_F', 'original_order_save_Q', 'repacked_ceiling_F', 'repacked_ceiling_Q')


def analyze(root=ROOT, tokenizer=TOKENIZER):
    samples = read(root.parent/'samples.json', SAMPLES_SHA)[0]['samples']
    special = {t['id'] for t in read(tokenizer, TOKENIZER_SHA)[0]['added_tokens'] if t['special']}
    arms, excluded = [], []
    for idx in CASES:
        missing = [k for k in ARMS if not (root/'compact'/f'cpu-q128-k{k}-{idx:02d}.json').is_file()]
        if missing:
            excluded.append(dict(index=idx, missing_compact_k=missing))
            continue
        pair = [case_arm(root, samples[idx], k, special) for k in ARMS]
        need(pair[0][0] == pair[1][0], 'paired input mismatch')
        arms.extend(record for _, record in pair)
    totals = [{key: sum(r[key] for r in arms if r['k'] == k) for key in ('N', 'F', 'Q', 'correct', *BOUNDS,*DRAIN_FIELDS)}
              | dict(k=k, requests=len(arms)//2) for k in ARMS]
    for t in totals:
        paired = [a for a in arms if a['k'] == t['k']]
        t['observed_macro_TPF'] = sum(a['N']/a['F'] for a in paired)/len(paired) if paired else None
        t['observed_pooled_efficiency'] = t['N']/t['Q'] if paired else None
    return dict(dataset='gemma-strict-eval-v1', budget=Q, canvas=B, preselected=list(CASES),
                paired_indices=[a['index'] for a in arms if a['k'] == 0], excluded=excluded,
                sample_manifest_sha256=SAMPLES_SHA, tokenizer_sha256=TOKENIZER_SHA,
                method=dict(measured='Observed baseline only. N excludes specials; F/Q include generation encoder, exclude prompt.',
                    original_order='Drop only the immediately preceding pure drain suffix fully overwritten by verify chunk 0. Conditional, not measured parity.',
                    conditions='All B rows already live. IDs, SC, RNG, encoder prefix, mask and ordinal unchanged. Every layer writes all selected KV before attention. Drain logits unused; keep both original verify chunks, argmax commit and encoder boundary.',
                    visibility='Same-block attention may read either half. Any remaining dirty row visible outside chunk 0 can affect chunk 0 and later-layer KV; sharing some coordinates or identical SC alone is insufficient.',
                    repacked='Optimistic ceiling: retain one verify projection per each of B positions, pack dirty rows first within cap; dropping all pure terminal drains may change cross-row dependencies, logits, EOS and all subsequent work.',
                    limitations='No model execution, rescoring, measured speedup, or counterfactual quality/TPF. No sample, nonterminal drain, verify coverage, or encoder saving claimed.'),
                arms=arms, totals=totals)


def tiers(root, tokenizer):
    """No missing cases or arms admitted. Count only the measured, pure refresh calls."""
    tasks = read(root.parent/'samples.json', SAMPLES_SHA)[0]['samples']
    special = {t['id'] for t in read(tokenizer, TOKENIZER_SHA)[0]['added_tokens'] if t['special']}
    rows = []
    for q in (32, 64, 128):
        groups = [[case_arm(root, tasks[i], k, special, q) for k in (0,q//4)] for i in CASES]
        for a, b in groups:
            need(a[0] == b[0], 'mismatched paired prompts')
        for j, k in enumerate((0,q//4)):
            arms = [pair[j][1] for pair in groups]
            agg = {key:sum(r[key] for r in arms) for key in (
                'N','F','Q','correct','decoder_Q','encoder_Q',*DRAIN_FIELDS)}
            agg['pure_drain_Q'] = sum(r['phase_Q'].get('drain',0) for r in arms)
            agg['terminal_drain_F'] = sum(r['repacked_ceiling_F'] for r in arms)
            agg['terminal_drain_Q'] = sum(r['repacked_ceiling_Q'] for r in arms)
            rows.append(dict(q=q,k=k,requests=len(arms),macro_TPF=sum(r['N']/r['F'] for r in arms)/len(arms),
                             efficiency=agg['N']/agg['Q'], **agg,
                             evidence_sha256=[r['sha256']['evidence'] for r in arms]))
    return dict(cases=CASES, device='cpu',cap=CAP,rows=rows,
                scope='Paid rows in pure drain calls; neither exposure in mixed calls nor a counterfactual saving.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=ROOT, type=Path)
    parser.add_argument('--tokenizer', default=TOKENIZER, type=Path)
    parser.add_argument('--out', default=OUTPUT, type=Path)
    parser.add_argument('--tiers', action='store_true', help='Audit the full Q32/Q64/Q128 three-case grid')
    args = parser.parse_args()
    if args.tiers and args.out == OUTPUT:
        parser.error('--tiers requires a distinct --out')
    report = tiers(args.root, args.tokenizer) if args.tiers else analyze(args.root, args.tokenizer)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('Wrote', args.out)

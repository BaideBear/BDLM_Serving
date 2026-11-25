"""CPU audit of the four frozen30 LLaDA B32 Q8/E1 context placement arms.

No selector, scorer, checkpoint or token features are loaded. The winner and
its linear score are observations, not an independent embedding-model replay.
"""
import argparse
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path
import random

from bdlm_serving.evaluation.frozen import MANIFEST
from tools.t2t_results import table


B, Q = 32, 8
POLICIES = tuple('t2t_q8_f0_m7_hard_context_' + s for s in
                 ('slack', 'slack_random', 'gate', 'gate_random'))
WEIGHTS = 'dbcaf8d1496bbd4867a9ad9f44e592bec520c14872c1cb6476f257521d03081d'
REASONS = ('overflow', 'no_room', 'no_old', 'protected_mask', 'no_slack',
           'below_threshold', 'already_selected', 'swap_old', 'swap_mask')
STAGES = ('prefill', 'bootstrap', 'selected', 'verify', 'commit')
CUTOFF = math.log(.05 / .95)


def check(ok, message):
    if not ok:
        raise ValueError(message)


def integer(value):
    return type(value) is int and value >= 0


def positions(xs):
    check(isinstance(xs, list) and all(map(integer, xs)) and xs == sorted(set(xs)),
          'expected sorted distinct positions')
    return set(xs)


def same(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def load(path):
    def unique(items):
        obj = dict(items)
        check(len(items) == len(obj), 'duplicate JSON key')
        return obj
    return json.loads(path.read_text(), object_pairs_hook=unique)


def placement(e, eligible, policy, seed):
    m = e['context_move']
    check(set(m) == {'base', 'fresh', 'old_pool', 'masks', 'rank', 'score', 'reason', 'entrant', 'victim'},
          'wrong move schema')
    fresh, masks, old, logged = (positions(m[k]) for k in ('fresh', 'masks', 'old_pool', 'base'))
    check(not (fresh & masks or fresh & old or masks & old) and fresh | masks | old == eligible,
          'incomplete/overlapping eligible partition')
    free = min(len(eligible), max(Q, len(fresh))) - len(fresh)
    first_masks = sorted(masks)[:free]
    base = fresh | set(first_masks) | set(sorted(old)[:free-len(first_masks)])
    check(base == logged and len(fresh) == e['mandatory_fresh'], 'noedit base or fresh count mismatch')
    slack = len(first_masks) < free
    rank, score = m['rank'], m['score']
    entrant = victim = None
    if len(fresh) > Q:
        reason = 'overflow'
    elif free == 0:
        reason = 'no_room'
    elif not old:
        reason = 'no_old'
    elif not slack and free < 2:
        reason = 'protected_mask'
    elif not slack and '_slack' in policy:
        reason = 'no_slack'
    else:
        check(integer(rank) and rank in old and type(score) in (float, int) and math.isfinite(score),
              'missing or invalid ranked evidence')
        if not slack and score < CUTOFF:
            reason = 'below_threshold'
        elif rank in base:
            reason = 'already_selected'
        else:
            reason = 'swap_old' if slack else 'swap_mask'
            victim = max(base & (old if slack else masks))
            rng = random.Random(f"context-swap:{seed}:{e['block']}:{e['hard_clock']}")
            entrant = rng.choice(sorted(old - base)) if policy.endswith('_random') else rank
    if reason in REASONS[:5]:
        check(rank is None and score is None, 'unexamined score or rank')
    check(same([reason, victim, entrant], [m['reason'], m['victim'], m['entrant']]),
          'wrong move reason, victim, or entrant')
    expected = base if entrant is None else (base - {victim}) | {entrant}
    shift = B * e['block']
    check(e['absolute_positions'] == [shift+i for i in sorted(expected)] and
          e['reserved_extra_edit_positions'] == ([] if entrant is None else [shift+entrant]),
          'physical Q or explicit reserve differs from replay')
    return masks, fresh


def transition(prev, masks, fresh):
    """MASK transitions locate writes. A subsequent selected step also names fresh IDs."""
    before, queried, reserve, hits, admits, revokes, strict = prev
    new, lost = before - masks, masks - before
    check(len(new) == admits and len(lost) == revokes and (new | lost) <= queried,
          'MASK transition impossible on preceding Q')
    check(not (hits & masks), 'reserved strict hit was revoked')
    if fresh is not None:
        check(new <= fresh <= queried - masks and len(fresh-new) == strict,
              'fresh rows disagree with preceding admissions or strict edits')
        check((fresh-new) & reserve == hits, 'reserved strict identities disagree with next fresh set')


def audit_request(raw, sample, policy):
    check(policy in POLICIES and raw['policy'] == policy and raw['model'] == 'llada21'
          and raw['status'] == 'success' and all(same(raw[k], sample[k]) for k in
          ('index', 'dataset', 'task_id', 'seed')), 'wrong request identity')
    c = Counter(dict.fromkeys((*REASONS, 'swaps', 'noops', 'random_rank_collisions',
        'reserved_hits', 'swap_old_hits', 'swap_mask_hits', 'strict_edits', 'admissions',
        'revocations', 'fresh_overflow_rows', 'selected_fresh', 'requests'), 0))
    c['requests'] = 1
    fs, qs, hist = Counter(), Counter(), Counter()
    prev = eligible = block = None
    last_stage, next_fresh, commits = None, 0, 0
    for clock, e in enumerate(raw['events'], 1):
        stage, b = e['stage'], e['block']
        check(stage in STAGES and integer(b) and integer(e['hard_clock']) and e['hard_clock'] == clock,
              'bad stage/block/clock')
        selected = stage == 'selected'
        denoise = stage in ('bootstrap', 'selected', 'verify')
        check(('context_move' in e) == selected and e['phase'] == ('denoise' if denoise else stage),
              'move missing or outside selected denoise')
        absolute = positions(e['absolute_positions'])
        q, shift = len(absolute), B*b
        check(e['store_kv'] is (not denoise) and e['approximate_same_block_kv'] is selected
              and same([e['prefix_kv_length'],e['visible_kv'],e['context_version']],
                       [0 if stage == 'prefill' else shift, shift if stage == 'prefill' else shift+B, commits]),
              'wrong physical KV range or version')
        check(q and all(integer(e[k]) for k in ('F','q_tokens','actual_q','physical_q','overflow',
              'q_budget','mandatory_fresh','hard_strict_edits','q_start','q_end')) and
              e['F'] == 1 and e['q_tokens'] == e['actual_q'] == e['physical_q'] == q and
              e['q_budget'] == Q and e['overflow'] == max(0,q-Q) and
              same(e['layer_physical_q'], [q]*20) and
              (e['q_start'], e['q_end']) == (min(absolute), max(absolute)+1), 'physical ledger mismatch')
        picked = {i-shift for i in absolute}
        reserve = {i-shift for i in positions(e['reserved_extra_edit_positions'])}
        hits = {i-shift for i in positions(e['reserved_strict_edit_positions'])}
        strict = e['hard_strict_edits']
        check(hits <= reserve <= picked and len(reserve) <= int(selected) and len(hits) <= strict,
              'invalid reserved seats or direct strict hits')
        check(e['mandatory_fresh'] == next_fresh, 'fresh count chain mismatch')
        if not selected:
            check(absolute == set(range(shift)) if stage == 'prefill' else picked == set(range(B)),
                  'invalid dense physical positions')
        if stage == 'prefill':
            check(clock == 1 and not strict and q == raw['prefill_rows'], 'invalid prefill')
            block = b
        elif stage == 'bootstrap':
            check(last_stage in (None, 'prefill', 'commit') and
                  (b == (0 if block is None else block + int(last_stage == 'commit'))), 'invalid block start')
            block = b
            n = e['masked_before']
            check(integer(n) and 1 <= n <= B and (last_stage != 'commit' or n == B), 'invalid bootstrap MASKs')
            eligible = set(range(B-n, B))  # First block can contain the tail of the prompt.
            masks, fresh = eligible, set()
            step, post = 0, 0
        else:
            check(prev is not None and b == block, 'invalid block continuation')
            masks, fresh = placement(e, eligible, policy, raw['seed']) if selected else (set(), None)
            transition(prev, masks, fresh)
        if denoise:
            step += 1
            post += stage == 'verify'
            check(same([e['step_in_block'],e['post_steps']], [step,post]), 'invalid denoise step counters')
            keys = ('masked_before','selected_masked_before','accepted','edited','masked_after')
            check(all(integer(e[k]) for k in keys), 'invalid transition counters')
            before, in_q, accepted, edits, after = (e[k] for k in keys)
            revokes = edits-strict
            admits = before-after+revokes
            check(len(masks) == before and (not selected or before > 0) and len(picked & masks) == in_q
                  and 0 <= admits <= accepted <= in_q and after <= len(eligible) and revokes >= 0
                  and edits <= len((picked & eligible)-masks) and
                  strict <= len((picked & eligible)-masks-(reserve-hits)), 'impossible denoise outcomes')
            next_fresh = admits+strict
            prev = masks, picked, reserve, hits, admits, revokes, strict
            c.update(strict_edits=strict, admissions=admits, revocations=revokes, native_edits=edits)
            if selected:
                m = e['context_move']
                reason = m['reason']
                c[reason] += 1
                c.update(swaps=len(reserve), noops=int(not reserve), reserved_hits=len(hits),
                         selected_fresh=len(fresh), fresh_overflow_rows=e['overflow'],
                         random_rank_collisions=int(bool(reserve) and policy.endswith('_random') and
                                                    m['rank'] == m['entrant']))
                if reserve:
                    c[reason+'_hits'] += len(hits)
        elif stage == 'commit':
            check(last_stage == 'verify' and not strict, 'missing final verify or commit logged strict edits')
            commits += 1
            prev, next_fresh = None, 0
        if stage != 'prefill':
            hist[str(q)] += 1
        fs[stage] += 1
        qs[stage] += q
        last_stage = stage
    check(last_stage == 'commit', 'incomplete request')
    for stage in STAGES:
        c[stage+'_F'], c[stage+'_Q'] = fs[stage], qs[stage]
    pre = qs.pop('prefill', 0)
    fs.pop('prefill', None)
    f, q, n = sum(fs.values()), sum(qs.values()), raw['output_tokens']
    check(isinstance(raw['output_ids'], list) and integer(n) and n <= len(raw['output_ids']) and
          same([f,q,q,max(map(int,hist)),pre,c['native_edits'],dict(fs),dict(qs),dict(hist)],
               [raw[k] for k in ('generation_forwards','generation_query_rows','generation_physical_padded_rows',
                'max_q','prefill_rows','t2t_edits','forwards_by_phase','q_by_phase','q_histogram')]) and
          math.isclose(raw['tpf'], n/f, rel_tol=1e-12) and math.isclose(raw['token_efficiency'], n/q, rel_tol=1e-12),
          'request totals differ from physical events')
    c.update(F=f, Q=q, N=n)
    return c


# Same-state fields at the first fork: widths and model input state, not the
# queried row identities, their MASK count, or any post-forward measurements.
FORK_FIELDS = ('stage','phase','block','hard_clock','step_in_block','post_steps','F','actual_q',
               'q_tokens','physical_q','layer_physical_q','prefix_kv_length','visible_kv','context_version',
               'store_kv','approximate_same_block_kv','mandatory_fresh','overflow','q_budget','masked_before')


def audit_pair(a, b):
    c = Counter(requests=1, forks=0, no_fork=0, shared_events=0, shared_noops=0, shared_swaps=0,
                shared_generation_F=0, shared_generation_Q=0, rank_random_collisions=0,
                fork_one_swap=0, fork_two_swaps=0, fork_Q_per_arm=0, fork_left_admissions=0,
                fork_right_admissions=0, fork_left_hits=0, fork_right_hits=0)
    for x, y in zip(a['events'], b['events']):
        check((x['stage'],x['block']) == (y['stage'],y['block']), 'stage diverged before placement')
        if x['stage'] == 'selected':
            m, n = x['context_move'], y['context_move']
            check(all(m[k] == n[k] for k in ('base','fresh','masks','old_pool')), 'shared input partition differs')
            if m['rank'] is not None and n['rank'] is not None:
                check((m['rank'], m['score']) == (n['rank'], n['score']), 'shared ranking differs')
            if x['absolute_positions'] != y['absolute_positions']:
                check(all(same(x[k], y[k]) for k in FORK_FIELDS), 'pre-state or work differs at fork')
                two = m['entrant'] is not None and n['entrant'] is not None
                c['fork_two_swaps' if two else 'fork_one_swap'] += 1
                c['forks'] = 1
                c['fork_Q_per_arm'] = x['physical_q']
                for side, e in (('left', x), ('right', y)):
                    c[f'fork_{side}_admissions'] = e['masked_before']-e['masked_after']+e['edited']-e['hard_strict_edits']
                    c[f'fork_{side}_hits'] = len(e['reserved_strict_edit_positions'])
                return c, x['hard_clock'], x['block']
            if m['entrant'] is None:
                c['shared_noops'] += 1
            else:
                c['shared_swaps'] += 1
                c['rank_random_collisions'] += a['policy'].endswith('_random') != b['policy'].endswith('_random')
        check(same({k:v for k,v in x.items() if k != 'context_move'},
                   {k:v for k,v in y.items() if k != 'context_move'}), 'history differs at identical placement')
        c['shared_events'] += 1
        if x['stage'] != 'prefill':
            c['shared_generation_F'] += x['F']
            c['shared_generation_Q'] += x['physical_q']
    check(len(a['events']) == len(b['events']) and same(
        {k:v for k,v in a.items() if k not in ('events','policy')},
        {k:v for k,v in b.items() if k not in ('events','policy')}), 'nonfork length, output, or work differs')
    c['no_fork'] = 1
    return c, None, None


def discover(runs):
    arms = {}
    for d in runs:
        meta = load(d/'run.json')
        check(same([meta[k] for k in ('model','batch_size','cap','block','seed_offset','context_weights_sha256')],
                   ['llada21',1,2048,B,0,WEIGHTS]), 'wrong run configuration or frozen weights')
        check(same(load(d/'qualified.json'), dict(model='llada21', full_width=B, matched=30)), 'unqualified run')
        names = meta['policies']
        check(names and len(meta['arms']) == len(names), 'empty or malformed arm list')
        for p, spec in zip(names, meta['arms']):
            check(p in POLICIES and p not in arms and same(spec, dict(budget=Q, fresh=0, mask=7,
                  selection=p.removeprefix('t2t_q8_f0_m7_'))), 'unknown, duplicated or misdeclared arm')
            check(same(load(d/f'complete-{p}.json'), dict(policy=p,requests=30)), 'incomplete arm')
            check({f.name for f in d.glob(f'[0-9]*-{p}.json')} == {f'{i:02d}-{p}.json' for i in range(30)},
                  'incomplete or unexpected request indices')
            arms[p] = d
    check(set(arms) == set(POLICIES), 'expected all four arms')
    return arms


def audit(runs, out):
    arms = discover(runs)
    samples = load(MANIFEST)['samples']
    check([s['index'] for s in samples] == list(range(30)), 'not frozen30')
    rows, totals = [], {}
    for s in samples:
        i = s['index']
        raw = {}

        def record(p, ref, counts, clock=None, block=None):
            totals.setdefault((p, ref), Counter()).update(counts)
            reasons = [raw[k]['events'][clock-1]['context_move']['reason'] for k in (p, ref)] if clock else ('', '')
            rows.append(dict(policy=p, comparison=ref, index=i, first_fork_clock=clock, first_fork_block=block,
                             first_left_reason=reasons[0], first_right_reason=reasons[1], **counts))

        for p in POLICIES:
            raw[p] = load(arms[p]/f'{i:02d}-{p}.json')
            try:
                record(p, '', audit_request(raw[p], s, p))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'{p} request {i}: {exc}') from exc
        for p, ref in combinations(POLICIES, 2):
            try:
                record(p, ref, *audit_pair(raw[p], raw[ref]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'{p} vs {ref} request {i}: {exc}') from exc
    summary = [dict(policy=p, comparison=r, **c) for (p,r), c in totals.items()]
    out.mkdir(parents=True, exist_ok=True)
    for label, data in [('requests', rows), ('summary', summary)]:
        keys = dict.fromkeys(k for row in data for k in row)
        table(out/f'context-gates-{label}.csv', [{k:row.get(k) for k in keys} for row in data])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True, help='disjoint rollout directory; repeatable')
    parser.add_argument('--out', type=Path, required=True, help='output directory for the two CSVs')
    args = parser.parse_args()
    audit(args.run, args.out)

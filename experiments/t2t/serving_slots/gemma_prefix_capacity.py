"""CPU-only, one-tick counterfactuals on sealed fixed-cap Gemma IBS/OBS.

Run from the checkout: python -B -m experiments.t2t.serving_slots.gemma_prefix_capacity
No model, scoring code or RNG. Every decision starts from the *observed* pre-call
state. Capacities are repeated position-ticks, not distinct tokens or a rollout.
``free_slots`` keeps every original row; ``free_k1_ticks`` uses at most one.
``clean_swap`` and ``mask_swap`` are separate K1 alternatives, not additive.
They keep physical width, right edge, resident KV coverage and the first MASK.
``*_full_ticks`` restrict swaps to full calls. ``k1_*`` spends slack first,
then (where enabled) replaces an optional clean row, then at most one MASK.
``mask_shrink_only_full_ticks`` is extra capacity if a shorter span is allowed.
The donor's historical admission is a retrospective conflict count, not a score.
Decoder stages use the observed selection: mask_work vs replay_only (no MASK).
Whole-request work/metrics appear only on stage=all; encoder prefill is not
generation, but inter-canvas B256 encoder calls are billed in F and Q.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from math import isclose
from pathlib import Path

from bdlm_serving.runtime.optimus_limits import validate_trace
from experiments.t2t.serving_slots.ibs_prefix_opportunity import (
    COHORTS, IDENTITY, load_manifest, ownership, read_json, require, sha256_file,
    verify_completion_marker,
)

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[3] / 't2t-native-1125'
CONTROLS = HERE / 'controls'
OUTPUT = CONTROLS / 'gemma-prefix-capacity.csv'
B, MAX_GEN = 256, 2048
MANIFEST = 'bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed'
# Each pinned run manifest identifies the checkpoint, controller hashes, canvas,
# M2T-only admission, decoder cap clock, and the separately billed encoder.
ARCHIVES = (
    ('prefix90/gemma-fixed-caps-q32q64-v1', (32, 64),
     'c390d5e7c02ecc608592448e19cb9ab8d2125241be38d1ee59cd728716277855',
     'gemma-fixed-caps-paired.csv', 'cf3fd5f2de2e5c0df8d4b935400831e9911236470be1ec477d8e033bd65a9ce9'),
    ('private-optimus-cap90-v1/gemma-optimus-cap90-v1', (128, 256),
     '8675ed5de76a4b24e70dac0213a1e275840f4fa3bee7a8e2fa16d5845b187b9a',
     'gemma-original-vs-capped-fixed.csv', 'b27ec916ac09267ba0ba51f4f6bdcc764f6e7168e94f761369da270578e21f8a'),
)
STAGES = ('all', 'mask_work', 'replay_only')


def seats(start, before, picked, resolved, dirty, seen, cap):
    """Pre-call selection only. No observed admitted/ready-after or future logits."""
    edge = max(picked)
    prefix = set(range(start, start + before))
    require(prefix <= (resolved & seen) - dirty, 'unclean pre-call frontier')
    pool = prefix - picked
    require(not pool or max(pool) <= edge, 'prefix outside resident view')
    masks = picked - resolved
    # Dropping a first visit in the interior leaves an uninitialized KV hole.
    # Keeping the original rightmost row also preserves the native visible span.
    clean = (picked & resolved & seen) - dirty - prefix - {edge}
    donors = masks - {min(masks)} if masks else set()
    interior = (donors & seen) - {edge}
    slack = cap - len(picked)
    free = min(slack, len(pool))
    mask_swap = bool(pool and interior)
    clean_swap = bool(pool and clean)
    # Alternative bound: a rightmost MASK can be removed even on its first
    # visit if we let the span shrink. Never include it in fixed-span counts.
    shrink_only = bool(pool and edge in donors and not interior)
    full = slack == 0
    stats = Counter(decoder_F=1, decoder_Q=len(picked), slack_rows=slack,
                    prefix_ticks=int(bool(pool)), prefix_pool_rows=len(pool),
                    selected_mask_rows=len(masks), full_ticks=int(full),
                    free_slots=free, free_quarter_slots=min(free, cap // 4),
                    free_k1_ticks=int(free > 0), clean_swap_ticks=int(clean_swap),
                    mask_swap_ticks=int(mask_swap),
                    clean_swap_full_ticks=int(clean_swap and full),
                    mask_swap_full_ticks=int(mask_swap and full),
                    mask_shrink_only_full_ticks=int(shrink_only and full),
                    k1_no_mask_ticks=int(bool(free or clean_swap)),
                    k1_any_ticks=int(bool(free or clean_swap or mask_swap)),
                    k1_mask_only_full_ticks=int(full and mask_swap and not clean_swap))
    donor = max(interior) if mask_swap else None
    return stats, donor


def replay(row, cap):
    """Prove the frontier and event order as well as canonical physical billing."""
    require(validate_trace(row) == cap, 'not the fixed cap')
    events = row['events']
    require(events[0][1] > 0, 'empty encoder prefill')
    expected = [events[0]]
    block, step, ready = -1, -1, 0
    resolved, dirty, seen = set(), set(), set()
    totals = {key: Counter() for key in STAGES}
    for s in row['trace']:
        b = s['block']
        if b != block:
            require(b == block + 1 and (block == -1 or ready >= B),
                    'unexpected block slide or untraced block')
            if block >= 0:
                expected.append(['encoder', B])
            # OBS primes from the previous pre-admission, clean shifted tail.
            ready, step, block = max(0, ready - B), -1, b
        step += 1
        require(s['step'] == step and 0 <= ready < B, 'wrong pre-call clock')
        start = b * B
        extent = min(MAX_GEN, start + B + (cap if row['policy'].startswith('obs_') else 0))
        picked, admitted = set(s['positions']), set(s['admitted'])
        require(s['positions'] == sorted(picked), 'unsorted or duplicate physical rows')
        expected.append(['decoder', len(picked)])
        stage = 'mask_work' if picked - resolved else 'replay_only'
        require(bool(admitted) == (stage == 'mask_work'), 'MASK progress mismatch')
        counts, donor = seats(start, ready, picked, resolved, dirty, seen, cap)
        counts['mask_swap_full_donor_admitted'] = int(counts['mask_swap_full_ticks'] and donor in admitted)
        counts['k1_mask_only_donor_admitted'] = int(counts['k1_mask_only_full_ticks'] and donor in admitted)
        counts['k1_mask_only_donor_obs_tail'] = int(counts['k1_mask_only_full_ticks'] and donor >= start + B)
        counts['inherited_prefix_first_ticks'] = int(step == 0 and ready > 0)
        dirty -= picked
        # The actual controller advances on prior resolved tokens after physical
        # dirty replay, BEFORE recording any new admissions from this forward.
        while start + ready < extent and start + ready in resolved and start + ready not in dirty:
            ready += 1
        require(s['ready'] == ready, 'recorded ready disagrees with clean pre-admission state')
        counts['free_slots_on_retire'] = counts['free_slots'] if ready >= B else 0
        totals[stage].update(counts)
        totals['all'].update(counts)
        dirty |= admitted
        resolved |= admitted
        seen |= picked
    require(ready >= B and expected == events, 'retirement or interleaved event order mismatch')
    require(block * B < len(row['output_ids']) <= (block + 1) * B, 'output/retired block mismatch')
    all_work = totals['all']
    f, q, n = row['generation_forwards'], row['generation_query_rows'], row['output_tokens']
    require(0 <= n <= len(row['output_ids']) and isclose(row['tpf'], n / f)
            and isclose(row['token_efficiency'], n / q), 'baseline N/F/Q mismatch')
    all_work.update(dict(generation_N=n, generation_F=f, generation_Q=q, encoder_gen_F=block,
                         encoder_gen_Q=block * B, encoder_prefill_F=1, encoder_prefill_Q=events[0][1],
                         request_tpf_sum=n / f))
    for c in totals.values():
        for key in ('prefix_ticks', 'free_k1_ticks', 'clean_swap_full_ticks', 'mask_swap_full_ticks',
                    'k1_no_mask_ticks', 'k1_any_ticks'):
            c[key.removesuffix('_ticks') + '_requests'] = int(c[key] > 0)
        c['requests_with_stage'] = int(c['decoder_F'] > 0)
    return totals


def verified_rows(folder, policy, tasks, owner):
    require(ownership(folder) == owner, 'wrong archive owner')
    for name in ('run.json', 'qualified-logits.json', f'complete-{policy}.json'):
        require(ownership(folder / name) == owner, f'wrong owner: {name}')
    verify_completion_marker(folder, policy)
    expected = {f'{idx:02d}-{policy}.json' for idx in tasks}
    actual = {f.name for f in folder.glob(f'*-{policy}.json') if f.name.split('-')[0].isdigit()}
    require(expected == actual, f'{policy}: wrong request inventory')
    for idx, task in tasks.items():
        path = folder / f'{idx:02d}-{policy}.json'
        require(ownership(path) == owner, f'wrong owner: {path}')
        raw = path.read_bytes()
        row = json.loads(raw)
        require(row['model'] == 'gemma' and row['policy'] == policy and row['status'] == 'success'
                and all(row[k] == task[k] for k in IDENTITY), f'wrong frozen request: {path}')
        yield idx, row, hashlib.sha256(raw).digest()


def baseline(c, archive, size):
    """Join previously scored baseline only, cross-checked against this ledger."""
    require(int(archive['requests']) == size and
            [int(archive['arm_' + key]) for key in ('N', 'F', 'Q', 'physical_Q')] ==
            [c['generation_N'], c['generation_F'], c['generation_Q'], c['generation_Q']],
            'archived score row not paired with this work')
    tpf = c.pop('request_tpf_sum') / size
    efficiency = c['generation_N'] / c['generation_Q']
    correct = int(archive['arm_correct'])
    require(isclose(float(archive['arm_macro_tpf']), tpf) and
            isclose(float(archive['arm_pooled_token_efficiency']), efficiency) and
            0 <= correct <= size and isclose(float(archive['arm_quality']), correct / size),
            'archived baseline metric mismatch')
    return dict(baseline_correct=correct, baseline_quality=correct / size,
                baseline_macro_tpf=tpf, baseline_pooled_efficiency=efficiency)


def analyze(data=DATA, controls=CONTROLS):
    digest, tasks, _ = load_manifest(data / 'prefix90')
    require(digest == MANIFEST, 'wrong frozen90 manifest')
    owner = ownership(data / 'prefix90/samples.json')
    output = []
    for directory, caps, run_sha, scores, scores_sha in ARCHIVES:
        folder = data / directory
        require(sha256_file(folder / 'run.json') == run_sha, 'wrong run fingerprint')
        meta = read_json(folder / 'run.json')
        require(meta['samples_sha256'] == MANIFEST and meta['hard_q_cap'] is True and
                meta['cap_clock'] == 'all_decoder_forwards_request_local' and meta['cap_excludes'] == 'encoder',
                'invalid frozen hard-cap semantics')
        require(read_json(folder / 'qualified-logits.json') ==
                dict(modes=['ibs', 'obs'], chunk=B, first_step_logits='bit_identical', index=0),
                'missing frozen logits qualification')
        require(sha256_file(controls / scores) == scores_sha, 'archived baseline table changed')
        with (controls / scores).open(newline='') as handle:
            refs = {(r['arm_policy'], int(r['window'])): r for r in csv.DictReader(handle)
                    if r['model'] == 'gemma' and r['scope'] == 'all'}
        for cap in caps:
            for mode in ('ibs', 'obs'):
                policy = f'{mode}_cap_{cap}'
                require(policy in meta['policies'] and [cap] in meta['capped_plans'], 'wrong policy owner')
                sums = {(name, stage): Counter() for name, _, _ in COHORTS for stage in STAGES}
                fingerprint = hashlib.sha256()
                for idx, row, row_hash in verified_rows(folder, policy, tasks, owner):
                    fingerprint.update(row_hash)
                    try:
                        request = replay(row, cap)
                    except ValueError as exc:
                        raise ValueError(f'{policy}/{idx}: {exc}') from exc
                    for name, lo, hi in COHORTS:
                        if lo <= idx < hi:
                            for stage in STAGES:
                                sums[name, stage].update(request[stage])
                for name, lo, hi in COHORTS:
                    size = hi - lo
                    for stage in STAGES:
                        c = sums[name, stage]
                        metrics = baseline(c, refs[policy, size], size) if stage == 'all' else {}
                        output.append(dict(policy=policy, cap=cap, cohort=name, requests=size,
                                           stage=stage, **c, **metrics, archive=directory,
                                           rows_sha256=fingerprint.hexdigest(), baseline_source=scores))
    return output


def main():
    rows = analyze()  # Validate the entire matrix before opening the aggregate.
    with OUTPUT.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)),
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f'{OUTPUT}: {len(rows)} rows; 8 sealed arms x 90 identities')


if __name__ == '__main__':
    main()

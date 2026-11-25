"""Independent Q8/E1 physical swaps on teacher-forced LLaDA B32 states."""
import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from fractions import Fraction
import gzip
import json
from math import log
from pathlib import Path

import numpy as np

from bdlm_serving.evaluation.frozen import MANIFEST, ROOT, sha
from bdlm_serving.methods.t2t.embedding_context import (
    CALIBRATION, ContextRanker, Projector, input_table, read_context_weights,
)
from bdlm_serving.methods.t2t.hazard import features
from tools.lexical_hazard import check_split, write_rows


STUDY = Path(__file__).resolve().parent
RAW = ROOT.parent / 't2t-native-1125'
Q, WIDTH = 8, 32
MODES = ('fixed', 'slack_only', 'gate5pct')
CUTOFF = log(.05 / .95)
FIT_SHA = 'dbcaf8d1496bbd4867a9ad9f44e592bec520c14872c1cb6476f257521d03081d'
QUALIFIED = dict(model='llada21', full_width=32, matched=30)
IDENTITY = ('index', 'dataset', 'task_id', 'seed')


@dataclass(frozen=True)
class Space:
    fresh: frozenset
    masks: tuple
    old: tuple
    base: frozenset

    @classmethod
    def build(cls, valid, fresh, masks):
        fresh, masks, valid = frozenset(fresh), frozenset(masks), frozenset(valid)
        if fresh & masks or not (fresh | masks) <= valid or not valid <= set(range(WIDTH)):
            raise ValueError('Invalid editable partition')
        masks, old = tuple(sorted(masks)), tuple(sorted(valid - fresh - masks))
        base = fresh.union((masks + old)[:max(Q - len(fresh), 0)])
        return cls(fresh, masks, old, base)

    def plans(self, scores):
        scores = np.asarray(scores)
        if scores.shape != (len(self.old),) or not np.isfinite(scores).all():
            raise ValueError('Scores do not match eligible old rows')
        top = self.old[np.argmax(scores)] if self.old else None  # ascending row breaks ties
        old_fill = [i for i in self.old if i in self.base]
        taken_masks = [i for i in self.masks if i in self.base]
        victim = taken_masks[-1] if len(taken_masks) >= 2 else None
        if old_fill:
            victim = old_fill[-1]
        allowed = top is not None and victim is not None
        plans = {}
        for mode in MODES:
            swap = allowed and top not in self.base and (
                bool(old_fill) or mode == 'fixed' or (mode == 'gate5pct' and scores.max() >= CUTOFF))
            plans[mode] = self.base.difference([victim]).union([top]) if swap else self.base
        return top, allowed, plans


def counts(space, picked, targets, admits):
    base = space.base
    outside = set(space.old) - base
    enter, leave = picked - base, base - picked
    protected = space.fresh.union(space.masks[:1] if space.masks and space.masks[0] in base else ())
    if (not targets <= set(space.old) or not admits <= set(space.masks) or
            not protected <= picked or not enter <= outside or
            len(enter) != len(leave) or len(enter) > 1):
        raise ValueError('Label, protection or physical swap space mismatch')
    added, lost = len(enter & targets), len(leave & targets)
    mask_loss = len(leave & admits)
    displaced = len(leave.intersection(space.masks))
    # Conditional on this gate actually changing the set; retain its victim.
    uniform = Fraction(len(outside & targets), len(outside)) if enter else Fraction()
    return dict(states=1, physical_q=len(picked), mandatory_fresh=len(space.fresh),
                eligible_old=len(space.old), old_targets=len(targets), base_old_hits=len(base & targets),
                ranked_old_hits_added=added, old_hits_lost=lost, ranked_net_old_hits=added-lost,
                ranked_selected_old_hits=len(picked & targets), uniform_expected_old_hits_added=uniform,
                uniform_expected_net_old_hits=uniform-lost,
                uniform_expected_selected_old_hits=len(base & targets)+uniform-lost,
                physical_swaps=len(enter), displaced_old=len(leave)-displaced, displaced_masks=displaced,
                natural_mask_admissions=len(admits), base_mask_admissions=len(base & admits),
                lost_mask_admissions=mask_loss, displaced_nonadmitting_masks=displaced-mask_loss,
                selected_mask_admissions=len(picked & admits))


def replay(rows, task, ranker, mask, vocab_size):
    meta, prompt = next(rows), next(rows)
    if (meta['schema'] != 1 or any(meta[k] != task[k] for k in IDENTITY) or
            prompt['mask_id'] != mask or not 0 <= mask < vocab_size or prompt['prompt_length'] < 0):
        raise ValueError('Unexpected trace header or request space')
    plen = prompt['prompt_length']
    totals = {mode: Counter() for mode in MODES}
    last = prior = None
    for row in rows:
        block, before = row['block'], row['before']
        bootstrap = last != block
        if bootstrap:
            if block != (plen // WIDTH * WIDTH if last is None else last + WIDTH):
                raise ValueError('Invalid native block sequence')
            valid = range(max(0, plen-block), WIDTH)
            fresh, strict, writes, clock = set(), set(), [-1]*WIDTH, 0
        elif before != prior:
            raise ValueError('Native before/after discontinuity')
        clock += 1
        if row['clock'] != clock or len(before) != WIDTH:
            raise ValueError('Invalid native B32 clock or width')
        masks = {i for i in valid if before[i] == mask}
        selected = not bootstrap and bool(masks)
        if selected:
            space = Space.build(valid, fresh, masks)
            # The ranker receives only pre-step IDs on eligible old and fresh rows.
            visible = {i: before[i] for i in (*space.old, *fresh)}
            pos = [features(i, space.masks, fresh, strict, writes, clock) for i in space.old]
            scores = ranker.scores(visible, fresh, space.old, pos) if space.old else ()
            top, allowed, plans = space.plans(scores)

        after = row['after']
        if (len(after) != WIDTH or any(type(x) is not int or not 0 <= x < vocab_size for x in before+after)
                or before[:valid.start] != after[:valid.start]
                or row['accepted'] != [x != mask for x in after]):
            raise ValueError('Unexpected token space, prompt write or acceptance mask')
        changed = {i for i in valid if before[i] != after[i]}
        written = {i for i in changed if after[i] != mask}
        admits, edits = written & masks, written - masks
        if selected:
            targets = edits - fresh  # Exclude mandatory repeats, unchanged and revoked IDs.
            for mode, picked in plans.items():
                totals[mode].update(counts(space, picked, targets, admits))
                totals[mode].update(top_in_base_noop=int(top in space.base),
                                    old_pool_states=int(top is not None), top1_old_hits=int(top in targets),
                                    fixed_rankable_states=int(allowed), fixed_rankable_top_hits=int(allowed and top in targets))
        for i in changed:
            writes[i] = clock
        fresh, strict, prior, last = written, edits, after, block
    if last is None:
        raise ValueError('Missing native denoising steps')
    return totals


def qualify(native, source_run, weights):
    meta = json.loads((source_run / 'run.json').read_text())
    arm = dict(budget=8, fresh=0, mask=7, selection='hard_context_fixed')
    if (sha(weights) != FIT_SHA or meta.get('context_weights_sha256') != FIT_SHA or
            json.loads((native / 'verified.json').read_text()) != dict(model='llada21', matched_requests=30) or
            json.loads((source_run / 'qualified.json').read_text()) != QUALIFIED or
            any(meta.get(k) != v for k, v in dict(model='llada21', batch_size=1, block=32, cap=2048, seed_offset=0).items()) or
            arm not in meta['arms'] or 't2t_q8_f0_m7_hard_context_fixed' not in meta['policies']):
        raise ValueError('Unqualified native traces, source run or calibration24 coefficients')
    return read_context_weights(weights)


def mask_token(checkpoint):
    known = json.loads((ROOT / 'benchmarks/checkpoints/llada21.json').read_text())
    path = checkpoint / 'tokenizer.json'
    if sha(path) != next(r['sha256'] for r in known if r['name'] == path.name):
        raise ValueError('Unexpected native tokenizer space')
    (mask,) = [item['id'] for item in json.loads(path.read_text())['added_tokens'] if item['content'] == '<|mask|>']
    return mask


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def public_row(mode, total, fields, **key):
    # Counter supplies explicit zeros even for a request with no selected states.
    return dict(**key, q=Q, e=1, mode=mode,
                **{k: float(total[k]) if isinstance(total[k], Fraction) else total[k] for k in fields})


def run(native, source_run, weights, checkpoint):
    tasks = json.loads(MANIFEST.read_text())['samples']
    check_split(json.loads(CALIBRATION.read_text())['samples'], tasks)
    w = qualify(native, source_run, weights)
    mask = mask_token(checkpoint)
    archived_base = [r for r in read_csv(STUDY.parents[1] / 'exchange/ceiling/requests.csv')
                     if r['policy'] == 'zero_reserve' and int(r['q']) == Q]
    summary = {mode: Counter() for mode in MODES}
    requests = []
    with input_table(checkpoint) as table:
        projector = Projector(table)
        ranker = ContextRanker(projector.unit, w)
        for task, ref in zip(tasks, archived_base, strict=True):
            with gzip.open(native / f"{task['index']:02d}.trace.jsonl.gz", 'rt') as f:
                totals = replay(map(json.loads, f), task, ranker, mask, projector.size)
            if (any(str(task[k]) != ref[k] for k in IDENTITY) or
                    any(totals['fixed'][k] != int(ref[k]) for k in ('states', 'physical_q', 'mandatory_fresh',
                        'old_targets', 'base_old_hits', 'natural_mask_admissions', 'base_mask_admissions'))):
                raise ValueError(f"Archived Q8 base space mismatch for request {task['index']}")
            for mode, total in totals.items():
                requests.append((task['index'], task['dataset'], mode, total))
                summary[mode].update(total)

    # Validate both the unconstrained old pool and original *nominal* fixed rankable space.
    comparison = {r['metric']: r for r in read_csv(STUDY / 'comparison.csv')}
    for name, seats, hits in [('top-1', 'old_pool_states', 'top1_old_hits'),
                              ('Q8/1', 'fixed_rankable_states', 'fixed_rankable_top_hits')]:
        ref, seen = comparison[name], summary['fixed']
        if any(seen[k] != int(ref[c]) for k, c in [(seats, 'seats'), (hits, 'context_hits'), ('old_targets', 'targets')]):
            raise ValueError(f'Archived {name} ranking space mismatch')
    out = STUDY / 'gated'
    out.mkdir(exist_ok=True)
    fields = tuple(summary['fixed'])
    write_rows(out / 'native_q8e1_requests.csv', [public_row(m, s, fields, request_index=i, dataset=d)
                                                  for i, d, m, s in requests])
    write_rows(out / 'native_q8e1_summary.csv', [public_row(m, s, fields, requests=len(tasks)) for m, s in summary.items()])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native', type=Path, default=RAW / 'llada21')
    p.add_argument('--source-run', type=Path, default=RAW / 'context-fixed-q4q8-v1')
    p.add_argument('--weights', type=Path, default=RAW / 'context-fixed-private/calibration24.json')
    p.add_argument('--checkpoint', type=Path, default=Path('/data0/ysy/models/LLaDA2.1-mini'))
    run(**vars(p.parse_args()))

"""Opt-in fixed sparse Gemma ablation. This changes native renoise semantics.

The ordinary draw is already in baseline_ids; never draw again or advance the
generator conditionally. SC membership means *previously sampled in this
block*, not merely projected. Newly revoked positions receive normal noise.
"""
from collections import Counter

import torch


class RejectNoise:
    def __init__(self, enabled, part):
        self.enabled, self.part = enabled, part

    def __call__(self, rows, seen, ids, old, accepted, baseline_ids):
        warm = torch.tensor([[r in seen for r in rows]], device=ids.device)
        eligible = warm & ~old & ~accepted
        def positions(mask):
            return [r for r,v in zip(rows,mask[0].tolist(),strict=True) if v]
        self.part['renoise'] = dict(eligible=positions(eligible), baseline_changed=positions(ids!=baseline_ids))
        return torch.where(eligible, ids, baseline_ids) if self.enabled else baseline_ids


def noise_receipt(trace, enabled):
    """Independent replay of sampled membership, eligibility, writes and debt."""
    totals = Counter()
    first = None
    seen, accepted, debt = set(), set(), set()
    ordinal = None
    for tick,e in enumerate(trace):
        if len(e['parts'])!=1 or e['tick']!=tick or e['route'] not in ('sparse','drain','verify'):
            raise ValueError('Not a head-only unfused control')
        p = e['parts'][0]
        if ordinal != e['ordinal']:
            if debt:
                raise ValueError('Unpaid debt at next block')
            seen, accepted, ordinal = set(), set(), e['ordinal']
        if p['ordinal']!=ordinal or set(p['dirty_before'])!=debt:
            raise ValueError('Changed head or missing debt')
        rows, selected, adds, revokes, changed = (set(p[k]) for k in
            ('rows','sampled','admitted','revoked','input_changed'))
        after = (accepted|adds)-revokes
        if selected:
            mark = p['renoise']
            eligible = (seen & selected) - accepted - after
            base = set(mark['baseline_changed'])
            avoidable = base & eligible
            if (set(mark['eligible'])!=eligible or not base<=selected
                    or changed!=(base-eligible if enabled else base)):
                raise ValueError('Noise eligibility or actual token writes do not reconcile')
            if first is None and avoidable:
                first = dict(tick=tick,ordinal=ordinal,rows=sorted(avoidable))
            totals.update(sample_ticks=1,sampled_rows=len(selected),warm_rows=len(selected&seen),
                          eligible=len(eligible),would_change=len(avoidable),
                          avoided=len(avoidable) if enabled else 0,baseline_token_changes=len(base),
                          token_changes=len(changed),acceptance_flips=len(adds|revokes),
                          dirty_writes=len(changed|adds|revokes))
        elif 'renoise' in p or adds or revokes or changed:
            raise ValueError('Sampling/write on a non-sampling step')
        if p['phase']=='drain':
            totals.update(drain_F=1, drain_Q=len(rows))
        if p['phase']=='verify' and debt:
            raise ValueError('Verify before all writes paid')
        debt = (debt-rows)|changed|adds|revokes
        if debt != set(p['dirty_after']):
            raise ValueError('Dirty ledger disagrees with actual tokens or flips')
        seen.update(selected)
        accepted = after
    if debt or not trace or trace[-1]['route']!='verify':
        raise ValueError('No debt-free terminal check')
    return {**{k:totals[k] for k in ('sample_ticks','sampled_rows','warm_rows','eligible','would_change',
              'avoided','baseline_token_changes','token_changes','acceptance_flips','dirty_writes',
              'drain_F','drain_Q')}, 'first_exposure':first}

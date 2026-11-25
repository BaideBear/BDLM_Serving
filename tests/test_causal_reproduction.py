"""Historical equality and row-observer validation, independent of the decoder."""
from copy import deepcopy
import csv
import json

import pytest

from tools import causal_seat_report as report


HASH = '1' * 64
FLAGS = dict(victim_queried=False, victim_unresolved_before=True, victim_accepted=False)


def pair(llada=False):
    # Both anchors have origin 20 in output coordinates. LLaDA uses block number,
    # Gemma uses absolute prefix, so neither can use the other's mapping.
    block, plen = (2, 44) if llada else (64, 44)

    def event(stage, positions, masked=0, admits=0, edits=0):
        e = dict(block=block, stage=stage, trajectory_sha256=HASH, accepted_sha256=HASH,
                 hard_strict_edits=edits)
        if llada:
            e.update(physical_q=len(positions), actual_q=len(positions), layer_physical_q=[len(positions)]*20,
                     absolute_positions=[block*32+i for i in positions],
                     selected_masked_before=masked, accepted=admits, edited=edits,
                     masked_before=masked, masked_after=masked-admits)
        else:
            e.update(q=len(positions), positions=positions, selected_mask=masked, admitted=admits,
                     revoked=0, edited=edits)
        return e

    path = [event('bootstrap', [0, 1, 2]), event('selected', [0, 2], 1, 1),
            event('selected', [0], 0), event('verify', [0, 1, 2], 1, 1, 1)]
    plan = dict(base=[0, 1], masks=[0, 1], executed=[0, 2], victim=1, stratum='mask',
                pre_sha256=HASH, after_sha256=HASH, post_rng_sha256=HASH, after_admissions=1)
    path[1]['intervention'] = plan
    if llada:
        path.append(event('commit', [0, 1, 2]))
    path.append(dict(path[0], block=block+(1 if llada else 256)))
    ids = [8777]*20+[7002, 7003]
    raw = dict(status='success', model='llada21' if llada else 'gemma', events=path, output_ids=ids,
               output_tokens=22, generation_forwards=4, generation_query_rows=11,
               generation_physical_padded_rows=11, tpf=22/4, token_efficiency=2,
               physical_events=[] if llada else [['encoder', plen]] + [['decoder', e['q']] for e in path])
    new = deepcopy(raw)
    for i, vals in enumerate(([7001, None], [7001, None], [7002, 7003]), 1):
        new['events'][i]['beneficiary'] = dict(rows=[0, 1], output_offsets=[20, 21], ids=vals)
    new['events'][1]['intervention'].update(FLAGS)
    return new, raw, plen


@pytest.mark.parametrize('llada', [False, True])
def test_observation_window_and_semantic_flags(llada):
    new, old, plen = pair(llada)
    assert report.check_reproduction(new, old, plen) == dict(requests=1, anchors=1, watched_rows=2,
        observations=3, row_observations=6, anchor_admissions=1)
    old['events'][1]['intervention'].update(FLAGS)
    report.check_reproduction(new, old, plen)
    del new['events'][1]['intervention']['victim_queried']
    with pytest.raises(ValueError):
        report.check_reproduction(new, old, plen)


@pytest.mark.parametrize('key', FLAGS)
def test_new_only_flag_must_be_independently_true(key):
    new, old, plen = pair()
    new['events'][1]['intervention'][key] = not FLAGS[key]
    with pytest.raises(ValueError, match=key):
        report.check_reproduction(new, old, plen)


@pytest.mark.parametrize('idx', [0, 1, 2, 3, 4, 5])
def test_exact_presence_including_commit_and_next_block(idx):
    new, old, plen = pair(True)
    e = new['events'][idx]
    if 'beneficiary' in e:
        del e['beneficiary']
    else:
        e['beneficiary'] = None
    with pytest.raises(ValueError, match='absent beneficiary'):
        report.check_reproduction(new, old, plen)


def test_no_anchor_no_telemetry():
    new, old, plen = pair()
    del old['events'][1]['intervention']
    assert report.check_reproduction(old, old, plen)['observations'] == 0
    del new['events'][1]['intervention']
    with pytest.raises(ValueError):
        report.check_reproduction(new, old, plen)


def test_empty_watch_is_not_a_missing_observation():
    new, old, plen = pair()
    for raw in (new, old):
        e = raw['events'][1]
        e.update(selected_mask=0, admitted=0)
        e['intervention'].update(masks=[], stratum='slack', after_admissions=0)
    new['events'][1]['intervention']['victim_unresolved_before'] = False
    for e in new['events'][1:4]:
        e['beneficiary'] = dict(rows=[], ids=[], output_offsets=[])
    c = report.check_reproduction(new, old, plen)
    assert c['observations'] == 3 and c['row_observations'] == 0


def test_admitted_mask_victim_and_later_revocation():
    new, old, plen = pair()
    for raw in (new, old):
        e = raw['events'][1]
        e.update(positions=[0, 1], selected_mask=2, admitted=2)
        e['intervention'].update(executed=[0, 1], after_admissions=2)
        raw['events'][2].update(positions=[1], revoked=1)
    new['events'][1]['intervention'].update(victim_queried=True, victim_accepted=True)
    new['events'][1]['beneficiary']['ids'][1] = 7009
    report.check_reproduction(new, old, plen)
    new['events'][1]['intervention']['victim_accepted'] = False
    with pytest.raises(ValueError, match='victim_accepted'):
        report.check_reproduction(new, old, plen)


@pytest.mark.parametrize('mutate', [
    lambda n: n['output_ids'].reverse(),
    lambda n: n['physical_events'].append(['decoder', 2]),
    lambda n: n['events'][0].update(unknown=False),
    lambda n: n['events'][1]['intervention'].update(post_rng_sha256='2'*64),
    lambda n: n['events'][1]['intervention'].update(new_flag=False),
    lambda n: n['events'][1]['beneficiary'].update(extra=None),
    lambda n: n['events'][2]['beneficiary'].update(rows=[0, 2]),
    lambda n: n['events'][2]['beneficiary'].update(output_offsets=[64, 65]),
    lambda n: n['events'][1]['beneficiary'].update(ids=[None, 7001]),  # same sum, skipped victim
    lambda n: n['events'][1]['beneficiary'].update(ids=[None, None]),
    lambda n: n['events'][2]['beneficiary'].update(ids=[7001, 7003]),  # off-Q admission later
    lambda n: n['events'][2]['beneficiary'].update(ids=[7003, None]),  # on-Q but no strict edit
    lambda n: n['events'][2]['beneficiary'].update(ids=[True, None]),
    lambda n: n['events'][3].update(q=3.0),  # numerical equality is not JSON type equality
])
def test_non_allowlisted_change_is_fatal(mutate):
    new, old, plen = pair()
    mutate(new)
    with pytest.raises(ValueError):
        report.check_reproduction(new, old, plen)


def test_llada_final_canvas_uses_emitted_ids_but_gemma_does_not():
    for llada in (True, False):
        new, old, plen = pair(llada)
        new['events'][3]['beneficiary']['ids'][0] = 4444
        if llada:
            with pytest.raises(ValueError, match='emitted'):
                report.check_reproduction(new, old, plen)
        else:
            report.check_reproduction(new, old, plen)


def dump(path, obj):
    path.write_text(json.dumps(obj))


def test_native32_identity_bridge_and_exact_non_aligned_length(tmp_path):
    new, old, plen = pair(True)
    task = dict(index=0, seed=123, task_id='x', dataset='x')
    full = dict(old, **task, policy='fullwidth', max_q=32, prefill_rows=32,
                events=[dict(phase='denoise', block=1, masked_before=20)])
    ref = dict(full, policy='native32', prompt_ids=[9]*plen, generation_cap=2048, phase='formal')
    p = tmp_path/'00-native32.json'
    dump(p, ref)
    assert report._prompt_length(full, task, tmp_path) == 44
    for changed in (dict(prompt_ids=[9]*43), dict(output_ids=[]), dict(seed=0), dict(policy='native256')):
        dump(p, dict(ref, **changed))
        with pytest.raises(ValueError):
            report._prompt_length(full, task, tmp_path)
    with pytest.raises(ValueError):
        report._prompt_length(full, task, None)


def cohort(root):
    old_dir, new_dir = root/'historical', root/'new'
    arms = [dict(budget=32, mask=32, fresh=0, selection='hard_once_'+m) for m in report.MODES]
    policies = ['t2t_q32_f0_m32_'+a['selection'] for a in arms]
    meta = dict(model='gemma', policies=policies, arms=arms, cap=2048, block=256, batch_size=1)
    for d in (old_dir, new_dir):
        d.mkdir()
        dump(d/'run.json', meta if d == old_dir else dict(meta, seed_offset=0))
        dump(d/'qualified.json', dict(matched=30, full_width=256, model='gemma'))
        for p in policies:
            dump(d/f'complete-{p}.json', dict(policy=p, requests=30))
    for s in report._json(report.MANIFEST)['samples']:
        ident = {k:s[k] for k in ('index', 'dataset', 'task_id', 'seed')}
        new, old, plen = pair()
        for d, row in ((old_dir, old), (new_dir, new)):
            raw = dict(row, **ident, prefill_rows=plen)
            full = dict(raw, policy='fullwidth', events=[dict(block=plen)], physical_events=[['encoder', plen]])
            dump(d/f"{s['index']:02d}-fullwidth.json", full)
            for p in policies:
                dump(d/f"{s['index']:02d}-{p}.json", dict(raw, policy=p))
    return new_dir, old_dir, policies


def test_public_csv_is_policy_counts_only(tmp_path):
    new, old, policies = cohort(tmp_path)
    out = tmp_path/'out'
    report.reproduce(new, old, out)
    f, = out.iterdir()
    assert b'\r' not in f.read_bytes()
    with f.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r.pop('policy') for r in rows] == policies
    assert rows == [dict(requests='30', anchors='30', watched_rows='60', observations='90',
                        row_observations='180', anchor_admissions='30')] * 3


@pytest.mark.parametrize('problem', ['unqualified', 'incomplete-old', 'missing-both', 'wrong-policy',
                                     'same-wrong-seed', 'config-drift', 'nonzero-offset', 'fullwidth-drift'])
def test_reject_cohort_before_publishing(tmp_path, problem):
    new, old, policies = cohort(tmp_path)
    if problem == 'unqualified':
        dump(new/'qualified.json', dict(matched=29, full_width=256, model='gemma'))
    elif problem == 'incomplete-old':
        (old/f'complete-{policies[0]}.json').unlink()
    elif problem == 'fullwidth-drift':
        p = new/'29-fullwidth.json'
        dump(p, dict(report._json(p), unknown=None))
    elif problem == 'config-drift':
        dump(new/'run.json', dict(report._json(new/'run.json'), new_option=1))
    else:
        for d in (old, new):
            p = d/f'29-{policies[0]}.json'
            if problem == 'missing-both':
                p.unlink()
            elif problem == 'same-wrong-seed':
                dump(p, dict(report._json(p), seed=1))
            else:
                m = report._json(d/'run.json')
                if problem == 'wrong-policy':
                    m['policies'][0] = 'other'
                else:
                    m['seed_offset'] = 1
                dump(d/'run.json', m)
    out = tmp_path/'out'
    with pytest.raises((ValueError, FileNotFoundError)):
        report.reproduce(new, old, out)
    assert not out.exists()

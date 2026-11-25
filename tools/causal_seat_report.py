"""Verify first-opportunity one-seat interventions against measured physical events."""
import argparse
import csv
import json
import re
from collections import Counter
from itertools import zip_longest
from pathlib import Path

from bdlm_serving.evaluation.frozen import MANIFEST
from bdlm_serving.methods.t2t.protocol import verify_bridge, verify_work
from tools.t2t_results import table

MODES = ("base", "rank", "random")


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def first_difference(a, b, key):
    """Relative denoise step; -1 = identical series. Includes stage/block alignment."""
    def series(ev):
        return [(x["block"], x["stage"], x[key]) for x in ev if key in x]
    return next((i for i, (x, y) in enumerate(zip_longest(series(a), series(b))) if x != y), -1)


def inspect(raw):
    paths = [r["events"] for r in raw]
    anchors = [[j for j, e in enumerate(ev) if "intervention" in e] for ev in paths]
    idx = raw[0]["index"]
    require(anchors[0] == anchors[1] == anchors[2] and len(anchors[0]) <= 1,
            f"Expected zero or one identical anchor index, task {idx}")
    anchor = anchors[0][0] if anchors[0] else -1
    plans = []
    keys = ("base","targeted","random","victim","candidate","random_candidate","fresh","masks",
            "stratum","clock","pre_sha256","post_rng_sha256")
    for t, path in enumerate(paths):
        require(raw[t]["status"] == "success" and all(raw[t][k] == raw[0][k] for k in
            ("index","dataset","seed","task_id","model")), f"Request identity mismatch on {idx}")
        if anchor < 0:
            require(path == paths[0] and raw[t]["output_ids"] == raw[0]["output_ids"] and all(
                raw[t][k] == raw[0][k] for k in ("generation_forwards", "generation_query_rows")),
                f"Unmatched no-op on {idx}")
            continue
        require(path[:anchor] == paths[0][:anchor], f"Prefixes differ on {idx}")
        e = path[anchor]
        p = e["intervention"]
        plans.append(p)
        require(all(p[k] == plans[0][k] for k in keys), f"Unmatched plan, pre-state or RNG on {idx}")
        require(all(len(p[k]) == 64 for k in ("pre_sha256", "post_rng_sha256", "after_sha256"))
                and p["after_sha256"] == e["trajectory_sha256"], "Missing state evidence")
        expected = p[("base", "targeted", "random")[t]]
        actual = e["positions"] if "positions" in e else [i-32*e["block"] for i in e["absolute_positions"]]
        base, picked = set(p["base"]), set(expected)
        masks, fresh = set(p["masks"]), set(p["fresh"])
        require(e["stage"] == "selected" and sorted(picked) == actual == p["executed"]
                and len(base) == len(expected) == e.get("physical_q", e.get("q"))
                and fresh <= picked and (not base & masks or picked & masks),
                f"Physical Q contract broken on {idx}")
        if t:
            extra = p["candidate" if t == 1 else "random_candidate"]
            require(base-picked == {p["victim"]} and picked-base == {extra}
                    and extra not in fresh|masks and p["victim"] not in fresh
                    and ((p["victim"] in masks) == (p["stratum"] == "mask")), "Expected exactly one swap")
    if plans and plans[0]["candidate"] == plans[0]["random_candidate"]:
        require(paths[1] == paths[2] and raw[1]["output_ids"] == raw[2]["output_ids"],
                "Identical placements must replay exactly")
    return anchor, plans


def main(run, out):
    meta = json.loads((run / "run.json").read_text())
    model, policies = meta["model"], meta["policies"]
    policy = [next(s for s in policies if s.endswith("hard_once_"+m)) for m in MODES]
    marks = [json.loads((run / f"marks-{p}.json").read_text()) for p in policy]
    rows, phases = [], []
    for i in range(30):
        raw = [json.loads((run / f"{i:02d}-{p}.json").read_text()) for p in policy]
        at, plans = inspect(raw)
        for a, b in ((1,0), (2,0), (1,2)):
            x, y = raw[a], raw[b]
            start = max(at, 0)
            all_diff, acc_diff = [first_difference(x["events"][start:], y["events"][start:], k)
                for k in ("trajectory_sha256", "accepted_sha256")]
            pair = f"{MODES[a]}_vs_{MODES[b]}"
            dx = dict(model=model, index=i, dataset=x["dataset"], task_id=x["task_id"], pair=pair,
                anchor=at, stratum=plans[0]["stratum"] if plans else "none",
                delta_admissions=(plans[a]["after_admissions"]-plans[b]["after_admissions"] if plans else 0),
                first_canvas_diff=all_diff, first_accepted_diff=acc_diff,
                output_changed=int(x["output_ids"] != y["output_ids"]),
                correct=int(marks[a][str(i)]), ref_correct=int(marks[b][str(i)]),
                delta_F=x["generation_forwards"]-y["generation_forwards"],
                delta_Q=x["generation_query_rows"]-y["generation_query_rows"],
                delta_N=x["output_tokens"]-y["output_tokens"], delta_TPF=x["tpf"]-y["tpf"])
            rows.append(dx)
            for phase in sorted(x["forwards_by_phase"].keys() | y["forwards_by_phase"].keys()):
                phases.append(dict(model=model, index=i, pair=pair, phase=phase,
                    delta_F=x["forwards_by_phase"].get(phase,0)-y["forwards_by_phase"].get(phase,0),
                    delta_Q=x["q_by_phase"].get(phase,0)-y["q_by_phase"].get(phase,0)))
    out.mkdir(parents=True, exist_ok=True)
    table(out/f"{model}-causal-requests.csv", rows)
    table(out/f"{model}-causal-phases.csv", phases)
    summary = []
    for pair in sorted({x["pair"] for x in rows}):
        for group in ("all", "slack", "mask", "none"):
            rr = [r for r in rows if r["pair"] == pair and (group == "all" or r["stratum"] == group)]
            summary.append(dict(model=model,pair=pair,stratum=group,n=len(rr),
                anchors=sum(r["anchor"]>=0 for r in rr), final_changed=sum(r["output_changed"] for r in rr),
                immediate_canvas=sum(r["first_canvas_diff"]==0 for r in rr),
                immediate_accepted=sum(r["first_accepted_diff"]==0 for r in rr),
                later_accepted=sum(r["first_accepted_diff"]>0 for r in rr),
                wins=sum(r["correct"]>r["ref_correct"] for r in rr), losses=sum(r["correct"]<r["ref_correct"] for r in rr),
                **{k:sum(r[k] for r in rr) for k in ("delta_N","delta_F","delta_Q")}))
    table(out/f"{model}-causal-summary.csv", summary)


def _exact(a, b):
    # Python equality alone silently equates True, 1 and 1.0, even in lists.
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def _without(obj, key):
    return {k: v for k, v in obj.items() if k != key}


def _json(path):
    def unique(pairs):
        result = dict(pairs)
        require(len(result) == len(pairs), 'Duplicate JSON key')
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique)


def _positions(items, width):
    require(isinstance(items, list) and all(type(i) is int and 0 <= i < width for i in items)
            and len(set(items)) == len(items), 'Invalid local rows')
    return set(items)


def _snapshot(e, rows, offsets, before, picked, llada):
    obs = e['beneficiary']
    require(set(obs) == {'rows', 'ids', 'output_offsets'} and _exact(obs['rows'], rows)
            and _exact(obs['output_offsets'], offsets), 'Beneficiary rows or output offsets differ')
    ids = obs['ids']
    require(isinstance(ids, list) and len(ids) == len(rows)
            and all(i is None or (type(i) is int and i >= 0) for i in ids), 'Invalid beneficiary IDs')
    require(all(a == b or i in picked for i, a, b in zip(rows, before, ids)), 'Skipped row changed')

    strict = e['hard_strict_edits']
    if llada:
        masked = e['selected_masked_before']
        revoked = e['edited'] - strict
        admitted = e['masked_before'] - e['masked_after'] + revoked
        require(admitted <= e['accepted'] <= masked, 'Invalid LLaDA accepts')
    else:
        masked, revoked, admitted = e['selected_mask'], e['revoked'], e['admitted']
    moves = [(a, b) for i, a, b in zip(rows, before, ids) if i in picked]
    new = sum(a is None and b is not None for a, b in moves)
    lost = sum(a is not None and b is None for a, b in moves)
    edits = sum(a is not None and b is not None and a != b for a, b in moves)
    tracked_masks = sum(a is None for a, _ in moves)
    # Remaining transitions must fit the *unwatched queried* MASK and resolved seats.
    require(0 <= admitted-new <= masked-tracked_masks and lost <= revoked and edits <= strict
            and revoked-lost + strict-edits <= len(picked)-masked-(len(moves)-tracked_masks),
            'Beneficiary transitions disagree with physical event counts')
    return ids


def check_reproduction(new, old, prompt_len):
    """Compare one raw request; return counts only. Never mutate either input."""
    require(_exact(_without(new, 'events'), _without(old, 'events')), 'Historical request fields differ')
    verify_work(old)
    path, replay = old['events'], new['events']
    require(path and len(path) == len(replay), 'Historical event count differs')
    llada = old['model'] == 'llada21'
    width = 32 if llada else 256
    require(isinstance(old['output_ids'], list) and all(type(t) is int and t >= 0 for t in old['output_ids']),
            'Missing historical output IDs')
    if llada:
        require(all(e['physical_q'] == e['actual_q'] == len(e['absolute_positions'])
                    and e['layer_physical_q'] == [e['physical_q']] * 20 for e in path), 'Missing physical Q evidence')
    else:
        physical = old['physical_events']
        require(physical and physical[0] == ['encoder', prompt_len]
                and [q for phase, q in physical if phase == 'decoder'] == [e['q'] for e in path],
                'Historical decoder ledger mismatch')
    anchors = [i for i, e in enumerate(path) if 'intervention' in e]
    require(len(anchors) <= 1, 'Multiple interventions')
    at = anchors[0] if anchors else None
    if at is not None:
        plan = path[at]['intervention']
        require(path[at]['stage'] == 'selected' and all(re.fullmatch('[0-9a-f]{64}', plan[k])
                for k in ('pre_sha256', 'after_sha256', 'post_rng_sha256'))
                and plan['after_sha256'] == path[at]['trajectory_sha256'], 'Missing anchor state or RNG evidence')
    counts = Counter(requests=1, anchors=len(anchors), watched_rows=0, observations=0,
                     row_observations=0, anchor_admissions=0)
    for j, (orig, event) in enumerate(zip(path, replay)):
        stage, block = orig['stage'], orig['block']
        denoise = stage in ('bootstrap', 'selected', 'verify')
        require(denoise or (llada and stage in ('prefill', 'commit')), 'Unknown historical stage')
        if denoise:
            require(all(re.fullmatch('[0-9a-f]{64}', orig[k]) for k in ('accepted_sha256', 'trajectory_sha256')),
                    'Missing historical state evidence')
        watch = denoise and at is not None and j >= at and block == path[at]['block']
        require('beneficiary' not in orig and ('beneficiary' in event) == watch,
                f'Unexpected or absent beneficiary at event {j}')
        clean = _without(event, 'beneficiary')
        if watch:
            # Use the historical physical Q, not the requested plan, for movement checks.
            positions = ([x - width * block for x in orig['absolute_positions']]
                         if llada else orig['positions'])
            picked = _positions(positions, width)
            require(len(picked) == orig['physical_q' if llada else 'q'], 'Physical width mismatch')
            if j == at:
                masks = _positions(plan['masks'], width)
                base = _positions(plan['base'], width)
                victim = plan['victim']
                rows = sorted(base & masks)
                origin = block * width if llada else block  # Gemma block is already absolute.
                offsets = [origin + r - prompt_len for r in rows]
                require(stage == 'selected' and picked == _positions(plan['executed'], width)
                        and victim in base and plan['stratum'] == ('mask' if victim in masks else 'slack')
                        and picked & masks <= set(rows) and all(o >= 0 for o in offsets), 'Invalid anchor plan')
                before = [None] * len(rows)
                counts['watched_rows'] = len(rows)
            after = _snapshot(event, rows, offsets, before, picked, llada)
            if j == at:
                admits = sum(x is not None for x in after)
                require(admits == plan['after_admissions'], 'Anchor admissions mismatch')
                require(len(picked & masks) == orig['selected_masked_before' if llada else 'selected_mask'],
                        'Anchor selected MASK count mismatch')
                flags = dict(victim_queried=victim in picked, victim_unresolved_before=victim in masks,
                             victim_accepted=victim in rows and after[rows.index(victim)] is not None)
                cp = dict(clean['intervention'])
                for key, expected in flags.items():
                    if key in cp:
                        require(cp[key] is expected, f'Incorrect {key}')
                        if key not in plan:  # Only later instrumentation absent from v1/v2.
                            del cp[key]
                clean['intervention'] = cp
                counts['anchor_admissions'] = admits
            before = after
            counts['observations'] += 1
            counts['row_observations'] += len(rows)
        require(_exact(clean, orig), f'Historical event {j} differs')
    if llada and at is not None:
        # LLaDA commits sampled IDs. Gemma instead emits the final argmax draft.
        output = old['output_ids']
        require(all(tok is not None and (pos >= len(output) or tok == output[pos])
                    for pos, tok in zip(offsets, before)),
                'Final watched LLaDA rows differ from emitted IDs')
    return counts


def _cohort(run, historical):
    new, old = (_json(d / 'run.json') for d in (run, historical))
    # run_t2t's original-seed default predates serialization of seed_offset.
    for m in (old, new):
        m.setdefault('seed_offset', 0)
    require(_exact(new, old), 'Historical run configuration differs')
    model = old['model']
    require(model in ('llada21', 'gemma'), 'Unknown model')
    width, q = (32, 4) if model == 'llada21' else (256, 32)
    arms = [dict(budget=q, fresh=0, mask=q, selection='hard_once_' + m) for m in MODES]
    policies = [f't2t_q{q}_f0_m{q}_' + a['selection'] for a in arms]
    require(_exact([old[k] for k in ('block', 'cap', 'batch_size', 'seed_offset')], [width, 2048, 1, 0])
            and sorted(old['policies']) == sorted(policies) and len(old['arms']) == len(arms)
            and all(_exact(a, arms[policies.index(p)]) for p, a in zip(old['policies'], old['arms'])),
            'Expected original-seed canonical one-seat policies')
    for d in (run, historical):
        require(_exact(_json(d / 'qualified.json'), dict(model=model, full_width=width, matched=30)),
                'Incomplete full-width qualification')
        for p in policies:
            require(_exact(_json(d / f'complete-{p}.json'), dict(policy=p, requests=30)), 'Incomplete policy')
        expected = {f'{i:02d}-{p}.json' for i in range(30) for p in ('fullwidth', *policies)}
        require({p.name for p in d.glob('[0-9]*-*.json')} == expected, 'Incomplete or unexpected request files')
    return model, policies


def _identity(row, sample, model, policy):
    require(row['status'] == 'success' and row['model'] == model and row['policy'] == policy
            and all(_exact(row[k], sample[k]) for k in ('index', 'task_id', 'dataset', 'seed')),
            'Frozen30 request identity mismatch')


def _prompt_length(full, sample, native_reference):
    if full['model'] == 'gemma':
        plen = full['prefill_rows']
        require(type(plen) is int and plen > 0 and full['physical_events'][0] == ['encoder', plen]
                and full['events'][0]['block'] == plen, 'Invalid Gemma prefill')
        return plen
    require(native_reference is not None, 'LLaDA requires --native-reference (native32 requests directory)')
    native = _json(native_reference / f"{sample['index']:02d}-native32.json")
    _identity(native, sample, 'llada21', 'native32')
    verify_bridge(full, native)
    ids = native['prompt_ids']
    require(native['phase'] == 'formal' and native['generation_cap'] == 2048 and isinstance(ids, list)
            and ids and all(type(t) is int and t >= 0 for t in ids), 'Invalid native32 reference')
    first = next((e for e in full['events'] if e['phase'] == 'denoise'), None)
    require(first and len(ids) == (first['block']+1)*32-first['masked_before']
            and full['prefill_rows'] == len(ids)//32*32, 'Native prompt length disagrees with fullwidth')
    return len(ids)


def reproduce(run, historical, out, native_reference=None):
    """Validate the entire cohort before publishing any per-policy counts."""
    require(run.resolve() != historical.resolve(), 'Expected distinct runs')
    model, policies = _cohort(run, historical)
    samples = _json(MANIFEST)['samples']
    require([s['index'] for s in samples] == list(range(30)), 'Incomplete frozen30 manifest')
    totals = {p: Counter() for p in policies}
    for s in samples:
        for p in ('fullwidth', *policies):
            name = f"{s['index']:02d}-{p}.json"
            new, old = (_json(d / name) for d in (run, historical))
            try:
                _identity(old, s, model, p)
                if p == 'fullwidth':
                    require(_exact(new, old) and all('beneficiary' not in e and 'intervention' not in e
                            for e in old['events']), 'Historical fullwidth differs or contains telemetry')
                    plen = _prompt_length(old, s, native_reference)
                    prefill = old['prefill_rows']
                else:
                    require(old['prefill_rows'] == prefill, 'Arm prefill differs from fullwidth')
                    totals[p].update(check_reproduction(new, old, plen))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'{p} request {s["index"]}: {exc}') from exc
    out.mkdir(parents=True, exist_ok=True)
    table(out / f'{model}-historical-reproduction.csv', [dict(policy=p, **totals[p]) for p in policies])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("run","out"):
        p.add_argument("--"+k,required=True,type=Path)
    p.add_argument('--historical', type=Path, help='Audit beneficiary telemetry against this exact one-seat archive')
    p.add_argument('--native-reference', type=Path, help='LLaDA: directory containing the formal NN-native32.json files')
    a = p.parse_args()
    if a.historical:
        reproduce(a.run, a.historical, a.out, a.native_reference)
    elif a.native_reference:
        p.error('--native-reference requires --historical')
    else:
        main(a.run,a.out)

"""Matched MASK trajectories in the historical or one-crowded-seat experiment (CPU only).

An observation is the after-state of a real denoise, including the anchor at 0.
Waits are row sums: 0..3, or 4 for no hit. Stable means through observation 3
only. Padding repeats state, not work. Base output IDs are references, not truth.
"""
import argparse
from collections import Counter
import random
import re
from pathlib import Path
from statistics import mean

from tools import context_gates_audit as gate
from tools.causal_seat_report import MODES, _identity, check_reproduction, inspect, require
from tools.conditional_seat import (DATASETS, DEFAULT_MANIFEST, DEFAULT_RUNS, STRATA,
                                    check_manifest, policy_name, read_json)
from tools.t2t_results import table

H = 4
KINDS = ("common_mask", "mask_victim")
PAIRS = (("rank", "base"), ("random", "base"), ("rank", "random"))
AGREE = tuple(f"agree_t{i}" for i in range(H))
COUNTS = ("carried_row_obs", "anchor_admitted", "branch_missing", "endpoint_missing",
          "endpoint_different", *AGREE, "first_hit", "first_wait_sum", "stable_wait_sum")


def strip_watch(raw):
    return dict(raw, events=[{k: v for k, v in e.items() if k != "beneficiary"}
                             for e in raw["events"]])


def historical_replay(new, old):
    keys = ("model", "policy", "index", "dataset", "task_id", "seed", "status", "output_ids",
            "output_tokens", "generation_forwards", "generation_query_rows",
            "generation_physical_padded_rows", "forwards_by_phase", "q_by_phase")
    require(all(new[k] == old[k] for k in keys) and
            new.get("physical_events") == old.get("physical_events"), "Historical replay mismatch")
    require(len(new["events"]) == len(old["events"]), "Historical event count mismatch")
    for a, b in zip(new["events"], old["events"], strict=True):
        # Older anchors lack the newer victim acceptance diagnostics. Every
        # archived field, including the actual hashes, must still be reproduced.
        require(("intervention" in a) == ("intervention" in b) and
                all(a.get(k) == v for k, v in b.items() if k != "intervention") and
                all(a["intervention"].get(k) == v for k, v in b.get("intervention", {}).items()),
                "Historical event mismatch")


def window(raw, at, rows, offsets, seen):
    states = []
    previous = [None] * len(rows)  # every watched row was MASK before the anchor
    events = raw["events"]
    for i, e in enumerate(events):
        watched = (at >= 0 and i >= at and e["block"] == events[at]["block"] and
                   e["stage"] in ("bootstrap", "dense", "selected", "verify"))
        require(watched == ("beneficiary" in e), f"Missing or unexpected beneficiary at event {i}")
        if not watched:
            continue
        b = e["beneficiary"]
        require(isinstance(b, dict) and b.get("rows") == rows and
                all(type(r) is int for r in b["rows"]), "Wrong watched MASK rows/order")
        ids, off = b.get("ids"), b.get("output_offsets")
        require(isinstance(ids, list) and len(ids) == len(rows) and all(
            t is None or (type(t) is int and t >= 0) for t in ids), "Invalid beneficiary IDs")
        require(isinstance(off, list) and len(off) == len(rows) and all(type(o) is int for o in off)
                and len({o - r for o, r in zip(off, rows)}) <= 1, "Misaligned output offsets")
        if offsets is None:
            offsets = off
        require(off == offsets, "Output offsets changed across observations or arms")
        selected = (set(e["positions"]) if "positions" in e else
                    {pos - 32 * e["block"] for pos in e["absolute_positions"]})
        require(all(row in selected or before == after for row, before, after in
                    zip(rows, previous, ids, strict=True)), "Off-Q beneficiary changed state")
        previous = ids
        # Equal post-state (even after divergence/reconvergence) implies equal
        # accepted IDs on the same watched rows. Check beyond H as well.
        for key in ("accepted_sha256", "trajectory_sha256"):
            digest = e.get(key)
            require(isinstance(digest, str) and len(digest) == 64, "Missing denoise state evidence")
            require(seen.setdefault((key, digest), ids) == ids, "Same state, different watched IDs")
        if len(states) < H:
            states.append(ids)
    return offsets, states


def measure(raw, *, inspection=None):
    """Return arm and paired counts for one complete base/rank/random triplet."""
    at, plans = inspect([strip_watch(r) for r in raw]) if inspection is None else inspection
    p = plans[0] if plans else {}
    stratum = p.get("stratum", "none")
    require(not plans or stratum in ("slack", "mask"), "Unknown anchor stratum")
    rows = sorted(set(p["base"]) & set(p["masks"])) if plans else []
    ref = raw[0]["output_ids"]
    records, pairs, seen, offsets = [], [], {}, None
    for arm, r in zip(MODES, raw, strict=True):
        offsets, states = window(r, at, rows, offsets, seen)
        if plans:
            plan = r["events"][at]["intervention"]
            require(sum(t is not None for t in states[0]) == plan["after_admissions"],
                    "Observed admissions disagree with anchor execution")
        carried = H - len(states) if states else 0
        states += states[-1:] * carried
        counts = {kind: Counter() for kind in KINDS}
        for j, (row, off) in enumerate(zip(rows, offsets or (), strict=True)):
            c = counts["mask_victim" if row == p["victim"] else "common_mask"]
            c["watched"] += 1
            c["carried_row_obs"] += carried
            c["anchor_admitted"] += states[0][j] is not None
            returned = 0 <= off < len(r["output_ids"])
            c["branch_missing"] += not returned  # All watched offsets, including outside base.
            if not 0 <= off < len(ref):
                continue
            c["eligible"] += 1
            c["endpoint_missing"] += not returned
            c["endpoint_different"] += returned and r["output_ids"][off] != ref[off]
            hits = [s[j] == ref[off] for s in states]
            for key, hit in zip(AGREE, hits, strict=True):
                c[key] += hit
            c["first_hit"] += any(hits)
            c["first_wait_sum"] += hits.index(True) if any(hits) else H
            c["stable_wait_sum"] += max((t + 1 for t, hit in enumerate(hits) if not hit), default=0)
        for kind, c in counts.items():
            records.append(dict(model=r["model"], index=r["index"], dataset=r["dataset"],
                                stratum=stratum, row_kind=kind, arm=arm, watched=c["watched"],
                                eligible=c["eligible"], **{k: c[k] for k in COUNTS}))
    by_arm = {(r["arm"], r["row_kind"]): r for r in records}
    for left, right in PAIRS:
        for kind in KINDS:
            a, b = by_arm[left, kind], by_arm[right, kind]
            pairs.append({**{k: a[k] for k in ("model", "index", "dataset", "stratum", "row_kind")},
                          "pair": f"{left}_vs_{right}", "watched": a["watched"], "eligible": a["eligible"],
                          **{f"delta_{k}": a[k] - b[k] for k in COUNTS}})
    return records, pairs


def summarize(records, pairs):
    """Add counts per arm or signed counts per pair; never pool requests across arms."""
    result = []
    for data, key, prefix in ((records, "arm", ""), (pairs, "pair", "delta_")):
        for name in dict.fromkeys(r[key] for r in data):
            for kind in KINDS:
                group = [r for r in data if r[key] == name and r["row_kind"] == kind]
                for dataset in DATASETS:
                    for stratum in STRATA:
                        sub = [r for r in group if dataset in ("all", r["dataset"]) and
                               stratum in ("all", r["stratum"])]
                        waits = [r[prefix + "stable_wait_sum"] / r["eligible"] for r in sub if r["eligible"]]
                        result.append(dict(model=group[0]["model"], dataset=dataset, stratum=stratum,
                            row_kind=kind, measure=key, comparison=name, requests=len(sub),
                            anchors=sum(r["stratum"] != "none" for r in sub),
                            eligible_requests=sum(r["eligible"] > 0 for r in sub),
                            watched=sum(r["watched"] for r in sub), eligible=sum(r["eligible"] for r in sub),
                            request_mean_stable_wait=mean(waits) if waits else "",
                            requests_faster=sum(w < 0 for w in waits) if key == "pair" else "",
                            requests_slower=sum(w > 0 for w in waits) if key == "pair" else "",
                            **{k: sum(r[prefix + k] for r in sub) for k in COUNTS}))
    return result


def main(run, out):
    meta = read_json(run / "run.json")
    model = meta["model"]
    require(model in DEFAULT_RUNS, "Expected llada21 Q4 or gemma Q32")
    history, _, q = DEFAULT_RUNS[model]
    policies = [policy_name(q, mode) for mode in MODES]
    for policy in policies:
        require(meta["policies"].count(policy) == 1 and read_json(run / f"complete-{policy}.json") ==
                dict(policy=policy, requests=30), f"Missing or incomplete {policy}")
    identity, records, pairs = [], [], []
    for i in range(30):
        raw = []
        for policy in policies:
            filename = f"{i:02d}-{policy}.json"
            r = read_json(run / filename)
            require((r["model"], r["policy"], r["index"]) == (model, policy, i), "Wrong request file")
            historical_replay(strip_watch(r), read_json(history / filename))
            raw.append(r)
        identity.append({k: raw[0][k] for k in ("index", "dataset", "task_id", "seed")})
        points, contrasts = measure(raw)
        records.extend(points)
        pairs.extend(contrasts)
    check_manifest(DEFAULT_MANIFEST, [(model, identity)])
    publish(out, f"{model}-beneficiary", records, pairs)


def publish(out, prefix_name, records, pairs):
    summary = summarize(records, pairs)
    for data, prefix in ((records, ""), (pairs, "delta_"), (summary, "")):
        for r in data:
            for metric in ("first_wait", "stable_wait"):
                k = prefix + metric
                r[k + "_mean"] = r[k + "_sum"] / r["eligible"] if r["eligible"] else ""
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("requests", records), ("paired", pairs), ("summary", summary)):
        table(out / f"{prefix_name}-{name}.csv", data)


ONCE = tuple('t2t_q8_f0_m7_hard_context_once_' + m for m in MODES)
NOEDIT = 't2t_q8_f0_m8_hard_noedit'
PREFIX = 'llada21-q8-crowded-beneficiary'
OBSERVER = {'crowded_screen', 'intervention', 'beneficiary', 'trajectory_sha256', 'accepted_sha256'}
MILESTONES = ('before_first_verify', 'after_first_verify', 'commit')


def crowded_cohort(runs, history):
    require(len({p.resolve() for p in runs}) == len(runs) == 2 and
            history.resolve() not in {p.resolve() for p in runs}, 'Expected two distinct qualified shards')
    owners = {}
    for d in [history, *runs]:
        historical = d == history
        meta = gate.load(d / 'run.json')
        require(gate.same([meta[k] for k in ('model', 'batch_size', 'cap', 'block')], ['llada21', 1, 2048, 32])
                and gate.same(meta.get('seed_offset', 0 if historical else None), 0), 'Wrong run configuration')
        require(gate.same(gate.load(d / 'qualified.json'), dict(model='llada21', full_width=32, matched=30)),
                'Incomplete LLaDA fullwidth qualification')
        names = meta['policies']
        require(len(names) == len(set(names)) == len(meta['arms']) and names, 'Invalid run arms')
        specs = dict(zip(names, meta['arms'], strict=True))
        if not historical:
            require(meta['context_weights_sha256'] == gate.WEIGHTS, 'Wrong frozen context weights')
        for p in ([NOEDIT] if historical else names):
            quota = 8 if historical else 7
            require((historical or (p in ONCE and p not in owners)) and gate.same(specs[p], dict(
                budget=8, fresh=0, mask=quota, selection=p.split('_', 4)[4])), 'Unexpected or duplicated policy')
            require(gate.same(gate.load(d / f'complete-{p}.json'), dict(policy=p, requests=30)), 'Incomplete policy')
            if not historical:
                owners[p] = d
        for p in ('fullwidth', *([NOEDIT] if historical else names)):
            require({f.name for f in d.glob(f'[0-9]*-{p}.json')} == {f'{i:02d}-{p}.json' for i in range(30)},
                    f'Wrong frozen30 files: {p}')
    require(set(owners) == set(ONCE), 'Incomplete crowded triplet')
    return owners


def compare_base(base, historical):
    # Do not ignore an observer key if a future historical archive already has it.
    require(gate.same({k:v for k,v in base.items() if k not in ('events','policy')},
                      {k:v for k,v in historical.items() if k not in ('events','policy')}),
            'Historical Q8 request differs')
    for e, old in zip(base['events'], historical['events'], strict=True):
        require(gate.same({k:v for k,v in e.items() if k in old or k not in OBSERVER}, old),
                'Historical Q8 event differs')


def verify_screen(e, valid, arm, seed):
    """Replay the hypothetical ranked gate, separately check the actual one-shot action."""
    m = e['crowded_screen']
    base, shift = set(m['base']), 32 * e['block']
    hypothetical = base.copy()
    if m['entrant'] is not None:
        hypothetical.discard(m['victim'])
        hypothetical.add(m['entrant'])
    absolute = lambda local: [shift+i for i in sorted(local)]
    hypothetical_event = dict(e, context_move=m, absolute_positions=absolute(hypothetical),
                              reserved_extra_edit_positions=absolute(hypothetical-base))
    masks, fresh = gate.placement(hypothetical_event, valid, gate.POLICIES[2], seed)
    anchor = m['reason'] == 'swap_mask'
    require(anchor == ('intervention' in e), 'First crowded opportunity not executed/recorded')
    picked = base
    if anchor:
        p = e['intervention']
        victim, candidate = m['victim'], m['rank']
        rng = random.Random(f"context-swap:{seed}:{e['block']}:{e['hard_clock']}")
        other = rng.choice(sorted(set(m['old_pool'])-base))
        swap = lambda i: sorted((base-{victim}) | {i})
        expected = dict(base=m['base'], targeted=swap(candidate), random=swap(other), victim=victim,
                        candidate=candidate, random_candidate=other, masks=m['masks'], fresh=m['fresh'],
                        stratum='mask', clock=e['hard_clock'], risk_score=m['score'])
        require(all(gate.same(v, p[k]) for k,v in expected.items()), 'Crowded plan, risk or salted placement differs')
        picked = set(p[dict(base='base', rank='targeted', random='random')[arm]])
        require(min(base & masks) in picked and p['executed'] == sorted(picked), 'First MASK or execution differs')
        for k in ('pre_sha256', 'after_sha256', 'post_rng_sha256'):
            require(re.fullmatch('[a-f0-9]{64}', p[k]), f'Invalid anchor {k}')
    require(gate.same(e['absolute_positions'], absolute(picked)) and
            gate.same(e['reserved_extra_edit_positions'], absolute(picked-base)) and
            e['masked_before'] == len(masks) and e['selected_masked_before'] == len(picked & masks),
            'Actual execution differs from screened decision')
    return masks, fresh


def crowded_path(raw, arm, plen, at):
    """Account for every forward, including events after divergence and other blocks."""
    fs, qs, hist = Counter(), Counter(), Counter()
    ignored_slack = screens = edits = commits = next_fresh = 0
    prev = previous_stage = None
    for j, e in enumerate(raw['events']):
        stage, b = e['stage'], e['block']
        selected = stage == 'selected'
        denoise = stage in ('bootstrap', 'verify', 'selected')
        shift = b*32
        valid = set(range(max(0, plen-shift), 32))
        absolute = gate.positions(e['absolute_positions'])
        picked = {i-shift for i in absolute}
        q = len(picked)
        require(stage in gate.STAGES and b == plen//32+commits and e['hard_clock'] == j+1 and
                e['phase'] == ('denoise' if denoise else stage) and e['F'] == 1 and q > 0 and
                gate.same(e['layer_physical_q'], [q]*20) and e['physical_q'] == e['actual_q'] == e['q_tokens'] == q and
                (e['q_start'], e['q_end']) == (min(absolute), max(absolute)+1) and e['q_budget'] == 8 and
                e['overflow'] == max(0, q-8) and e['context_version'] == commits and
                e['store_kv'] is (not denoise) and e['approximate_same_block_kv'] is selected and
                (e['prefix_kv_length'], e['visible_kv']) == ((0, shift) if stage=='prefill' else (shift, shift+32)),
                'Invalid physical row, layer or context ledger')
        require(e['mandatory_fresh'] == next_fresh and (not selected or
                (picked <= valid and q == min(len(valid), max(8,next_fresh)))), 'Invalid mandatory Q width')
        if not selected:
            require(absolute == (set(range(shift)) if stage=='prefill' else set(range(shift, shift+32))), 'Wrong full Q')
        screen = selected and (at == -1 or j <= at)
        require(('crowded_screen' in e) == screen, 'Missing or unexpected crowded screen')
        if j != at or arm == 'base':
            require(e['reserved_extra_edit_positions'] == e['reserved_strict_edit_positions'] == [],
                    'Unexpected explicit swap outside intervention')
        if screen:
            masks, fresh = verify_screen(e, valid, arm, raw['seed'])
            require(prev is not None, 'Missing preceding denoise')
            gate.transition(prev, masks, fresh)
            screens += 1
            ignored_slack += e['crowded_screen']['reason'] == 'swap_old'
        elif stage == 'bootstrap':
            masks = valid
            require(previous_stage in (None, 'prefill', 'commit') and e['masked_before'] == len(valid),
                    'Bootstrap prompt geometry or boundary differs')
        else:
            masks = set()
        if denoise:
            if stage != 'bootstrap':
                previous = raw['events'][j-1]
                require(previous_stage in ('bootstrap','selected','verify') and previous['block'] == b and
                        e['masked_before'] == previous['masked_after'] and (e['masked_before'] > 0) == selected,
                        'Invalid MASK count or denoise boundary')
            hits = {i-shift for i in gate.positions(e['reserved_strict_edit_positions'])}
            reserved = {i-shift for i in gate.positions(e['reserved_extra_edit_positions'])}
            strict = e['hard_strict_edits']
            revokes = e['edited'] - strict
            admits = e['masked_before'] - e['masked_after'] + revokes
            require(0 <= admits <= e['accepted'] <= e['selected_masked_before'] <= q and revokes >= 0 and
                    len(hits) <= strict and hits <= reserved <= picked, 'Invalid physical transitions')
            if selected and not screen:
                require(e['selected_masked_before'] == min(e['masked_before'], q-next_fresh),
                        'Post-intervention selection not zero reserve')
            if j == at:
                plan = e['intervention']
                watched = {plan[k] for k in ('victim', 'candidate', 'random_candidate')}
                strict_rows, revoked = (gate.positions(plan[k]) for k in ('strict_rows', 'revoked_rows'))
                require(strict_rows | revoked <= (picked & watched)-masks and not strict_rows & revoked and
                        strict_rows & reserved == hits and len(strict_rows) <= strict and len(revoked) <= revokes,
                        'Invalid direct edit evidence')
            for k in ('accepted_sha256', 'trajectory_sha256'):
                require(re.fullmatch('[0-9a-f]{64}', e[k]), f'Missing {k}')
            prev = masks, picked, reserved, hits, admits, revokes, strict
            next_fresh = admits + strict
            edits += e['edited']
        if stage == 'commit':
            require(previous_stage in ('bootstrap','selected','verify') and raw['events'][j-1]['masked_after']==0,
                    'Commit without a resolved canvas')
            next_fresh, prev = 0, None
            commits += 1
        previous_stage = stage
        if stage != 'prefill':
            fs[stage] += 1
            qs[stage] += q
            hist[str(q)] += 1
    require(raw['events'][-1]['stage'] == 'commit' and raw['prefill_rows'] == plen//32*32 and
            gate.same([dict(fs), dict(qs), dict(hist), sum(fs.values()), sum(qs.values()), edits, max(map(int,hist))],
                      [raw[k] for k in ('forwards_by_phase','q_by_phase','q_histogram','generation_forwards',
                                         'generation_query_rows','t2t_edits','max_q')]), 'Request ledger differs')
    return dict(screened_steps=screens, skipped_slack_offers=ignored_slack, anchored=int(at >= 0))


def milestones(raw, arm, ref, at):
    """Secondary, anchor-block only. Never pad an unobserved stage or carry across blocks."""
    events = raw['events']
    rows, offsets, snapshots = [], [], {}
    victim = None
    commit_seen = False
    if at >= 0:
        anchor = events[at]
        rows, offsets = (anchor['beneficiary'][k] for k in ('rows', 'output_offsets'))
        victim = anchor['intervention']['victim']
        prev = None
        for e in events[at:]:
            if e['block'] != anchor['block']:
                break
            if e['stage'] == 'commit':
                commit_seen = True
                # LLaDA commit consumes the last actual canvas without editing it.
                # check_reproduction checks the watched IDs against delivery up to branch EOS.
                if prev is not None:
                    snapshots['commit'] = prev
                break
            ids = e['beneficiary']['ids']
            if e['stage'] == 'verify' and 'after_first_verify' not in snapshots:
                if prev is not None:
                    snapshots['before_first_verify'] = prev
                snapshots['after_first_verify'] = ids
            prev = ids
    for stage in MILESTONES:
        reached = commit_seen if stage == 'commit' else 'after_first_verify' in snapshots
        observed = stage in snapshots
        for kind in KINDS:
            group = [j for j,r in enumerate(rows) if (r == victim) == (kind == 'mask_victim')]
            eligible = [j for j in group if 0 <= offsets[j] < len(ref)]
            n = len(eligible) if observed else 0
            agree = sum(snapshots[stage][j] == ref[offsets[j]] for j in eligible) if observed else 0
            yield dict(index=raw['index'], dataset=raw['dataset'], arm=arm, row_kind=kind, stage=stage,
                       requests=1, no_anchor_requests=int(at < 0), missing_stage_requests=int(at >= 0 and not reached),
                       stage_without_snapshot_requests=int(reached and not observed), snapshot_requests=int(observed),
                       commit_input_from_last_denoise=int(stage=='commit' and observed), watched_rows=len(group),
                       no_base_final_id_rows=len(group)-len(eligible), eligible_rows=len(eligible),
                       eligible_missing_stage_rows=0 if reached else len(eligible),
                       eligible_stage_without_snapshot_rows=len(eligible) if reached and not observed else 0,
                       eligible_missing_snapshot_rows=len(eligible)-n, compared_rows=n, agreeing_rows=agree,
                       agreement=agree/n if n else '')


def secondary_summary(rows):
    by = {}
    for r in rows:
        for dataset in ('all', r['dataset']):
            key = (dataset, r['arm'], r['row_kind'], r['stage'])
            by.setdefault(key, Counter()).update({k:v for k,v in r.items() if k not in
                                                  ('index','dataset','arm','row_kind','stage','agreement')})
    return [dict(dataset=d, arm=a, row_kind=k, stage=s, **c,
                 agreement=c['agreeing_rows']/c['compared_rows'] if c['compared_rows'] else '')
            for (d,a,k,s), c in by.items()]


def crowded(runs, out, history):
    owners = crowded_cohort(runs, history)
    samples = gate.load(DEFAULT_MANIFEST)['samples']
    require([s['index'] for s in samples] == list(range(30)) and Counter(s['dataset'] for s in samples) ==
            dict.fromkeys(DATASETS[1:], 10), 'Incomplete frozen30 manifest')
    records, pairs, secondary, audit = [], [], [], []
    for s in samples:
        def load(d, p):
            r = gate.load(d / f"{s['index']:02d}-{p}.json")
            _identity(r, s, 'llada21', p)
            return r
        try:
            full = load(history, 'fullwidth')
            for d in runs:
                require(gate.same(load(d, 'fullwidth'), full), 'Historical fullwidth qualifier differs')
            first = next(e for e in full['events'] if e['stage'] == 'bootstrap')
            plen = (first['block']+1)*32-first['masked_before']
            require(full['prefill_rows'] == plen//32*32, 'Invalid qualifier prompt geometry')
            raw = [load(owners[p], p) for p in ONCE]
            compare_base(raw[0], load(history, NOEDIT))
            stripped = [strip_watch(r) for r in raw]
            at, plans = inspect(stripped)
            if at >= 0:
                shared = ('crowded_screen', *gate.FORK_FIELDS)
                require(all(gate.same(r['events'][at][k], raw[0]['events'][at][k]) for r in raw[1:] for k in shared),
                        'Anchor input, screen or layer work differs')
            same_paths = [(0,1),(0,2)] if at < 0 else [(1,2)] if plans[0]['candidate']==plans[0]['random_candidate'] else []
            for a,b in same_paths:
                require(gate.same(dict(raw[a], policy=None), dict(raw[b], policy=None)), 'Identical placements differ')
            for arm, r, clean in zip(MODES, raw, stripped, strict=True):
                stats = crowded_path(r, arm, plen, at)
                # Use existing beneficiary validation for *all* real watched denoises and output offsets.
                check_reproduction(r, clean, plen)
                e = r['events'][at] if at >= 0 else None
                audit.append(dict(index=s['index'], dataset=s['dataset'], arm=arm, **stats,
                                  anchor_clock=e['hard_clock'] if e else '', anchor_admissions=e['intervention']['after_admissions'] if e else 0,
                                  direct_candidate_strict_hits=len(e['reserved_strict_edit_positions']) if e else 0))
                secondary.extend(milestones(r, arm, raw[0]['output_ids'], at))
            points, contrasts = measure(raw, inspection=(at, plans))
            records.extend(points)
            pairs.extend(contrasts)
        except (KeyError, TypeError, StopIteration, ValueError) as exc:
            raise ValueError(f"crowded request {s['index']}: {exc}") from exc
    publish(out, PREFIX, records, pairs)
    for name, data in [('audit', audit), ('secondary-requests', secondary), ('secondary-summary', secondary_summary(secondary))]:
        table(out / f'{PREFIX}-{name}.csv', data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crowded', action='store_true', help='audit both qualified Q8 one-crowded-exchange shards')
    parser.add_argument('--run', action='append', required=True, type=Path, help='raw run directory; repeat for crowded shards')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--historical', type=Path, help='crowded only: original Q8 hard_noedit archive (default hard-v1-llada21)')
    args = parser.parse_args()
    try:
        if args.crowded:
            crowded(args.run, args.out, args.historical or DEFAULT_RUNS['llada21'][1])
        elif len(args.run) != 1 or args.historical:
            parser.error('repeated --run and --historical require --crowded')
        else:
            main(args.run[0], args.out)
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

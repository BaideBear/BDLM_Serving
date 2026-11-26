import csv
import json
from copy import deepcopy
from hashlib import sha256
import random
import subprocess
import sys

import pytest

from tools import beneficiary_rows as report


def fixture(model="llada21", stratum="mask"):
    q = {"llada21": 4, "gemma": 32}[model]
    victim = 1 if stratum == "mask" else 3
    base = list(range(q))
    swap = lambda row: sorted((set(base) - {victim}) | {row})
    paths = [([None, 30, 40], [20, 30, 40]),
             ([20, None, 40], [None, None, 40], [20, None, 40], [20, 30, 40], [None, 30, 40]),
             ([None, None, 40],)]
    outputs = [[9, 20, 30], [9, 99], [9, 20, 33, 40]]
    raw = []
    for mode, selected, path, output in zip(report.MODES, (base, swap(q), swap(q+1)), paths, outputs):
        events = [dict(block=3, stage="bootstrap")]
        for t, ids in enumerate(path):
            fingerprint = sha256(str(ids).encode()).hexdigest()
            positions = selected if t == 0 else base
            event = dict(block=3, stage="selected" if t == 0 else "verify", q=len(positions),
                         accepted_sha256=fingerprint, trajectory_sha256=fingerprint,
                         beneficiary=dict(rows=[0, 1, 2], ids=ids, output_offsets=[1, 2, 3]))
            if model == "llada21":
                event.update(absolute_positions=[96+i for i in positions], phase="denoise")
            else:
                event["positions"] = positions
            if t == 0:
                event["intervention"] = dict(base=base, targeted=swap(q), random=swap(q+1),
                    victim=victim, candidate=q, random_candidate=q+1, fresh=[], masks=[0, 1, 2],
                    stratum=stratum, clock=7, executed=positions, pre_sha256="1"*64,
                    post_rng_sha256="2"*64, after_sha256=fingerprint,
                    after_admissions=sum(x is not None for x in ids))
            events.append(event)
        events.extend([dict(block=3, stage="commit"), dict(block=4, stage="bootstrap")])
        if stratum == "none":
            events, output = [dict(block=3, stage="bootstrap")], [9]
        raw.append(dict(index=0, model=model, dataset="gsm8k", task_id="task", seed=1,
                        policy=report.policy_name(q, mode), status="success", events=events,
                        output_ids=output, output_tokens=len(output), generation_forwards=len(events),
                        generation_query_rows=50, generation_physical_padded_rows=50,
                        forwards_by_phase={"denoise": 4}, q_by_phase={"denoise": 50}))
    return raw


@pytest.mark.parametrize("model", ["llada21", "gemma"])
def test_fixed_offsets_and_horizon(model):
    raw = fixture(model)
    original = deepcopy(raw)
    points, pairs = report.measure(raw)
    assert raw == original
    by = {(r["arm"], r["row_kind"]): r for r in points}
    base, rank, rand = (by[m, "common_mask"] for m in report.MODES)
    # The other common MASK is beyond base EOS, but still counts for admissions.
    assert (base["watched"], base["eligible"], base["anchor_admitted"]) == (2, 1, 1)
    assert (base["first_wait_sum"], base["stable_wait_sum"], base["carried_row_obs"]) == (1, 1, 4)
    assert [rank[k] for k in report.AGREE] == [1, 0, 1, 1]
    assert (rank["first_wait_sum"], rank["stable_wait_sum"]) == (0, 2)  # revoked again at t=4
    assert (rank["anchor_admitted"], rank["endpoint_different"], rank["endpoint_missing"]) == (2, 1, 0)
    assert (rand["first_hit"], rand["first_wait_sum"], rand["stable_wait_sum"]) == (0, 4, 4)
    assert (rand["branch_missing"], rand["carried_row_obs"]) == (0, 6)
    assert (base["branch_missing"], base["endpoint_missing"]) == (1, 0)
    victim = by["rank", "mask_victim"]
    assert (victim["eligible"], victim["anchor_admitted"], victim["endpoint_missing"]) == (1, 0, 1)
    assert (victim["first_wait_sum"], victim["stable_wait_sum"]) == (3, 3)
    assert by["random", "mask_victim"]["endpoint_different"] == 1
    for r in pairs:
        a, b = (by[m, r["row_kind"]] for m in r["pair"].split("_vs_"))
        assert r["eligible"] == a["eligible"] == b["eligible"]
        for k in report.COUNTS:
            assert r["delta_"+k] == a[k] - b[k]


def test_anchor_with_no_watched_masks_and_collision_replay():
    raw = fixture(stratum="slack")
    for r in raw:
        for e in r["events"]:
            if "intervention" in e:
                e["intervention"].update(masks=[], after_admissions=0)
            if "beneficiary" in e:
                e["beneficiary"] = dict(rows=[], ids=[], output_offsets=[])
    points, pairs = report.measure(raw)
    assert all(r["stratum"] == "slack" and r["watched"] == r["eligible"] == 0 for r in points + pairs)
    raw = fixture()
    for r in raw:
        p = r["events"][1]["intervention"]
        p.update(random=p["targeted"], random_candidate=p["candidate"])
    raw[2] = deepcopy(raw[1])
    assert len(report.measure(raw)[1]) == 6
    raw[2]["events"][2]["beneficiary"]["ids"][0] = 4321
    with pytest.raises(ValueError, match="Same state"):
        report.measure(raw)


def test_own_final_wait_uses_same_offsets_but_branch_delivery():
    raw = fixture()
    raw[1]['output_ids'] = [9, 99, 30]
    points, pairs = report.measure(raw, own_final=True)
    rec = {(r['arm'], r['row_kind']): r for r in points}
    assert rec['rank', 'common_mask']['stable_wait_sum'] == 4
    assert rec['random', 'mask_victim']['stable_wait_sum'] == 4
    assert all(r['eligible'] == 1 for r in points + pairs)
    # Remove one branch's victim offset: drop from every paired denominator.
    raw[1]['output_ids'].pop()
    assert all(not r['eligible'] for r in report.measure(raw, own_final=True)[0]
               if r['row_kind'] == 'mask_victim')


@pytest.mark.parametrize("error,match", [
    ("offset", "offsets"), ("rows", "rows/order"), ("late", "Missing or unexpected"),
    ("null", "IDs"), ("off_q", "Off-Q"), ("state", "Same state"),
    ("pre", "pre-state"), ("prefix", "Prefixes differ"), ("admit", "admissions"),
])
def test_reject_invalid_evidence(error, match):
    raw = fixture()
    ev = raw[1]["events"]
    watch = ev[1]["beneficiary"]
    if error == "offset":
        watch["output_offsets"] = [2, 3, 4]
    elif error == "rows":
        watch["rows"].reverse()
    elif error == "late":
        del ev[5]["beneficiary"]  # still required after the reporting horizon
    elif error == "null":
        watch["ids"][1] = False
    elif error == "off_q":
        watch["ids"][1] = 30
    elif error == "state":
        ev[2]["accepted_sha256"] = raw[0]["events"][1]["accepted_sha256"]
    elif error == "pre":
        ev[1]["intervention"]["pre_sha256"] = "e"*64
    elif error == "admit":
        ev[1]["intervention"]["after_admissions"] += 1
    else:
        ev[0]["stage"] = "dense"
    with pytest.raises(ValueError, match=match):
        report.measure(raw)


def emit(path, data):
    path.write_text(json.dumps(data))


@pytest.mark.parametrize("model,q", [("llada21", 4), ("gemma", 32)])
def test_report_and_historical_replay(tmp_path, monkeypatch, model, q):
    new, old, out = (tmp_path / d for d in ("private-new", "private-archive", "out"))
    new.mkdir()
    old.mkdir()
    monkeypatch.setitem(report.DEFAULT_RUNS, model, (old, None, q))
    policies = [report.policy_name(q, arm) for arm in report.MODES]
    emit(new / "run.json", dict(model=model, policies=policies))
    for p in policies:
        emit(new / f"complete-{p}.json", dict(policy=p, requests=30))
    for task in report.read_json(report.DEFAULT_MANIFEST)["samples"]:
        i = task["index"]
        for r in fixture(model, {0: "mask", 10: "slack"}.get(i, "none")):
            r.update({k: task[k] for k in ("index", "task_id", "dataset", "seed")})
            name = f"{i:02d}-{r['policy']}.json"
            archived = report.strip_watch(r)
            emit(old / name, archived)
            if i in (0, 10):
                r["events"][1]["intervention"]["victim_queried"] = r["policy"].endswith("base")
            emit(new / name, r)
    report.main(new, out)
    outputs = {f.name: f.read_bytes() for f in out.iterdir()}
    report.main(new, out)
    assert outputs == {f.name: f.read_bytes() for f in out.iterdir()}
    for name, count in (("requests", 180), ("paired", 180), ("summary", 192)):
        with (out / f"{model}-beneficiary-{name}.csv").open() as f:
            data = list(csv.DictReader(f))
        assert len(data) == count
        assert not ({"output_ids", "ids", "rows", "output_offsets", "prompt_ids", "task_id"} & data[0].keys())
        if name == "summary":
            rows = {(r["dataset"], r["stratum"]): r for r in data
                    if r["comparison"] == "rank_vs_base" and r["row_kind"] == "mask_victim"}
            assert tuple(rows["all", "all"][k] for k in ("requests", "anchors", "eligible_requests")) == ("30", "2", "1")
            assert (rows["all", "none"]["requests"], rows["all", "slack"]["watched"]) == ("28", "0")
            assert all(rows[d, "all"]["requests"] == "10" for d in report.DATASETS[1:])
            assert rows["gsm8k", "mask"]["anchor_admitted"] == "-1"
            assert float(rows["all", "mask"]["first_wait_mean"]) == 3
            assert rows["all", "slack"]["first_wait_mean"] == ""
    path = old / f"29-{policies[-1]}.json"
    changed = report.read_json(path)
    changed["output_ids"] = [31415926]
    emit(path, changed)
    with pytest.raises(ValueError, match="Historical replay mismatch"):
        report.main(new, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def crowded_case(sample, *, allow=True):
    """Prompt tail, screened prefix, divergent selected sets, two dense verification steps."""
    triplet = []
    for arm, policy in zip(report.MODES, report.ONCE):
        events = []
        tokens = [1]*16 + [None]*16
        fresh, watched = set(), []
        strict_sum = post = 0
        def event(stage, picked):
            q = len(picked)
            e = dict(phase='commit' if stage=='commit' else 'denoise', stage=stage, block=0,
                     F=1, hard_clock=len(events)+1, q_budget=8, q_tokens=q, actual_q=q, physical_q=q,
                     layer_physical_q=[q]*20, absolute_positions=sorted(picked), q_start=min(picked), q_end=max(picked)+1,
                     prefix_kv_length=0, visible_kv=32, store_kv=stage=='commit', context_version=0,
                     approximate_same_block_kv=stage=='selected', mandatory_fresh=len(fresh), overflow=max(0,q-8),
                     hard_strict_edits=0, reserved_extra_edit_positions=[], reserved_strict_edit_positions=[])
            events.append(e)
            return e
        while post < 2:
            clock = len(events)+1
            mask = {i for i in range(16,32) if tokens[i] is None}
            old = set(range(16,32))-mask-fresh
            stage = 'bootstrap' if clock==1 else 'selected' if mask else 'verify'
            base = fresh | set((sorted(mask)+sorted(old))[:max(0,8-len(fresh))])
            screen = plan = None
            picked = set(range(32)) if stage!='selected' else base
            if stage=='selected' and not watched:
                top = min(old) if old else None
                slack = len(fresh)+len(mask)<8
                score = report.gate.CUTOFF if allow else -10.
                if not old: reason = 'no_old'
                elif len(fresh)>8: reason = 'overflow'
                elif len(fresh)==8: reason = 'no_room'
                elif not slack and len(base & mask)==1: reason = 'protected_mask'
                elif not slack and not allow: reason = 'below_threshold'
                elif top in base: reason = 'already_selected'
                else: reason = 'swap_old' if slack else 'swap_mask'
                screened = reason in ('below_threshold','already_selected','swap_old','swap_mask')
                victim = max(base & (old if slack else mask)) if reason.startswith('swap_') else None
                screen = dict(base=sorted(base), old_pool=sorted(old), fresh=sorted(fresh), masks=sorted(mask),
                              reason=reason, rank=top if screened else None, score=score if screened else None,
                              entrant=top if victim is not None else None, victim=victim)
                if reason=='swap_mask':
                    other = random.Random(f"context-swap:{sample['seed']}:0:{clock}").choice(sorted(old-base))
                    swap = lambda i: sorted(base-{victim}|{i})
                    plan = dict(base=sorted(base), targeted=swap(top), random=swap(other), candidate=top,
                                random_candidate=other, fresh=screen['fresh'], masks=screen['masks'],
                                victim=victim, stratum='mask', clock=clock, pre_sha256='a'*64, post_rng_sha256='b'*64,
                                risk_score=score, strict_rows=[], revoked_rows=[])
                    picked = set(plan[{'base':'base', 'rank':'targeted','random':'random'}[arm]])
                    watched = sorted(base & mask)
            e = event(stage,picked)
            if screen is not None:
                e['crowded_screen'] = screen
            new = sorted(picked & mask)[:4 if clock==1 else 1 if clock==2 else 3]
            for i in new:
                tokens[i] = i+100
            strict = set()
            if stage=='verify':
                if post == 0:
                    i = 21
                    tokens[i] += 100
                    strict.add(i)
                post += 1
            fresh = set(new) | strict
            strict_sum += len(strict)
            digest = sha256(str(tokens).encode()).hexdigest()
            e.update(masked_before=len(mask), selected_masked_before=len(picked & mask), accepted=len(new),
                     edited=len(strict), hard_strict_edits=len(strict), masked_after=len(mask)-len(new),
                     step_in_block=clock, post_steps=post, accepted_sha256=digest, trajectory_sha256=digest)
            if watched:
                e['beneficiary'] = dict(rows=watched, ids=[tokens[i] for i in watched], output_offsets=[i-16 for i in watched])
            if plan:
                e['reserved_extra_edit_positions'] = sorted(picked-base)
                plan.update(after_sha256=digest, executed=sorted(picked), after_admissions=len(new),
                            victim_unresolved_before=True, victim_queried=plan['victim'] in picked,
                            victim_accepted=plan['victim'] in new)
                e['intervention'] = plan
        event('commit', set(range(32)))
        fs, qs, hist = (report.Counter() for _ in range(3))
        for e in events:
            fs[e['stage']]+=1; qs[e['stage']]+=e['q_tokens']; hist[str(e['q_tokens'])]+=1
        n, f, q = 16, len(events), sum(qs.values())
        triplet.append(dict(**{k:sample[k] for k in ('index','dataset','task_id','seed')}, model='llada21', status='success',
                            policy=policy, events=events, output_ids=tokens[16:], output_tokens=n, prefill_rows=0,
                            generation_forwards=f, generation_query_rows=q, generation_physical_padded_rows=q,
                            tpf=n/f, token_efficiency=n/q, max_q=32, t2t_edits=strict_sum,
                            forwards_by_phase=dict(fs), q_by_phase=dict(qs), q_histogram=dict(hist)))
    return triplet


@pytest.mark.parametrize('arm',report.MODES)
def test_crowded_window_and_secondary(arm):
    raw = crowded_case(dict(index=1, dataset='gsm8k', task_id='toy', seed=3))
    at, _ = report.inspect([report.strip_watch(r) for r in raw])
    r = raw[report.MODES.index(arm)]
    assert report.crowded_path(r, arm, 16, at)['anchored'] == 1
    report.check_reproduction(r, report.strip_watch(r), 16)
    def stages(data):
        return {(x['stage'], x['row_kind']):x for x in report.milestones(data, arm, raw[0]['output_ids'], at)}
    by = stages(r)
    assert by['before_first_verify','common_mask']['agreeing_rows'] == 5
    assert by['after_first_verify','common_mask']['agreeing_rows'] == 6
    assert by['commit','mask_victim']['agreement'] == 1.
    assert all(x['compared_rows'] == x['eligible_rows'] for x in by.values())
    first_verify = next(i for i,e in enumerate(r['events']) if e['stage']=='verify')
    truncated = stages(dict(r, events=r['events'][:first_verify]))
    assert all(x['missing_stage_requests'] == 1 and not x['compared_rows'] and x['agreement']=='' for x in truncated.values())
    missing_commit = stages(dict(r, events=r['events'][:-1]))
    assert missing_commit['commit','common_mask']['missing_stage_requests']==1
    assert missing_commit['after_first_verify','common_mask']['compared_rows']==6
    short = list(report.milestones(r, arm, raw[0]['output_ids'][:10], at))
    assert all(x['no_base_final_id_rows']>0 for x in short)


@pytest.mark.parametrize('mutation', ['screen','early','late','risk','first','salt','digest','layer','physical','later_swap'])
def test_crowded_rejects_corrupt_evidence(mutation):
    r = crowded_case(dict(index=1, dataset='gsm8k', task_id='toy', seed=3))[1]
    at = next(i for i,e in enumerate(r['events']) if 'intervention' in e)
    e, plan = r['events'][at], r['events'][at]['intervention']
    if mutation=='screen': del r['events'][at-1]['crowded_screen']
    if mutation=='early': r['events'][at-1]['intervention']=plan
    if mutation=='late': r['events'][at+1]['crowded_screen']=e['crowded_screen']
    if mutation=='risk': plan['risk_score'] = -9.
    if mutation=='first': plan['victim'] = min(set(plan['base']) & set(plan['masks']))
    if mutation=='salt': plan['random_candidate'] = -1
    if mutation=='digest': plan['pre_sha256'] = 'z'*64
    if mutation=='layer': r['events'][-1]['layer_physical_q'][-1]-=1
    if mutation=='physical': r['events'][at+1]['actual_q']-=1
    if mutation=='later_swap': r['events'][at+1]['reserved_extra_edit_positions']=[16]
    with pytest.raises(ValueError):
        report.crowded_path(r,'rank',16,at)


def test_crowded_two_shard_cli(tmp_path):
    history, pair, rand, out = [tmp_path / name for name in ('history','pair','rand','out')]
    samples = report.read_json(report.DEFAULT_MANIFEST)['samples']
    for d, names in ((history,[report.NOEDIT]),(pair,report.ONCE[:2]),(rand,report.ONCE[2:])):
        d.mkdir()
        emit(d/'qualified.json',dict(model='llada21',matched=30,full_width=32))
        emit(d/'run.json',dict(model='llada21',block=32,batch_size=1,cap=2048,seed_offset=0, policies=names,
                               context_weights_sha256=report.gate.WEIGHTS, arms=[dict(budget=8,fresh=0,
                               mask=8 if d==history else 7,selection=p.split('_',4)[4]) for p in names]))
        for p in names:
            emit(d/f'complete-{p}.json',dict(policy=p,requests=30))
    for s in samples:
        i = s['index']
        raw = crowded_case(s, allow=i%2==0)
        full = dict(raw[0], policy='fullwidth')
        for d in (history,pair,rand):
            emit(d/f'{i:02d}-fullwidth.json',full)
        old = dict(raw[0], policy=report.NOEDIT, events=[{k:v for k,v in e.items() if k not in report.OBSERVER}
                                                         for e in raw[0]['events']])
        emit(history/f'{i:02d}-{report.NOEDIT}.json',old)
        for r,d in zip(raw,(pair,pair,rand)):
            emit(d/f"{i:02d}-{r['policy']}.json",r)
    args = [sys.executable,'-B','-m','tools.beneficiary_rows','--crowded','--historical',str(history),
            '--run',str(pair),'--run',str(rand),'--out',str(out)]
    proc = subprocess.run(args,capture_output=True,text=True)
    assert proc.returncode==0, proc.stderr
    assert len(list(out.glob('*.csv')))==7
    with (out/f'{report.PREFIX}-secondary-summary.csv').open() as f:
        r = next(r for r in csv.DictReader(f) if (r['arm'],r['stage'],r['dataset'],r['row_kind'])==
                 ('base','before_first_verify','all','common_mask'))
        assert [r[k] for k in ('requests','no_anchor_requests','missing_stage_requests','compared_rows','agreeing_rows')]==['30','15','0','90','75']
    for path, key, val in [(rand/'run.json','context_weights_sha256','0'*64),
                           (pair/'qualified.json','matched',29), (rand/'29-fullwidth.json','seed',0),
                           (history/f'29-{report.NOEDIT}.json','output_ids',[]),
                           (history/f'29-{report.NOEDIT}.json','unknown_archived_field',1)]:
        orig = report.read_json(path)
        emit(path,dict(orig,**{key:val}))
        with pytest.raises(ValueError):
            report.crowded([pair,rand],tmp_path/'fail',history)
        assert not (tmp_path/'fail').exists()
        emit(path,orig)

"""Local synthetic contracts only: no checkpoint, CUDA, tmux, or holder."""
from collections import Counter
from types import SimpleNamespace as NS

import pytest
import torch

if __package__ and __package__.startswith("experiments."):
    from . import llada_all_slots90 as r
    from .mask_pair_llada import head_to_tail_mask
else:
    try:
        from experiments.t2t.serving_slots import llada_all_slots90 as r
        from experiments.t2t.serving_slots.mask_pair_llada import head_to_tail_mask
    except ModuleNotFoundError:
        import llada_all_slots90 as r
        from mask_pair_llada import head_to_tail_mask


def test_coverage_and_priority():
    assert len(r.GROUPS) == 10
    coverage = Counter((arm, j) for arms, indices in r.GROUPS for arm in arms for j in indices)
    assert coverage == Counter((arm, j) for arm in r.ARMS for j in range(90))
    assert all(arms == r.PAIR and len(indices) == 30 for arms, indices in r.GROUPS[:3])
    assert all(len(arms) == 1 for arms, _ in r.GROUPS[3:])
    assert r.ARMS["replay"] == r.ARMS["replay_open"] == r.ARMS["replay_right"]
    assert sorted(r.SHIFT) == sorted(r.TRACE) == [4, 8, 16, 32, 64] and r.SHIFT != r.TRACE


class Probe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, *, attention_mask, position_ids):
        self.seen.append(attention_mask.clone())


@pytest.mark.parametrize("arm,opened", [("replay", 0), ("replay_open", 64)])
def test_hook_on_actual_geometry_not_offer_or_tail_query(arm, opened):
    model = Probe()
    state = NS(block=1, prompt_length=35, x=torch.zeros(1, 96), mask_id=0,
               last_query=[], last_write=[], committed_end=32, context_version=0,
               rng_cpu=torch.tensor([1]), rng_cuda=torch.tensor([]), events=[], offer_tick=0)
    original = []

    def run(s, clock):
        for tick, (block, rows, end) in enumerate(((1, [0, 1], 32), (1, [32, 35, 39, 67], 96),
                                                  (1, [70, 71], 96), (2, [70, 71], 96))):
            s.block, s.offer_tick = block, tick
            p = torch.tensor(rows)
            mask = (p[:, None]//32 >= torch.arange(end)[None, :]//32)[None, None]
            original.append(mask.clone())
            model(position_ids=p[None], attention_mask=mask)
            assert torch.equal(mask, original[-1])
            assert clock(NS(tick=tick)) == r.TRACE[tick]
            s.events.append(dict(block=block, absolute_positions=rows, visible_kv=end))

    stamp = r.observe(NS(model=model, run=run), state, arm, head_to_tail_mask)
    assert len(stamp) == 64 and not model._forward_pre_hooks
    assert [e["opened_edges"] for e in state.events] == [0, opened, 0, 0]
    for i, m in enumerate(original):
        if i == 1 and opened:
            m[..., 1:3, 64:96] = True
        assert torch.equal(m, model.seen[i])


def event(tick, action, *, arm="replay", debt=(), pos=None, end=32):
    offer = r.ARMS[arm][tick % len(r.ARMS[arm])]
    reason = {"head_bootstrap": "bootstrap", "head_verify": "verify", "commit_head": "commit",
              "fused_growth": "bootstrap"}.get(action)
    width = 64 if action == "fused_growth" else 32 if reason else 0
    dense = list(range(width))
    pos = dense if pos is None else pos
    q = len(pos)
    geom = r.geometry(0, 3, pos, end)
    refresh = [v for v in debt if v >= 32]
    sampled = [] if action == "commit_head" else [v for v in pos if v not in refresh]
    over, dover = max(0, q-offer), max(0, width-offer)
    why = {reason: dover} if dover else {}
    if over > dover:
        why["mandatory_refresh"] = over-dover
    return dict(top_action=action, phase="commit" if action == "commit_head" else "denoise",
                stage=reason or "selected", block=0, context_version=0, offer_tick=tick,
                previous_offer=r.ARMS[arm][(tick-1) % len(r.ARMS[arm])] if tick else None,
                q_budget=offer, nominal_q=offer, absolute_positions=pos, visible_kv=end, F=1,
                layer_physical_q=[q]*20, physical_q=q, actual_q=q, q_tokens=q,
                dense_required_q=width, dense_required_positions=dense, dense_reason=reason,
                refresh_only_positions=refresh, refreshed_pending=list(debt), mandatory_q=len(debt),
                mandatory_head_positions=[v for v in debt if v < 32], mandatory_tail_positions=refresh,
                sampled_q=len(sampled), sampled_positions=sampled, underfill=max(0, offer-q),
                overflow=over, overflow_reasons=why, mask_window=geom,
                opened_edges=geom["opened_edges"] if arm == "replay_open" and geom else 0)


def row(arm="replay", exposed=False):
    events = [event(0, "head_bootstrap", arm=arm)]
    if exposed:
        for tick in range(1, 4):
            events.append(event(tick, "head_verify", arm=arm))
        events.append(event(4, "fused_growth", arm=arm, end=64))
    events.append(event(len(events), "commit_head", arm=arm))
    result = dict(index=0, task_id="case", dataset="toy", seed=17000, model="llada21",
                  status="success", policy=arm, prompt_ids=[3]*3, special_ids=[0], eos_id=0,
                  output_ids=[9, 0], output_text="nine", output_tokens=1, eos=True, truncated=False,
                  prefill_rows=0, first_input_sha256="f"*64 if exposed else None, events=events)
    return tally(result)


def tally(row):
    events = row["events"]
    queries = Counter()
    for e in events:
        queries[e["stage"]] += e["physical_q"]
    q = sum(queries.values())
    row.update(generation_forwards=len(events), generation_query_rows=q, generation_physical_padded_rows=q,
               q_by_phase=dict(queries), forwards_by_phase=dict(Counter(e["stage"] for e in events)),
               tpf=1/len(events), token_efficiency=1/q, max_q=max(e["physical_q"] for e in events),
               q_histogram=dict(Counter(str(e["physical_q"]) for e in events)))
    return row


def test_pair_exposure_and_no_exposure():
    on, off = row(), row("replay_open")
    assert r.paired(on, off) is None
    off["output_ids"][0] = 8
    with pytest.raises(ValueError, match="Unexposed"):
        r.paired(on, off)
    on, off = row(exposed=True), row("replay_open", exposed=True)
    off["output_ids"][0] = 8
    assert r.paired(on, off) == 4
    off["first_input_sha256"] = "a"*64
    with pytest.raises(ValueError, match="first exposure or input"):
        r.paired(on, off)
    off["first_input_sha256"] = on["first_input_sha256"]
    off["events"][0]["accepted"] = 1
    with pytest.raises(ValueError, match="pre-exposure"):
        r.paired(on, off)


def test_first_noncontiguous_geometry():
    on, off = row(exposed=True), row("replay_open", exposed=True)
    for result, mid in ((on, 6), (off, 7)):
        e = result["events"][4]
        result["events"][4] = event(4, "pair_denoise", arm=result["policy"],
                                     pos=[3, mid, 21, 33], end=64)
        assert e["mask_window"]
        tally(result)
    assert on["events"][4]["mask_window"] == off["events"][4]["mask_window"]
    with pytest.raises(ValueError, match="First geometry differs"):
        r.paired(on, off)


def test_true_slack_charges_parked_tail_first():
    arm = "replay_right"
    result = row(arm)
    masks, base, pool = [28, 29, 30], [28, 29, 30, 31], list(range(3, 28))
    e = event(1, "head_denoise", arm=arm, debt=[41, 42],
              pos=[26, 27, *base, 41, 42], end=64)
    e.update(observed_head_masks=masks, prefix_plan=dict(mode="right", quota=4, slack=2,
             zero_rows=base, zero_mask_rows=masks, zero_backfill_rows=[31], prefix_end=28,
             eligible_prefix_rows=pool, added_rows=[26, 27], evicted_rows=[], selected_rows=[26, 27, *base]))
    result["events"][1:] = [e, event(2, "commit_head", arm=arm)]
    assert r.audit(tally(result) | {"first_input_sha256": "c"*64}, arm)["optional"] == 2
    e["observed_head_masks"] = [22, *masks]
    with pytest.raises(ValueError, match="observed physical head"):
        r.audit(result, arm)


@pytest.mark.parametrize("key,value", [("offer_tick", 3), ("q_tokens", 4), ("layer_physical_q", [32]),
                                       ("overflow_reasons", {}), ("opened_edges", 1)])
def test_corrupt_physical_evidence(key, value):
    result = row()
    result["events"][0][key] = value
    with pytest.raises(ValueError):
        r.audit(result, "replay")


def pred_fixture(directory):
    directory.mkdir()
    policies = ["toptrace_" + t + "_" + v for t in ("32", "64", "32_32_64_64", "64_64_32_32")
                for v in ("fused", "split")]
    header = dict(model="llada21", suite="frozen90", samples_sha256=r.SAMPLES_SHA)
    r.save(directory/"run.json", dict(header, formal=True, requests=90, gen_length=r.CAP,
                                      limit=None, policies=policies))
    r.save(directory/"qualified.json", dict(model="llada21", suite="frozen90", completed_requests=90,
                                           gen_length=r.CAP, policies=policies, trace_contract_checked=True))
    r.save(directory/"top-transition-audit.json", dict(header, requests=90, checks={"all_rows_audited": True},
                                                       arms={p: {"requests": 90} for p in policies}))
    for p in policies:
        r.save(directory/f"complete-{p}.json", dict(policy=p, requests=90))


def test_offline_mask_only_after_three_sealed_leases(tmp_path, monkeypatch):
    pred = tmp_path/r.PRED.name
    pred_fixture(pred)
    holder = dict(uuid=r.UUID, sole_holder_pid=29)
    reg = dict(predecessor=r.predecessor(pred), holder=holder, sources={})
    study = tmp_path/"new"
    study.mkdir()
    r.save(study/"registered.json", reg)
    prev = r.digest(reg)
    tasks = [dict(index=j, seed=17000+j, task_id=str(j), dataset="toy") for j in range(90)]
    for i in range(3):
        out, files = r.folder(study, i), {}
        out.mkdir()
        r.save(out/"run.json", r.spec(i))
        for index in r.GROUPS[i][1]:
            for arm in r.PAIR:
                name = r.filename(index, arm)
                r.save(out/name, row(arm, exposed=index % 2 == 0) | tasks[index])
                files[name] = r.sha(out/name)
        work = dict(spec_sha256=r.digest(r.spec(i)), files=files)
        r.save(out/"worker-complete.json", work)
        receipt = dict(spec_sha256=r.digest(r.spec(i)), worker_sha256=r.digest(work),
                       previous_sha256=prev, launcher_exit=0, holder=holder)
        r.save(out/"protected-complete.json", receipt)
        prev = r.digest(receipt)
    judge = lambda text, _: {"correct": text == "nine"}
    monkeypatch.setattr(r, "holder", lambda: pytest.fail("Offline scoring queried GPU"))
    score = r.score_files(study, tasks, judge, mask_only=True)
    summary = score["summary"][0]
    assert (summary["requests"], summary["correct"], summary["exposed"]) == (90, 90, 45)
    assert [p["requests"] for p in score["contrasts"][:3]] == [90, 45, 45]
    assert score["protected_receipt_sha256"] == prev and (study/"mask-pair-summary.csv").exists()
    with pytest.raises(FileNotFoundError):
        r.score_files(study, tasks, judge)
    for i in range(3, len(r.GROUPS)):
        (arm,), indices = r.GROUPS[i]
        out = r.folder(study, i)
        out.mkdir()
        r.save(out/"run.json", r.spec(i))
        files = {}
        for j in indices:
            name = r.filename(j, arm)
            obj = row(arm) | tasks[j]
            r.save(out/name, obj)
            files[name] = r.sha(out/name)
            if arm in ("fixed32", "fixed64"):
                r.save(pred/f"{j:02d}-toptrace_{arm[5:]}_fused.json", obj)
        work = dict(spec_sha256=r.digest(r.spec(i)), files=files)
        r.save(out/"worker-complete.json", work)
        receipt = dict(spec_sha256=r.digest(r.spec(i)), worker_sha256=r.digest(work),
                       previous_sha256=prev, launcher_exit=0, holder=holder)
        r.save(out/"protected-complete.json", receipt)
        prev = r.digest(receipt)
    r.save(study/"complete.json", dict(groups=10, requests=90, policies=list(r.ARMS),
                                       final_receipt_sha256=prev))
    full = r.score_files(study, tasks, judge)
    assert len(full["summary"]) == 9*3 and len(full["contrasts"]) == 5*3
    assert all(case["old_ids_equal"] and case["old_work_equal"] for arm in ("fixed32", "fixed64")
               for case in full["cases"][arm])
    r.save(r.folder(study, 0)/r.filename(0, "replay"), {"modified": True})
    with pytest.raises(ValueError, match="digest mismatch"):
        r.score_files(study, tasks, judge, mask_only=True)
    with pytest.raises(ValueError, match="return receipt"):
        r.return_receipt(study, 1, "wrong previous receipt")
    (r.folder(study, 2)/"protected-complete.json").unlink()
    with pytest.raises(FileNotFoundError):
        r.return_receipt(study, 2, prev)
    (pred/"complete-toptrace_32_split.json").unlink()
    with pytest.raises(FileNotFoundError):
        r.predecessor(pred)


def test_sole_holder_only(monkeypatch):
    import json
    values = iter((json.dumps([dict(running=True, pid=3, gpu_status=dict(uuid=r.UUID))]).encode(),
                   b"3\n4\n"))
    monkeypatch.setattr(r.subprocess, "check_output", lambda args: next(values))
    with pytest.raises(ValueError, match="solely held"):
        r.holder()

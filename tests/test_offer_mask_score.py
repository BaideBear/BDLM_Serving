from copy import deepcopy
import json

import pytest

from experiments.t2t.serving_slots import offer_mask_score as mod


HASH = "a" * 64
GEMMA = "diffusiongemma"
PAIR = mod.Pair(32, "q32", "q32_unmask32")


def task(i):
    return dict(index=i, dataset=mod.DATASETS[i % 3], task_id=f"synthetic/{i}", seed=17000+i)


def stamp(first, rounds, edges):
    return dict(first_tick=first, pretrace_sha256=HASH,
                geometry_sha256=HASH if first is not None else mod.NULL_DIGEST,
                rounds=rounds, edge_pairs=edges)


def gemma(i, arm, offers=(32, 32), hits=(), target=32, *, answer="synthetic-a"):
    trace = [dict(tick=tick, offer=offer, q=12, opportunity=tick in hits,
                  encoded=[0] if tick == 0 else [], head=0, phase="sparse",
                  prefix=0, strict_edits=0)
             for tick, offer in enumerate(offers)]
    on_target = [t for t in hits if offers[t] == target]
    f, q = len(trace) + 1, 12*len(trace) + 256
    return dict(**task(i), model=GEMMA, policy=arm, status="success",
                input_sha256=HASH, prefill_q=64, output_ids=[1], output_text=answer,
                output_tokens=1, generation_forwards=f, generation_physical_padded_rows=q,
                encoder_q=256, decoder_q=12*len(trace), tpf=1/f, token_efficiency=1/q,
                trace=trace, offer_mask_witness={str(target): stamp(
                    min(on_target) if on_target else None, len(on_target), 5*len(on_target))})


def gemma_pair(hits=(1,)):
    return gemma(0, PAIR.control, hits=hits), gemma(0, PAIR.trial, hits=hits)


def audit(a, b, fields=None):
    return mod.audit_pair(a, b, task(0), GEMMA, PAIR, fields or mod.default_fields(GEMMA), (32,))


def test_exposed_allows_first_tick_outcome_and_later_trajectory_to_change():
    a = gemma(0, PAIR.control, (32, 32), (0,))
    b = gemma(0, PAIR.trial, (32, 32, 32), (0, 2), answer="synthetic-b")
    b["trace"][0]["strict_edits"] = 2
    wa, wb = audit(a, b)
    assert wa[0] == wb[0] == 0
    assert wa[3:] == (1, 5)
    assert wb[3:] == (2, 10)


@pytest.mark.parametrize(("mutate", "message"), [
    (lambda r: r.update(seed=0), "seed"),
    (lambda r: r.update(task_id="other"), "task ID"),
    (lambda r: r.update(model="llada21"), "Wrong policy, model"),
    (lambda r: r.update(input_sha256="b"*64), "Different paired input"),
    (lambda r: r["trace"][0].update(strict_edits=1), "before first intervention"),
    (lambda r: r["trace"][1].update(head=1), "First intervention geometry"),
    (lambda r: r["trace"][1].update(offer=64), "registered offer schedule"),
    (lambda r: r["trace"][1].update(tick=2), "offer clock"),
    (lambda r: r.update(encoder_q=0), "does not reconstruct"),
    (lambda r: r["offer_mask_witness"]["32"].update(first_tick=0), "target exposure trace"),
    (lambda r: r["offer_mask_witness"]["32"].update(rounds=2), "target exposure trace"),
    (lambda r: r["offer_mask_witness"]["32"].update(pretrace_sha256="b"*64), "pretrace"),
    (lambda r: r["offer_mask_witness"]["32"].update(geometry_sha256="b"*64), "geometry"),
])
def test_reject_corrupted_pair(mutate, message):
    a, b = gemma_pair()
    mutate(b)
    with pytest.raises(ValueError, match=message):
        audit(a, b)


@pytest.mark.parametrize("field", ["trace", "output_ids", "output_text", "truncated", "work"])
def test_unexposed_must_not_change(field):
    a, b = gemma_pair(())
    audit(a, b)
    if field == "trace":
        b[field][1]["strict_edits"] = 1
    elif field == "work":
        b.update(output_tokens=0, tpf=0, token_efficiency=0)
    else:
        b[field] = {"output_ids": [2], "output_text": "changed", "truncated": True}[field]
    with pytest.raises(ValueError, match="Unexposed"):
        audit(a, b)


def test_custom_fields_and_off_target_opening():
    pair = mod.Pair(64, "replay", "replay_unmask64")
    a, b = [gemma(0, arm, (32, 64), (0, 1), 64) for arm in (pair.control, pair.trial)]
    for row, trial in ((a, False), (b, True)):
        row["proof"] = row.pop("offer_mask_witness")
        for e in row["trace"]:
            e.update(eligible=5, actual=5 if trial and e["offer"] == 64 else 0, arm_note=row["policy"])
    fields = mod.Fields("eligible", "actual", "proof", ("arm_note",))
    assert mod.audit_pair(a, b, task(0), GEMMA, pair, fields)[1][3:] == (1, 5)
    b["trace"][0]["actual"] = 5
    with pytest.raises(ValueError, match="target offer"):
        mod.audit_pair(a, b, task(0), GEMMA, pair, fields)


def test_llada_events_keep_prefill_in_prefix_and_check_actual_edges():
    pair = mod.Pair(8, "fixed8", "fixed8_unmask8")
    events = [
        dict(phase="prefill", F=1, physical_q=32, actual_q=32),
        dict(phase="sparse", F=1, physical_q=4, actual_q=4, offer_tick=0,
             q_budget=8, mask_window={"opened_edges": 16}, opened_edges=0),
    ]
    a = dict(**task(0), model="llada21", policy=pair.control, status="success", events=events,
             prompt_ids=[1]*32, prefill_rows=32, special_ids=[2], eos_id=2,
             output_text="synthetic-a", output_ids=[1], output_tokens=1, generation_forwards=1,
             generation_physical_padded_rows=4, offer_mask_witness={"8": stamp(0, 1, 16)})
    b = deepcopy(a)
    b.update(policy=pair.trial)
    b["events"][1]["opened_edges"] = 16
    fields = mod.default_fields("llada21")
    assert mod.audit_pair(a, b, task(0), "llada21", pair, fields)[1][3:] == (1, 16)
    b["events"][0]["context_version"] = 0
    with pytest.raises(ValueError, match="before first intervention"):
        mod.audit_pair(a, b, task(0), "llada21", pair, fields)


def test_metrics_keep_macro_tpf_and_micro_efficiency_distinct():
    a, b = (False, 12, 3, 48, 1, 5), (True, 9, 3, 18, 2, 10)
    c = (True, 5, 1, 10, 0, 0)
    rows = mod.summarize([(0, "gsm8k", a, b), (30, "math500", c, c)], draws=25)
    full, exposed, unexposed = rows[:3]
    assert [r["requests"] for r in rows[::3]] == [2, 1, 1]
    assert (full["control"]["Q"], full["trial"]["Q"], full["delta"]["F"]) == (58, 28, 0)
    assert full["control"]["macro_TPF"] == 4.5
    assert full["control"]["micro_efficiency"] == 17/58
    assert full["trial"]["micro_efficiency"] == 14/28
    assert full["delta"]["trigger_rounds"] == 1
    assert full["rescued"] == 1 and full["lost"] == 0
    assert full["paired_ci95"]["quality"] == [.5, .5]
    assert full["paired_ci95"]["micro_efficiency"] == pytest.approx([.5-17/58]*2)
    assert exposed["requests"] == 1 and unexposed["paired_ci95"]["quality"] == [0, 0]
    assert rows[7]["scope"] == "exposed" and rows[7]["paired_ci95"] is None


def test_shared_control_and_registered_full90_coverage(tmp_path):
    pairs = [mod.Pair(q, "replay", f"replay_unmask{q}") for q in (32, 64, 128, 256)]
    arms = [pairs[0].control, *(p.trial for p in pairs)]
    tasks = [task(i) for i in range(90)]
    offers = mod.REPLAYS[GEMMA]
    for t in tasks:
        i = t["index"]
        for arm in arms:
            row = gemma(i, arm, offers, (2, 3, 4, 5) if i < 45 else ())
            row["offer_mask_witness"] = {str(p.target): stamp(
                offers.index(p.target) if i < 45 else None, int(i < 45), 5*int(i < 45))
                for p in pairs}
            (tmp_path/f"{i:02d}-{arm}.json").write_text(json.dumps(row))
    calls = []

    def judge(text, t):
        calls.append(t["index"])
        return {"correct": True}

    assert mod.discover_pairs(tmp_path) == pairs
    meta = dict(samples_sha256=HASH, indices=list(range(90)),
                unmask_only_on_offer={arms[0]: None, **{p.trial: p.target for p in pairs}})
    mod.check_registration(meta, pairs, HASH)
    records = mod.score_group(tmp_path, GEMMA, pairs, tasks, judge, draws=5)
    assert len(calls) == 450
    assert [r["target"] for r in records] == [32, 64, 128, 256]
    for record in records:
        report = record["statistics"]
        assert [r["requests"] for r in report[::3]] == [90, 30, 60]
        assert [r["trial"]["trigger_requests"] for r in report[::3]] == [45, 30, 15]
    assert "synthetic-a" not in json.dumps(records)
    for p in tmp_path.glob("*-replay_unmask256.json"):
        p.unlink()
    assert mod.discover_pairs(tmp_path, meta) == pairs
    with pytest.raises(ValueError, match="exactly 90.*unmask256"):
        mod.score_group(tmp_path, GEMMA, pairs, tasks, judge, draws=5)
    assert len(calls) == 450


def test_cli_fixed_pair_writes_only_aggregate(tmp_path, monkeypatch):
    from bdlm_serving.evaluation import frozen90, scoring

    tasks = [task(i) for i in range(90)]
    monkeypatch.setattr(frozen90, "load_tasks", lambda _: tasks)
    monkeypatch.setattr(scoring, "score", lambda text, _: {"correct": text == "synthetic-a"})
    for t in tasks:
        i = t["index"]
        for arm in (PAIR.control, PAIR.trial):
            answer = "synthetic-b" if i == 0 and arm == PAIR.trial else "synthetic-a"
            row = gemma(i, arm, hits=(1,) if i == 0 else (), answer=answer)
            (tmp_path/f"{i:02d}-{arm}.json").write_text(json.dumps(row))
    samples, output = tmp_path/"samples.json", tmp_path/"result.json"
    samples.write_text("{}")
    args = ["--group", str(tmp_path), "--samples", str(samples), "--pair",
            "32", PAIR.control, PAIR.trial, "--draws", "5", "--out", str(output)]
    assert mod.main(args) == 0
    payload = output.read_text()
    assert "synthetic" not in payload
    pair, = json.loads(payload)["pairs"]
    full = pair["statistics"][0]
    assert pair["offers"] == [32]
    assert full["control"]["correct"] == 90 and full["trial"]["correct"] == 89
    assert full["lost"] == 1 and full["trial"]["trigger_rounds"] == 1
    assert full["delta"]["quality"] == pytest.approx(-1/90)
    with pytest.raises(SystemExit, match="2"):
        mod.main(args)
    assert output.read_text() == payload

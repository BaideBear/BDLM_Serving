"""Synthetic CPU tests for the protected Q64 spawn extension (no GPU/queue)."""
from pathlib import Path

import pytest

from experiments.t2t.serving_slots import spawn_frozen as sf

TASKS = [
    dict(index=0, dataset="gsm8k", task_id="t0", seed=17000),
    dict(index=10, dataset="math500", task_id="t10", seed=17010),
    dict(index=20, dataset="humaneval", task_id="t20", seed=17020),
]


def _event(*, phase="denoise", stage="bootstrap", block=1, positions, q, end,
           tick=None, previous=None, sampled=None, refresh=(), pending=(),
           admission=None, action=None):
    positions = list(positions)
    sampled = list(positions) if sampled is None else list(sampled)
    return dict(
        phase=phase, stage=stage, block=block, absolute_positions=positions,
        physical_q=q, actual_q=q, q_tokens=q, visible_kv=end, F=1,
        layer_physical_q=[q] * sf.reg.LAYERS,
        offer_tick=tick, q_budget=64 if tick is not None else None,
        nominal_q=64 if tick is not None else None, previous_offer=previous,
        overflow=max(0, q - 64), underfill=max(0, 64 - q),
        sampled_positions=sampled, sampled_q=len(sampled),
        refresh_only_positions=list(refresh), refreshed_pending=list(pending),
        tail_admission=admission, top_action=action)


def _row(index, arm, *, correct=True, admission="open"):
    events = [
        _event(phase="prefill", stage="prefill", positions=range(32), q=32,
               end=32, sampled=[]),
        _event(stage="bootstrap", positions=range(32, 96), q=64, end=96,
               tick=0, previous=None, admission=admission, action="pair_bootstrap"),
        _event(phase="commit", stage="commit", positions=range(32, 64), q=32,
               end=64, tick=1, previous=64, sampled=[], action="commit_head"),
    ]
    return dict(
        index=index, task_id=f"t{index}", dataset="gsm8k", seed=17000 + index,
        policy=arm, model="llada21", status="success", prompt_ids=[1] * 32,
        prefill_rows=32, events=events, generation_forwards=2,
        generation_physical_padded_rows=96, generation_query_rows=96,
        output_ids=[1, 1, 4], output_tokens=2, tpf=1.0,
        token_efficiency=2 / 96, max_q=64, q_histogram={"64": 1, "32": 1},
        forwards_by_phase={"bootstrap": 1, "commit": 1},
        q_by_phase={"bootstrap": 64, "commit": 32},
        opened_edges=0, opportunities=1, first="a" * 64,
        output_text="x", eos_id=4, special_ids=[0, 4], eos=True,
        truncated=False, correct=correct)


def test_audit_accepts_causal_rows_and_counts_admission():
    eager = sf.audit(_row(0, "eager-causal", admission="open"), "eager-causal")
    progress = sf.audit(_row(0, "progress80-causal", admission="defer"),
                        "progress80-causal")
    assert (eager["N"], eager["F"], eager["Q"]) == (2, 2, 96)
    assert (eager["opened"], eager["deferred"]) == (1, 0)
    assert (progress["opened"], progress["deferred"]) == (0, 1)


def test_progress80_mask_control_charges_same_rows_and_verifies_geometry():
    row = _row(0, "progress80-open")
    row["opened_edges"] = 32*32
    assert sf.audit(row, "progress80-open", opened_control=True)["Q"] == 96
    row["opened_edges"] -= 1
    with pytest.raises(ValueError, match="actual Q/K"):
        sf.audit(row, "progress80-open", opened_control=True)
    with pytest.raises(ValueError, match="registered"):
        sf.audit(row, "eager-open", opened_control=True)


@pytest.mark.parametrize("event_index,key,value,match", [
    (1, "q_budget", 63, "Wrong external offer clock"),
    (1, "physical_q", 32, "Invalid physical width"),
    (1, "layer_physical_q", [64] * 19, "Invalid physical width"),
    (1, "overflow", 1, "Unexplained physical overrun"),
    (2, "underfill", 0, "Unexplained physical overrun"),
    (1, "tail_admission", "bad", "Bad tail admission"),
    (1, "sampled_q", 63, "Wrong sampled/scatter partition"),
])
def test_audit_rejects_bad_offer_or_physical_width(event_index, key, value, match):
    row = _row(0, "eager-causal")
    row["events"][event_index][key] = value
    with pytest.raises(ValueError, match=match):
        sf.audit(row, "eager-causal")


@pytest.mark.parametrize("key,value,match", [
    ("opened_edges", 1, "must not open cross-block edges"),
    ("first", None, "First fingerprint"),
    ("q_histogram", {"64": 2}, "Wrong physical histogram"),
    ("generation_forwards", 3, "Wrong packaged metrics"),
    ("output_ids", [1, 4, 4], "Wrong output/EOS convention"),
])
def test_audit_rejects_bad_packaging(key, value, match):
    row = _row(0, "eager-causal")
    row[key] = value
    with pytest.raises(ValueError, match=match):
        sf.audit(row, "eager-causal")


def test_spec_is_original30_then_new60_with_two_causal_arms():
    first, second = sf.spec(0), sf.spec(1)
    assert first["cohort"] == "original30" and first["indices"] == list(range(30))
    assert second["cohort"] == "new60" and second["indices"] == list(range(30, 90))
    for spec in (first, second):
        assert spec["gpu"] == 4 and spec["gpu_uuid"] == sf.reg.UUID
        assert spec["samples_sha256"] == sf.reg.SAMPLES_SHA
        assert spec["launcher_sha256"] == sf.reg.LAUNCHER_SHA
        assert set(spec["arms"]) == {"eager-causal", "progress80-causal"}
        assert spec["arms"]["eager-causal"]["spawn_threshold"] is None
        assert spec["arms"]["progress80-causal"]["spawn_threshold"] == 0.8
        assert all(arm["mask"] == "causal" and not arm["opened"]
                   for arm in spec["arms"].values())


def _write_lease(study, i=0, *, bad_receipt=False, missing=None):
    out = sf.reg.folder(study, i)
    out.mkdir(parents=True)
    sf.reg.save(out / "run.json", sf.spec(i))
    files = {}
    for index in sf.GROUPS[i][1]:
        for arm in sf.ARMS:
            row = _row(index, arm)
            name = sf.reg.filename(index, arm)
            sf.reg.save(out / name, row)
            files[name] = sf.reg.sha(out / name)
    if missing:
        (out / missing).unlink()
        files.pop(missing)
    work = dict(spec_sha256=sf.reg.digest(sf.spec(i)), files=files)
    sf.reg.save(out / "worker-complete.json", work)
    receipt = dict(worker_sha256=sf.reg.digest(work),
                   spec_sha256=sf.reg.digest(sf.spec(i)),
                   previous_sha256="0" * 64, launcher_exit=0,
                   holder=dict(uuid=sf.reg.UUID, sole_holder_pid=1))
    if bad_receipt:
        receipt["worker_sha256"] = "0" * 64
    sf.reg.save(out / "protected-complete.json", receipt)
    return out


def test_qualify_original30_seals_the_first_lease(tmp_path):
    study = tmp_path / "study"
    _write_lease(study)
    summary = sf.qualify_original30(study)
    assert set(summary) == set(sf.ARMS)
    assert all(row["requests"] == 30 for row in summary.values())
    assert all(row["N"] == 60 and row["F"] == 60 and row["Q"] == 2880
               for row in summary.values())
    qualified = sf.reg.read(sf.reg.folder(study, 0) / "qualified.json")
    assert qualified["eager_sources"] == {"unknown": 30}


def test_reuse_eager_uses_qualified_fixed64_row(tmp_path, monkeypatch):
    monkeypatch.setattr(sf.reg, "STUDY", tmp_path / "allslots")
    lease = sf.reg.folder(sf.reg.STUDY, sf.FIXED64_GROUP)
    lease.mkdir(parents=True)
    historical = _row(0, "fixed64")
    for event in historical["events"]:
        event.pop("tail_admission")
    sf.reg.save(lease / sf.reg.filename(0, "fixed64"), historical)
    name = sf.reg.filename(0, "fixed64")
    sf.reg.save(lease / "worker-complete.json", {"files": {name: sf.reg.sha(lease/name)}})
    row = sf.reuse_eager(0, TASKS[0], [1] * 32)
    assert row["policy"] == "eager-causal"
    assert row["eager_source"] == "fixed64-reuse"
    assert sf.audit(row, "eager-causal")["opened"] == 1
    changed = sf.reg.read(lease/name)
    changed["output_text"] += " different"
    sf.reg.save(lease/name, changed)
    with pytest.raises(ValueError, match="digest mismatch"):
        sf.reuse_eager(0, TASKS[0], [1] * 32)


def test_reuse_eager_raises_on_missing_or_corrupt_qualified_row(tmp_path, monkeypatch):
    monkeypatch.setattr(sf.reg, "STUDY", tmp_path / "allslots")
    lease = sf.reg.folder(sf.reg.STUDY, sf.FIXED64_GROUP)
    lease.mkdir(parents=True)
    with pytest.raises(ValueError, match="Missing qualified fixed64 eager row"):
        sf.reuse_eager(0, TASKS[0], [1] * 32)
    row = _row(0, "fixed64")
    row["q_histogram"] = {"64": 2}
    name = sf.reg.filename(0, "fixed64")
    sf.reg.save(lease / name, row)
    sf.reg.save(lease / "worker-complete.json", {"files": {name: sf.reg.sha(lease/name)}})
    with pytest.raises(ValueError, match="Wrong physical histogram"):
        sf.reuse_eager(0, TASKS[0], [1] * 32)


def test_reuse_eager_returns_none_when_lease_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(sf.reg, "STUDY", tmp_path / "allslots")
    assert sf.reuse_eager(0, TASKS[0], [1] * 32) is None


def test_qualify_original30_rejects_incomplete_or_bad_receipt(tmp_path):
    study = tmp_path / "study"
    _write_lease(study, missing=sf.reg.filename(3, "progress80-causal"))
    with pytest.raises(ValueError, match="Incomplete or wrong lease manifest"):
        sf.qualify_original30(study)
    study2 = tmp_path / "study2"
    _write_lease(study2, bad_receipt=True)
    with pytest.raises(ValueError, match="Missing original30 protected receipt"):
        sf.qualify_original30(study2)


def test_summarize_records_all_cohorts_and_paired_effects():
    rows = {arm: {} for arm in sf.ARMS}
    for index in range(90):
        for arm in sf.ARMS:
            rows[arm][index] = _row(index, arm, correct=index % 2 == (arm == "eager-causal"))
    summary = sf.summarize(rows)
    eager30 = next(r for r in summary if r["cohort"] == "original30"
                   and r["arm"] == "eager-causal")
    assert eager30["requests"] == 30 and eager30["correct"] == 15
    paired = next(r for r in summary if r["cohort"] == "full90"
                  and r.get("baseline") == "eager-causal")
    assert paired["rescued"] == 45 and paired["lost"] == 45
    assert paired["macro_tpf_delta"] == 0 and paired["physical_Q_delta"] == 0


def test_audit_accepts_real_multiblock_commit_row():
    path = (sf.reg.DATA / "llada21-q64-admission-mask-pilot3"
            / "00-progress80-causal.json")
    if not path.is_file():
        pytest.skip("real Q64 pilot row not present")
    from experiments.t2t.serving_slots.llada_mask_pilot import CKPT
    if not Path(CKPT).is_dir():
        pytest.skip("checkpoint not present")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True,
                                        local_files_only=True)
    row = sf.reg.read(path)
    eos = tok.eos_token_id
    mask = tok.convert_tokens_to_ids("<|mask|>")
    row.update(model="llada21", dataset="gsm8k", eos_id=eos,
               special_ids=sorted(set(tok.all_special_ids) | {mask, eos}))
    result = sf.audit(row, "progress80-causal")
    assert result["F"] == row["generation_forwards"]
    assert result["Q"] == row["generation_physical_padded_rows"]

"""CPU-only tests for the Gemma one-seat repeat report."""

import csv
import json

import pytest

from tools import seat_repeat_report as report


OFFSET = 1_000_003
BASE_POLICY = "t2t_q32_f0_m32_hard_once_base"
MASK_POLICY = "t2t_q32_f0_m32_hard_once_mask"
RANK_POLICY = "t2t_q32_f0_m32_hard_once_rank"


def task(index, seed):
    return dict(index=index, dataset="gsm8k", task_id=f"gsm8k/{index}",
                seed=seed, reference="1", prompt=f"p{index}", source={})


TASKS = [task(0, 100), task(1, 101), task(2, 102)]
REPEAT_TASKS = [task(0, 100 + OFFSET), task(1, 101 + OFFSET), task(2, 102 + OFFSET)]


def plan(stratum, victim, candidate, base, targeted, executed, *, pre="pre",
         after="after", queried=True, unresolved=None, accepted=False):
    return dict(
        base=list(base), targeted=list(targeted), random=list(targeted),
        victim=victim, candidate=candidate, random_candidate=candidate,
        stratum=stratum, fresh=[], masks=[], clock=0,
        pre_sha256=pre, post_rng_sha256="post", after_sha256=after,
        executed=list(executed), victim_queried=queried,
        victim_unresolved_before=(stratum == "mask") if unresolved is None else unresolved,
        victim_accepted=accepted,
    )


def row(index, seed, *, n=10, f=4, q=40, policy="fullwidth", text="yes",
        events=None, ids=None):
    return dict(
        status="success", output_ids=ids if ids is not None else [index, n, f],
        output_text=text, output_tokens=n, generation_forwards=f,
        generation_query_rows=q, generation_physical_padded_rows=q, max_q=256,
        tpf=n / f, token_efficiency=n / q, eos=True, truncated=False,
        prefill_rows=8, forwards_by_phase={"bootstrap": f},
        q_by_phase={"bootstrap": q}, events=events or [], physical_events=[],
        t2t_edits=0, index=index, task_id=f"gsm8k/{index}", dataset="gsm8k",
        seed=seed, model="gemma", policy=policy,
    )


def event(plan_value):
    return dict(stage="selected", block=0, clock=0, q=32,
                positions=plan_value["executed"], intervention=plan_value,
                trajectory_sha256=plan_value["after_sha256"],
                accepted_sha256=plan_value["after_sha256"])


def build_run(folder, meta, rows, seed_offset=0):
    folder.mkdir(parents=True)
    (folder / "run.json").write_text(json.dumps(
        dict(model="gemma", block=256, **meta, seed_offset=seed_offset)))
    for policy, items in rows.items():
        for item in items:
            (folder / f"{item['index']:02d}-{policy}.json").write_text(
                json.dumps(item))


def good_runs(tmp_path):
    """Build original, bridge and repeat runs with mask/slack/no-anchor cases."""
    # Request 0: MASK anchor, bridge mask matches archived rank and wins.
    base0 = [0, 1, 2]
    target0 = [0, 1, 3]
    plan_base0 = plan("mask", 2, 3, base0, target0, base0, pre="pre0",
                      after="after0", queried=True, unresolved=True, accepted=True)
    plan_mask0 = plan("mask", 2, 3, base0, target0, target0, pre="pre0",
                      after="after0", queried=False, unresolved=True, accepted=False)
    # Request 1: slack anchor, no swap, mask is identical to base.
    base1 = [4, 5]
    target1 = [4, 6]
    plan_base1 = plan("slack", 5, 6, base1, target1, base1, pre="pre1",
                      after="after1", queried=True, unresolved=False, accepted=False)
    plan_mask1 = plan("slack", 5, 6, base1, target1, base1, pre="pre1",
                      after="after1", queried=True, unresolved=False, accepted=False)
    # Request 2: no anchor.
    original_base = [
        row(0, 100, n=10, policy=BASE_POLICY, text="no", events=[event(plan_base0)]),
        row(1, 101, n=12, policy=BASE_POLICY, text="yes", events=[event(plan_base1)]),
        row(2, 102, n=8, policy=BASE_POLICY, text="yes"),
    ]
    original_rank = [
        row(0, 100, n=11, policy=RANK_POLICY, text="yes", events=[event(plan_mask0)]),
        row(1, 101, n=12, policy=RANK_POLICY, text="yes", events=[event(plan_mask1)]),
        row(2, 102, n=8, policy=RANK_POLICY, text="yes"),
    ]
    original_full = [row(0, 100, n=20), row(1, 101, n=20), row(2, 102, n=20)]
    bridge_base = [
        row(0, 100, n=10, policy=BASE_POLICY, text="no", events=[event(plan_base0)]),
        row(1, 101, n=12, policy=BASE_POLICY, text="yes", events=[event(plan_base1)]),
        row(2, 102, n=8, policy=BASE_POLICY, text="yes"),
    ]
    bridge_mask = [
        row(0, 100, n=11, policy=MASK_POLICY, text="yes", events=[event(plan_mask0)]),
        row(1, 101, n=12, policy=MASK_POLICY, text="no", events=[event(plan_mask1)]),
        row(2, 102, n=8, policy=MASK_POLICY, text="yes"),
    ]
    bridge_full = [row(0, 100, n=20), row(1, 101, n=20), row(2, 102, n=20)]

    # Repeat uses the same three cases at seed + OFFSET.
    def shifted(rows, policy):
        return [dict(item, seed=item["seed"] + OFFSET, policy=policy) for item in rows]

    repeat_base = shifted(bridge_base, BASE_POLICY)
    repeat_mask = shifted(bridge_mask, MASK_POLICY)
    repeat_full = [row(0, 100 + OFFSET, n=21), row(1, 101 + OFFSET, n=21),
                   row(2, 102 + OFFSET, n=21)]
    native = [row(0, 100 + OFFSET, n=21, policy="native256"),
              row(1, 101 + OFFSET, n=21, policy="native256"),
              row(2, 102 + OFFSET, n=21, policy="native256")]

    build_run(tmp_path / "original",
              dict(policies=[BASE_POLICY, RANK_POLICY]),
              {BASE_POLICY: original_base, RANK_POLICY: original_rank,
               "fullwidth": original_full})
    build_run(tmp_path / "bridge",
              dict(policies=[BASE_POLICY, MASK_POLICY]),
              {BASE_POLICY: bridge_base, MASK_POLICY: bridge_mask,
               "fullwidth": bridge_full})
    build_run(tmp_path / "repeat",
              dict(policies=[BASE_POLICY, MASK_POLICY]),
              {BASE_POLICY: repeat_base, MASK_POLICY: repeat_mask,
               "fullwidth": repeat_full}, seed_offset=OFFSET)
    build_run(tmp_path / "seat-repeat-native-v1", dict(policies=[]),
              {"native256": native}, seed_offset=OFFSET)
    return tmp_path


def read_csv(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def patch_tasks(monkeypatch):
    def fake_load_tasks(path, *, seed_offset=0):
        return REPEAT_TASKS if seed_offset else TASKS
    monkeypatch.setattr(report, "load_tasks", fake_load_tasks)


def test_main_scores_and_audits_the_repeat(tmp_path, monkeypatch):
    patch_tasks(monkeypatch)
    monkeypatch.setattr(report, "score", lambda text, task: {"correct": text == "yes"})
    good_runs(tmp_path)
    # Old marks must be ignored: put inverted values in every marks file.
    for folder in ("original", "bridge", "repeat"):
        for path in (tmp_path / folder).glob("*.json"):
            if path.name == "run.json":
                continue
            policy = json.loads(path.read_text())["policy"]
            if policy == "fullwidth":
                continue
            (tmp_path / folder / f"marks-{policy}.json").write_text(
                json.dumps({"0": 0, "1": 1, "2": 0}))

    out = tmp_path / "out"
    report.main([
        "--samples", str(tmp_path / "samples.json"),
        "--bridge", str(tmp_path / "bridge"),
        "--original", str(tmp_path / "original"),
        "--repeat", str(tmp_path / "repeat"),
        "--out", str(out),
    ])

    checks = read_csv(out / "seat-repeat-checks.csv")
    assert checks and all(row["passed"] == "1" for row in checks)
    summary = read_csv(out / "seat-repeat-summary.csv")
    by_key = {(row["seed_group"], row["arm"]): row for row in summary}
    original_mask = by_key[("original", "mask")]
    assert original_mask["correct"] == "2"
    assert original_mask["paired_wins"] == "1"
    assert original_mask["paired_losses"] == "1"
    assert original_mask["matching_victim_accepts_base"] == "1"
    repeat_mask = by_key[("repeat", "mask")]
    assert repeat_mask["correct"] == "2"
    assert repeat_mask["paired_wins"] == "1"
    assert repeat_mask["paired_losses"] == "1"
    requests = read_csv(out / "seat-repeat-requests.csv")
    base0 = next(r for r in requests
                 if r["seed_group"] == "original" and r["arm"] == "base" and r["index"] == "0")
    mask0 = next(r for r in requests
                 if r["seed_group"] == "original" and r["arm"] == "mask" and r["index"] == "0")
    assert base0["victim_queried"] == "True"
    assert base0["victim_accepted"] == "True"
    assert mask0["victim_queried"] == "False"
    assert mask0["victim_accepted"] == "False"
    assert mask0["victim_kind"] == "mask"


def test_audit_original_rejects_a_mask_anchor_that_does_not_replay_rank(tmp_path):
    good_runs(tmp_path)
    bridge = report.load_run(tmp_path / "bridge", ("hard_once_base", "hard_once_mask"),
                             TASKS, 0, "bridge")
    original = report.load_run(tmp_path / "original", ("hard_once_base", "hard_once_rank"),
                               TASKS, 0, "original")
    bridge["rows"]["mask"][0]["output_ids"] = [999]
    checks = []
    report.audit_original(original, bridge, TASKS, checks)
    failed = {c["check"] for c in checks if not c["passed"]}
    assert "original/00/mask_anchor_matches_old_rank" in failed


def test_audit_original_rejects_a_slack_anchor_that_swaps(tmp_path):
    good_runs(tmp_path)
    bridge = report.load_run(tmp_path / "bridge", ("hard_once_base", "hard_once_mask"),
                             TASKS, 0, "bridge")
    original = report.load_run(tmp_path / "original", ("hard_once_base", "hard_once_rank"),
                               TASKS, 0, "original")
    bridge["rows"]["mask"][1]["output_ids"] = [999]
    checks = []
    report.audit_original(original, bridge, TASKS, checks)
    failed = {c["check"] for c in checks if not c["passed"]}
    assert "original/01/slack_no_swap_identical" in failed


def test_audit_original_rejects_a_changed_paired_plan(tmp_path):
    good_runs(tmp_path)
    bridge = report.load_run(tmp_path / "bridge", ("hard_once_base", "hard_once_mask"),
                             TASKS, 0, "bridge")
    original = report.load_run(tmp_path / "original", ("hard_once_base", "hard_once_rank"),
                               TASKS, 0, "original")
    bridge["rows"]["mask"][0]["events"][0]["intervention"]["pre_sha256"] = "changed"
    checks = []
    report.audit_original(original, bridge, TASKS, checks)
    failed = {c["check"] for c in checks if not c["passed"]}
    assert "original/00/paired_plan" in failed


def test_load_run_rejects_a_wrong_seed(tmp_path):
    good_runs(tmp_path)
    path = tmp_path / "bridge" / f"00-{BASE_POLICY}.json"
    data = json.loads(path.read_text())
    data["seed"] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        report.load_run(tmp_path / "bridge", ("hard_once_base", "hard_once_mask"),
                        TASKS, 0, "bridge")


def test_score_rows_uses_the_actual_scorer_not_marks(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "score", lambda text, task: {"correct": text == "yes"})
    rows = [row(0, 100, text="yes"), row(1, 101, text="no")]
    (tmp_path / "marks-ignored.json").write_text(json.dumps({"0": 0, "1": 1}))
    assert report.score_rows(rows, TASKS[:2]) == [1, 0]

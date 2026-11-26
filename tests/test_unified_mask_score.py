"""Synthetic-fixture tests for the aggregate-only Gemma unified90 mask sidecar."""
import json

import pytest

from experiments.t2t.serving_slots import unified_mask_score as ums

TASKS = [
    dict(index=0, dataset="gsm8k", task_id="t0", seed=17000),
    dict(index=10, dataset="math500", task_id="t10", seed=17010),
    dict(index=20, dataset="humaneval", task_id="t20", seed=17020),
]


def case(index, dataset, on_correct, off_correct, *, on_tpf=2.0, off_tpf=2.0,
         on_n=10, off_n=10, on_f=5, off_f=5, on_q=50, off_q=50, exposed=True):
    return dict(index=index, dataset=dataset, on_correct=on_correct,
                off_correct=off_correct, on_N=on_n, off_N=off_n,
                on_F=on_f, off_F=off_f, on_Q=on_q, off_Q=off_q,
                on_tpf=on_tpf, off_tpf=off_tpf, exposed=exposed)


def _write_run(folder, *, suite="cpu-pilot", indices=None, arms=None,
               samples_sha=None, skip_complete=()):
    tasks = {task["index"]: task for task in TASKS}
    indices = [0, 10, 20] if indices is None else list(indices)
    if arms is None:
        arms = {"replay": ums.unified90.ARMS["replay"],
                "replay_open": ums.unified90.ARMS["replay_open"]}
    if samples_sha is None:
        samples_sha = ums.unified90.caps.SAMPLES_SHA
    (folder / "run.json").write_text(json.dumps(dict(
        model="diffusiongemma", suite=suite, device="cpu" if suite=="cpu-pilot" else "cuda:0",
        indices=indices, arms=arms,
        samples_sha256=samples_sha)))
    for policy in arms:
        if policy not in skip_complete:
            (folder / f"complete-{policy}.json").write_text(
                json.dumps(dict(policy=policy, requests=len(indices))))
    for index in (0, 10, 20):
        for policy, text in (("replay", "yes"), ("replay_open", "no")):
            task = tasks[index]
            (folder / f"{index:02d}-{policy}.json").write_text(json.dumps(dict(
                index=index, task_id=task["task_id"], dataset=task["dataset"],
                seed=task["seed"], policy=policy, input_sha256="same",
                first_opportunity=0, pretrace_sha256="pre",
                first_geometry_sha256="geom", output_ids=[1], trace=[{"tick": j,"q":10} for j in range(5)],
                output_tokens=10, generation_forwards=5,
                generation_physical_padded_rows=50, encoder_q=0, decoder_q=50,
                tpf=2.0, token_efficiency=0.2, status="success",
                output_text=text, truncated=False)))
    return tasks


def test_aggregate_exact_rescued_lost_and_efficiency():
    rows = [case(0, "gsm8k", True, False), case(1, "gsm8k", False, True),
            case(2, "math500", True, True), case(3, "humaneval", False, False)]
    agg = ums.aggregate(rows)
    assert (agg["requests"], agg["on_correct"], agg["off_correct"]) == (4, 2, 2)
    assert (agg["rescued"], agg["lost"]) == (1, 1)
    assert agg["on_macro_tpf"] == agg["off_macro_tpf"] == 2.0
    assert agg["on_eff"] == agg["off_eff"] == 10 / 50


def test_pilot3_is_exact_counts_only_without_intervals():
    rows = [case(0, "gsm8k", True, False),
            case(10, "math500", True, True, exposed=False),
            case(20, "humaneval", False, True)]
    ci = ums.intervals(rows, pilot=True)
    assert ci["tpf"] is ci["quality"] is ci["efficiency"] is None
    assert ci["method"] == "pilot3 exact counts only"
    table = ums.table_rows(rows, pilot=True)
    assert {row["scope"] for row in table} == {"all", "exposed", "unexposed"}
    assert all(row["macro_tpf_ci_low"] is None
               and row["ci_method"] == "pilot3 exact counts only" for row in table)


def test_gpu_intervals_are_dataset_stratified():
    rows = []
    for index in range(30):
        dataset = ("gsm8k", "math500", "humaneval")[index % 3]
        rows.append(case(
            index, dataset, index % 4 != 0, index % 5 != 0,
            on_tpf=2.2 if dataset == "gsm8k" else 1.9 if dataset == "math500" else 2.0,
            off_tpf=2.0))
    ci = ums.intervals(rows, pilot=False)
    assert ci["tpf"] is not None and ci["quality"] is not None
    assert ci["efficiency"] is not None
    assert ci["tpf"][2] <= ci["tpf"][0] <= ci["tpf"][3]
    full = next(row for row in ums.table_rows(rows, pilot=False)
                if row["cohort"] == "full90" and row["scope"] == "all")
    assert full["macro_tpf_ci_low"] is not None
    assert full["token_efficiency_ci_low"] is not None


def test_load_pair_and_pair_cases_use_committed_paired(tmp_path):
    _write_run(tmp_path)
    meta, pilot, on, off, indices, loaded = ums.load_pair(tmp_path, tasks=TASKS)
    assert pilot and on == "replay" and off == "replay_open"
    assert indices == [0, 10, 20] and set(loaded) == {0, 10, 20}
    rows = ums.pair_cases(
        tmp_path, on, off, indices, loaded,
        scorer=lambda text, task: {"correct": text == "yes"})
    assert len(rows) == 3
    assert all(row["on_correct"] and not row["off_correct"] for row in rows)
    agg = ums.aggregate(rows)
    assert agg["rescued"] == 3 and agg["lost"] == 0


def test_pair_cases_rejects_different_pre_intervention_input(tmp_path):
    _write_run(tmp_path)
    path = tmp_path / "00-replay_open.json"
    row = json.loads(path.read_text())
    row["input_sha256"] = "different"
    path.write_text(json.dumps(row))
    meta, pilot, on, off, indices, tasks = ums.load_pair(tmp_path, tasks=TASKS)
    with pytest.raises(ValueError, match="Different pre-intervention input"):
        ums.pair_cases(tmp_path, on, off, indices, tasks,
                       scorer=lambda text, task: {"correct": True})


@pytest.mark.parametrize("key,value,msg", [("seed", 19, "Wrong frozen input"),
                         ("tpf", 5., "Invalid physical work"),
                         ("decoder_q", 500, "Invalid physical work")])
def test_matching_corruption_on_both_arms_is_not_a_valid_pair(tmp_path,key,value,msg):
    _write_run(tmp_path)
    _, _, on, off, idx, tasks = ums.load_pair(tmp_path, tasks=TASKS)
    for policy in (on, off):
        path = tmp_path / f"00-{policy}.json"
        row = json.loads(path.read_text())
        row[key] = value
        path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match=msg):
        ums.pair_cases(tmp_path, on, off, idx, tasks, scorer=lambda *_: {"correct": True})


@pytest.mark.parametrize("kwargs,match", [
    (dict(arms={"replay": ums.unified90.ARMS["replay"],
                "q512": ums.unified90.ARMS["q512"]}), "exactly one causal-on"),
    (dict(arms={"replay_open": ums.unified90.ARMS["replay_open"],
                "q512_open": ums.unified90.ARMS["q512_open"]}), "exactly one causal-on"),
    (dict(arms={"replay": ums.unified90.ARMS["replay"],
                "q512_open": ums.unified90.ARMS["q512_open"]}), "mask-only"),
    (dict(arms={"replay": ums.unified90.ARMS["replay"],
                "replay_open": ums.unified90.ARMS["replay_open"],
                "q512": ums.unified90.ARMS["q512"]}), "exactly one on/off"),
    (dict(arms={"replay": [list(ums.unified90.ARMS["replay"][0]), False, {}],
                "replay_open": ums.unified90.ARMS["replay_open"]}),
     "disagrees with unified90.ARMS"),
    (dict(arms={"nope": ums.unified90.ARMS["replay"],
                "replay_open": ums.unified90.ARMS["replay_open"]}),
     "Unregistered policy"),
    (dict(samples_sha="0" * 64), "frozen90 cohort"),
    (dict(suite="cpu-pilot", indices=list(range(90))), "cpu-pilot must have indices"),
    (dict(suite="frozen90", indices=[0, 10, 20]), "frozen90 must have indices"),
    (dict(skip_complete=("replay_open",)), "Incomplete arm"),
])
def test_strict_pair_rejection(tmp_path, kwargs, match):
    _write_run(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=match):
        ums.load_pair(tmp_path, tasks=TASKS)

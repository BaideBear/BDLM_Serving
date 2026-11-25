"""CPU-only tests for tools/placement_comparison.py (scores-dir report)."""
import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import placement_comparison as pc


def sample(index):
    return dict(index=index, dataset="gsm8k", task_id="gsm8k/%d" % index, seed=1000 + index)


def score_row(model, policy, task, correct, *, nominal_q=4, mask=3, edit=1,
              N=4, F=1, Q=1):
    return dict(
        model=model, policy=policy, nominal_q=str(nominal_q), fresh="0",
        mask=str(mask), edit=str(edit), index=str(task["index"]),
        dataset=task["dataset"], task_id=task["task_id"], max_q="32",
        N=str(N), F=str(F), Q=str(Q), physical_Q=str(Q),
        correct=str(int(correct)), failed="0", truncated="0",
    )


def write_scores(folder, model, rows):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ("%s-online-requests.csv" % model)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def event(model, stage, positions, block=0, reserved=None, state=None):
    positions = list(positions)
    row = dict(stage=stage, block=block, reserved_extra_edit_positions=list(reserved or []))
    if model == "llada21":
        row.update(phase="denoise", absolute_positions=positions)
    else:
        row.update(positions=positions)
    row.update(state or {})
    return row


def raw(model, policy, task, correct, steps):
    events = [event(model, stage, positions, block=block, reserved=reserved, state=state)
              for stage, positions, block, reserved, state in steps]
    return dict(
        model=model, policy=policy, index=task["index"], dataset=task["dataset"],
        task_id=task["task_id"], seed=task["seed"], status="success",
        output_ids=[42] if correct else [41],
        output_text="\\boxed{42}" if correct else "\\boxed{41}",
        events=events,
    )


def write_run(folder, model, records):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for task_index, by_policy in records.items():
        for policy, row in by_policy.items():
            (folder / ("%02d-%s.json" % (task_index, policy))).write_text(json.dumps(row))
    return folder


def step(stage, positions, reserved=(), block=0, state=None):
    return (stage, positions, block, reserved, state)


def test_pair_metrics_reports_main_paired_metrics():
    model, policy = "llada21", "t2t_q4_f0_m3_hard_slack_placebo"
    tasks = [sample(0), sample(1)]
    placebo = [score_row(model, policy, tasks[0], True),
               score_row(model, policy, tasks[1], False)]
    reference = [score_row(model, "t2t_q4_f0_m3_hard_slack", tasks[0], False),
                 score_row(model, "t2t_q4_f0_m3_hard_slack", tasks[1], True)]
    metrics = pc._pair_metrics(placebo, reference)
    assert metrics["requests"] == 2
    assert metrics["correct"] == 1 and metrics["reference_correct"] == 1
    assert metrics["delta_correct"] == 0
    assert metrics["wins"] == 1 and metrics["losses"] == 1
    assert metrics["strict_success"] == 1 and metrics["reference_strict_success"] == 1
    assert metrics["tpf"] == pytest.approx(4.0)
    assert metrics["delta_tpf"] == pytest.approx(0.0)
    assert metrics["N"] == 8 and metrics["F"] == 2 and metrics["Q"] == 2
    assert metrics["token_efficiency"] == pytest.approx(4.0)
    assert metrics["q_per_f"] == pytest.approx(1.0)


def test_raw_alignment_first_difference_output_and_prefix_seats():
    model, policy = "llada21", "p"
    task = sample(0)
    left = raw(model, policy, task, True, [
        step("bootstrap", [0, 1], reserved=[0]),
        step("selected", [0, 2], reserved=[1, 2]),
    ])
    same = raw(model, policy, task, True, [
        step("bootstrap", [0, 1], reserved=[0]),
        step("selected", [0, 2], reserved=[1, 2]),
    ])
    late = raw(model, policy, task, True, [
        step("bootstrap", [0, 1], reserved=[0]),
        step("selected", [0, 3], reserved=[1]),
    ])
    result = pc.compare_request(model, left, same)
    assert result["positions_all_identical"] is True
    assert result["first_different_step"] == ""
    assert result["output_identical"] is True
    assert result["prefix_placebo_reserved_seats"] == 3
    assert result["prefix_reference_reserved_seats"] == 3
    assert result["prefix_reserved_counts_identical"] is True
    assert result["state_evidence"] == "missing_state"
    assert result["same_state_claim"] is False

    result = pc.compare_request(model, left, late)
    assert result["positions_all_identical"] is False
    assert result["first_different_step"] == 2
    assert result["first_different_stage"] == "selected"
    assert result["q_symmetric_difference"] == 2
    # The divergent step is outside the common Q prefix.
    assert result["prefix_reserved_compared_steps"] == 1
    assert result["prefix_reserved_mismatch_steps"] == 0
    assert result["prefix_placebo_reserved_seats"] == 1
    assert result["prefix_reference_reserved_seats"] == 1
    assert result["prefix_reserved_counts_identical"] is True


def test_raw_alignment_requires_state_evidence_for_same_state_claim():
    model, policy = "llada21", "p"
    task = sample(0)
    state = dict(fresh_positions=[1], resolved_positions=[0, 1])
    other = dict(fresh_positions=[], resolved_positions=[0, 1])
    left = raw(model, policy, task, True,
               [step("selected", [0, 1], reserved=[0], state=state)])
    right = raw(model, policy, task, True,
                [step("selected", [0, 2], reserved=[0], state=state)])
    result = pc.compare_request(model, left, right)
    assert result["state_evidence"] == "verified"
    assert result["same_state_claim"] is True

    right["events"][0].update(other)
    result = pc.compare_request(model, left, right)
    assert result["state_evidence"] == "verified"
    assert result["same_state_claim"] is False


def test_gemma_local_positions_are_normalized_by_block():
    model, policy = "gemma", "p"
    task = sample(0)
    left = raw(model, policy, task, True, [step("selected", [0, 1], block=48)])
    right = raw(model, policy, task, True, [step("selected", [0, 1], block=48)])
    result = pc.compare_request(model, left, right)
    assert result["positions_all_identical"] is True
    assert pc._steps(model, left)[0]["positions"] == (48, 49)

    right = raw(model, policy, task, True, [step("selected", [0, 2], block=48)])
    result = pc.compare_request(model, left, right)
    assert result["first_different_step"] == 1
    assert result["q_symmetric_difference"] == 2


def test_build_comparison_uses_scores_and_reference_dirs(tmp_path):
    model = "llada21"
    placebo_policy = "t2t_q4_f0_m3_hard_slack_placebo"
    adaptive_policy = "t2t_q4_f0_m3_hard_slack"
    noedit_policy = "t2t_q4_f0_m4_hard_noedit"
    tasks = [sample(0), sample(1)]

    placebo = [score_row(model, placebo_policy, tasks[0], True),
               score_row(model, placebo_policy, tasks[1], False)]
    adaptive = [score_row(model, adaptive_policy, tasks[0], False),
                score_row(model, adaptive_policy, tasks[1], True)]
    noedit = [score_row(model, noedit_policy, tasks[0], True, mask=4, edit=0),
              score_row(model, noedit_policy, tasks[1], True, mask=4, edit=0)]
    native = [score_row(model, "native", tasks[0], True, nominal_q=32, mask="", edit=""),
              score_row(model, "native", tasks[1], True, nominal_q=32, mask="", edit="")]

    scores_dir = write_scores(tmp_path / "scores", model, placebo).parent
    ref_dir = write_scores(tmp_path / "refs", model, adaptive + noedit + native).parent
    references, sources = pc.load_reference_scores([ref_dir], model)

    raw_records = {}
    for policy, correct in ((placebo_policy, [True, False]),
                            (adaptive_policy, [False, True]),
                            (noedit_policy, [True, True]),
                            ("native", [True, True])):
        for i, task in enumerate(tasks):
            raw_records[(policy, i)] = raw(
                model, policy, task, correct[i],
                [step("selected", [0, 1], reserved=[0])])
    comparison, alignment = pc.build_comparison(
        model, pc.read_scores(scores_dir, model), references, sources,
        raw_records, include_native=True)
    assert len(comparison) == 3
    assert {row["reference_source"] for row in comparison} == {
        "adaptive", "hard_noedit", "native"}
    adaptive_row, = [row for row in comparison if row["reference_source"] == "adaptive"]
    assert adaptive_row["wins"] == 1 and adaptive_row["losses"] == 1
    assert adaptive_row["prefix_reserved_compared_requests"] == 2
    assert adaptive_row["prefix_reserved_identical_requests"] == 2
    assert adaptive_row["prefix_placebo_reserved_seats"] == 2
    assert adaptive_row["prefix_reference_reserved_seats"] == 2
    assert adaptive_row["output_identical"] == 0
    assert adaptive_row["all_q_identical"] == 2
    assert adaptive_row["state_evidence"] == "missing_state"
    assert adaptive_row["same_state_claim"] == 0
    assert adaptive_row["causal_claim"] == 0
    assert len(alignment) == 6


def test_cli_reads_scores_dir_and_leaves_scoring_csv_unchanged(tmp_path):
    model = "gemma"
    placebo_policy = "t2t_q32_f0_m28_hard_trigger_placebo"
    adaptive_policy = "t2t_q32_f0_m28_hard_trigger"
    noedit_policy = "t2t_q32_f0_m32_hard_noedit"
    tasks = [sample(0), sample(1)]

    out = tmp_path / "scores"
    scoring = write_scores(out, model, [
        score_row(model, placebo_policy, tasks[0], True, nominal_q=32, mask=28, edit=4),
        score_row(model, placebo_policy, tasks[1], False, nominal_q=32, mask=28, edit=4),
    ])
    before = scoring.read_bytes()
    refs = tmp_path / "refs"
    write_scores(refs, model, [
        score_row(model, adaptive_policy, tasks[0], False, nominal_q=32, mask=28, edit=4),
        score_row(model, adaptive_policy, tasks[1], True, nominal_q=32, mask=28, edit=4),
        score_row(model, noedit_policy, tasks[0], True, nominal_q=32, mask=32, edit=0),
        score_row(model, noedit_policy, tasks[1], True, nominal_q=32, mask=32, edit=0),
        score_row(model, "native", tasks[0], True, nominal_q=256, mask="", edit=""),
        score_row(model, "native", tasks[1], True, nominal_q=256, mask="", edit=""),
    ])
    placebo_run = write_run(tmp_path / "placebo-run", model, {
        i: {placebo_policy: raw(model, placebo_policy, task, correct,
                               [step("selected", [0, 1], block=48, reserved=[0])])}
        for i, (task, correct) in enumerate(zip(tasks, (True, False)))
    })
    reference_run = write_run(tmp_path / "reference-run", model, {
        i: {
            adaptive_policy: raw(model, adaptive_policy, task, correct,
                                 [step("selected", [0, 1], block=48, reserved=[0])]),
            noedit_policy: raw(model, noedit_policy, task, True,
                               [step("selected", [0, 1], block=48, reserved=[0])]),
            "native": raw(model, "native", task, True,
                          [step("selected", [0, 1], block=48, reserved=[0])]),
        }
        for i, (task, correct) in enumerate(zip(tasks, (False, True)))
    })

    assert pc.main([
        "--model", model, "--scores-dir", str(out), "--out", str(out),
        "--reference-scores-dir", str(refs),
        "--run", str(placebo_run), "--reference-run", str(reference_run),
    ]) == 0
    assert scoring.read_bytes() == before
    assert (out / ("%s-placement-comparison.csv" % model)).is_file()
    assert (out / ("%s-placement-alignment.csv" % model)).is_file()
    assert not (out / ("%s-online-summary.csv" % model)).exists()

    with (out / ("%s-placement-comparison.csv" % model)).open() as stream:
        comparison = list(csv.DictReader(stream))
    adaptive_row, = [row for row in comparison if row["reference_source"] == "adaptive"]
    assert adaptive_row["wins"] == "1" and adaptive_row["losses"] == "1"
    assert adaptive_row["prefix_reserved_identical_requests"] == "2"
    assert adaptive_row["state_evidence"] == "missing_state"
    assert adaptive_row["same_state_claim"] == "0"


def test_cli_without_raw_runs_writes_header_and_missing_raw(tmp_path):
    model = "llada21"
    policy = "t2t_q4_f0_m3_hard_slack_placebo"
    task = sample(0)
    out = tmp_path / "scores"
    write_scores(out, model, [score_row(model, policy, task, True)])
    refs = tmp_path / "refs"
    write_scores(refs, model, [
        score_row(model, "t2t_q4_f0_m3_hard_slack", task, True),
        score_row(model, "t2t_q4_f0_m4_hard_noedit", task, True, mask=4, edit=0),
    ])
    assert pc.main([
        "--model", model, "--scores-dir", str(out), "--out", str(out),
        "--reference-scores-dir", str(refs), "--no-native",
    ]) == 0
    with (out / ("%s-placement-alignment.csv" % model)).open() as stream:
        assert tuple(next(csv.reader(stream))) == pc.ALIGNMENT_COLUMNS
    with (out / ("%s-placement-comparison.csv" % model)).open() as stream:
        rows = list(csv.DictReader(stream))
    assert {row["state_evidence"] for row in rows} == {"missing_raw"}
    assert all(row["same_state_claim"] == "0" for row in rows)

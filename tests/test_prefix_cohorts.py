import json
import math
from collections import Counter

import pytest

import tools.prefix_cohorts as prefix_cohorts


DATASETS = ("gsm8k", "math500", "humaneval")


def make_cases():
    cases = []
    for index in range(90):
        dataset = DATASETS[(index % 30) // 10]
        case = dict(index=index, source_index=index, dataset=dataset,
                    task_id=f"{dataset}/{index}", seed=17000 + index,
                    prompt=f"prompt {index}", reference="42")
        if dataset == "humaneval":
            case["source"] = dict(entry_point="f", prompt="def f():",
                                  test="assert True")
        cases.append(case)
    return cases


def native_correct(index):
    return index % 3 != 0 and index % 10 != 1


def arm_correct(index):
    if index % 10 == 1:
        return True
    if index % 10 == 2 and index >= 30:
        return False
    return native_correct(index)


def make_requests(cases, arm):
    rows = []
    for case in cases:
        index = case["index"]
        n = 10 + index % 4 + (2 if arm else 0)
        f = 2 if (not arm or index % 2 == 0) else 3
        q = 8 + (1 if arm else 0)
        correct = arm_correct(index) if arm else native_correct(index)
        rows.append(dict(
            status="success", model="llada21",
            policy="arm" if arm else "zero_arm",
            index=index, dataset=case["dataset"], task_id=case["task_id"],
            seed=case["seed"], max_q=32,
            output_tokens=n, generation_forwards=f,
            generation_query_rows=q, generation_physical_padded_rows=q,
            tpf=n / f, token_efficiency=n / q, truncated=False,
            output_text="\\boxed{42}" if correct else "\\boxed{41}",
        ))
    return rows


def fake_score(text, task):
    return {"correct": "42" in text}


def write_requests(path, rows, as_dir):
    if as_dir:
        path.mkdir()
        for row in rows:
            name = f"{row['index']:02d}-{row['policy']}.json"
            (path / name).write_text(json.dumps(row))
    else:
        path.write_text(json.dumps({"requests": rows}))


def run_study(tmp_path, monkeypatch, *, cases=None, baseline_rows=None,
              arm_rows=None, baseline_dir=False, arm_dir=False, **kwargs):
    cases = cases or make_cases()
    baseline_rows = baseline_rows if baseline_rows is not None else make_requests(cases, False)
    arm_rows = arm_rows if arm_rows is not None else make_requests(cases, True)
    monkeypatch.setattr(prefix_cohorts, "load_cases", lambda path: cases)
    monkeypatch.setattr(prefix_cohorts, "score", fake_score)
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps({"samples": cases}))
    baseline_path = tmp_path / ("baseline_dir" if baseline_dir else "baseline.json")
    arm_path = tmp_path / ("arm_dir" if arm_dir else "arm.json")
    write_requests(baseline_path, baseline_rows, baseline_dir)
    write_requests(arm_path, arm_rows, arm_dir)
    out = tmp_path / "summary.csv"
    summary, paired, audit = prefix_cohorts.score_files(
        baseline_path, arm_path, cases_path, out, **kwargs)
    return summary, paired, audit, cases


def test_windows_pairing_quality_work_and_outputs(tmp_path, monkeypatch):
    summary, paired, audit, cases = run_study(tmp_path, monkeypatch)
    rows = {(row["window"], row["scope"]): row for row in summary}
    assert len(summary) == 12
    expected = {30: (18, 21, 3, 0), 60: (36, 38, 6, 4), 90: (54, 59, 9, 4)}
    for window, (baseline, arm, rescued, lost) in expected.items():
        row = rows[(window, "all")]
        assert row["requests"] == window
        assert (row["baseline_correct"], row["arm_correct"],
                row["rescued"], row["lost"]) == (baseline, arm, rescued, lost)
        assert row["quality_delta"] == pytest.approx((arm - baseline) / window)
        assert (row["both_correct"] + row["both_wrong"] +
                row["rescued"] + row["lost"]) == window
    assert rows[(30, "gsm8k")]["requests"] == 10
    assert rows[(30, "gsm8k")]["rescued"] == 1
    baseline_all = make_requests(cases, False)
    arm_all = make_requests(cases, True)
    window_indices = {30: set(range(0, 30)), 60: set(range(30, 90)),
                      90: set(range(0, 90))}
    for window in (30, 60, 90):
        baseline = [row for row in baseline_all if row["index"] in window_indices[window]]
        arm = [row for row in arm_all if row["index"] in window_indices[window]]
        row = rows[(window, "all")]
        assert row["baseline_macro_tpf"] == pytest.approx(
            sum(item["tpf"] for item in baseline) / window)
        assert row["arm_macro_tpf"] == pytest.approx(
            sum(item["tpf"] for item in arm) / window)
        assert row["baseline_pooled_token_efficiency"] == pytest.approx(
            sum(item["output_tokens"] for item in baseline) /
            sum(item["generation_query_rows"] for item in baseline))
        assert row["arm_pooled_token_efficiency"] == pytest.approx(
            sum(item["output_tokens"] for item in arm) /
            sum(item["generation_query_rows"] for item in arm))
        assert row["baseline_q_per_f"] == pytest.approx(
            sum(item["generation_query_rows"] for item in baseline) /
            sum(item["generation_forwards"] for item in baseline))
        assert row["arm_q_per_f"] == pytest.approx(
            sum(item["generation_query_rows"] for item in arm) /
            sum(item["generation_forwards"] for item in arm))
        assert row["baseline_N"] == sum(item["output_tokens"] for item in baseline)
        assert row["arm_F"] == sum(item["generation_forwards"] for item in arm)
        assert row["delta_Q"] == (sum(item["generation_query_rows"] for item in arm) -
                                  sum(item["generation_query_rows"] for item in baseline))
    assert len(paired) == 180
    assert Counter(row["window"] for row in paired) == {30: 30, 60: 60, 90: 90}
    assert audit["cases"] == 90
    assert audit["baseline_policy"] == "zero_arm"
    assert audit["requests"] == {"baseline": 90, "arm": 90}
    assert audit["cohorts"] == {"30": "original-0-29", "60": "added-30-89",
                                "90": "all-0-89"}
    assert audit["window_indices"]["30"] == list(range(30))
    assert audit["window_indices"]["60"] == list(range(30, 90))
    assert audit["window_indices"]["90"] == list(range(90))
    assert audit["dataset_counts_per_window"]["30"] == {
        "gsm8k": 10, "math500": 10, "humaneval": 10}
    assert audit["dataset_counts_per_window"]["60"] == {
        "gsm8k": 20, "math500": 20, "humaneval": 20}
    assert audit["dataset_counts_per_window"]["90"] == {
        "gsm8k": 30, "math500": 30, "humaneval": 30}
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "summary-paired.csv").exists()
    assert (tmp_path / "summary-audit.json").exists()


def test_paired_ci_formula_and_quality_clipping():
    estimate, se, low, high = prefix_cohorts.paired_ci(
        {"a": [1.0, 2.0, 3.0], "b": [-1.0, 0.0, 1.0]})
    assert estimate == pytest.approx(1.0)
    assert se == pytest.approx(math.sqrt(1.0 / 6.0))
    assert low == pytest.approx(1.0 - prefix_cohorts.Z95 * math.sqrt(1.0 / 6.0))
    assert high == pytest.approx(1.0 + prefix_cohorts.Z95 * math.sqrt(1.0 / 6.0))
    estimate, se, low, high = prefix_cohorts.paired_ci(
        {"a": [1.0, -1.0]}, bounds=(-1.0, 1.0))
    assert estimate == 0.0
    assert low == -1.0 and high == 1.0


def test_balanced_window_selection_with_grouped_added_cases():
    cases = []
    index = 0
    for dataset in DATASETS:
        for offset in range(10):
            cases.append(dict(index=index, dataset=dataset,
                              task_id=f"{dataset}/original-{offset}", seed=index))
            index += 1
    for dataset in DATASETS:
        for offset in range(20):
            cases.append(dict(index=index, dataset=dataset,
                              task_id=f"{dataset}/added-{offset}", seed=index))
            index += 1
    selected30 = prefix_cohorts.select_window_cases(cases, 30)
    assert [case["index"] for case in selected30] == list(range(30))
    selected60 = prefix_cohorts.select_window_cases(cases, 60)
    assert [case["index"] for case in selected60] == list(range(30, 90))
    assert Counter(case["dataset"] for case in selected60) == {
        "gsm8k": 20, "math500": 20, "humaneval": 20}
    selected90 = prefix_cohorts.select_window_cases(cases, 90)
    assert [case["index"] for case in selected90] == list(range(90))
    original = {case["index"] for case in selected30}
    added = {case["index"] for case in selected60}
    assert original.isdisjoint(added)
    assert original | added == set(range(90))


def test_quality_ci_is_nonzero_when_all_cases_agree(tmp_path, monkeypatch):
    cases = make_cases()
    baseline = make_requests(cases, False)
    arm = make_requests(cases, True)
    for row in baseline + arm:
        row["output_text"] = "\\boxed{42}"
    summary, _, _, _ = run_study(tmp_path, monkeypatch, cases=cases,
                                 baseline_rows=baseline, arm_rows=arm)
    row = {(item["window"], item["scope"]): item for item in summary}[(30, "all")]
    assert row["quality_delta"] == 0
    assert row["quality_delta_ci_low"] < 0 < row["quality_delta_ci_high"]
    assert row["quality_delta_se"] > 0
    assert "Wilson" in row["quality_ci_method"]
    assert row["quality_ci_fallback_strata"] == 3

    baseline = make_requests(cases, False)
    arm = make_requests(cases, True)
    for row in baseline + arm:
        row["output_text"] = "\\boxed{41}"
    summary, _, _, _ = run_study(tmp_path, monkeypatch, cases=cases,
                                 baseline_rows=baseline, arm_rows=arm)
    row = {(item["window"], item["scope"]): item for item in summary}[(30, "all")]
    assert row["quality_delta"] == 0
    assert row["quality_delta_ci_low"] < 0 < row["quality_delta_ci_high"]
    assert row["quality_delta_se"] > 0


def test_quality_ci_newcombe_and_zero_discordance_fallback():
    estimate, se, low, high, fallback = prefix_cohorts.quality_ci(
        {"d": (30, 0, 0, 0)})
    assert estimate == 0
    assert fallback == 1
    assert low < 0 < high
    assert se > 0
    estimate, se, low, high, fallback = prefix_cohorts.quality_ci(
        {"d": (27, 3, 0, 0)})
    assert estimate == pytest.approx(-0.1)
    assert fallback == 0
    assert low < estimate < high
    assert high - low > 0


def test_window_selection_rejects_unbalanced_or_noncanonical_cohorts():
    cases = make_cases()
    cases[0]["dataset"] = "math500"
    with pytest.raises(ValueError, match="expected 10 per dataset"):
        prefix_cohorts.select_window_cases(cases, 30)
    cases = make_cases()
    cases[30]["dataset"] = "humaneval"
    with pytest.raises(ValueError, match="expected 20 per dataset"):
        prefix_cohorts.select_window_cases(cases, 60)
    cases = make_cases()
    cases[0]["index"] = 99
    with pytest.raises(ValueError, match="indices must be exactly 0..89"):
        prefix_cohorts.select_window_cases(cases, 30)


def test_arbitrary_zero_baseline_and_same_nominal_q(tmp_path, monkeypatch):
    cases = make_cases()
    baseline = make_requests(cases, False)
    arm = make_requests(cases, True)
    for row in baseline:
        row["policy"] = "t2t_q8_f0_m0_zero"
    for row in arm:
        row["policy"] = "t2t_q8_f0_m0_prefix"
    summary, _, audit, _ = run_study(tmp_path, monkeypatch, cases=cases,
                                     baseline_rows=baseline, arm_rows=arm)
    assert audit["baseline_policy"] == "t2t_q8_f0_m0_zero"
    assert audit["nominal_q"] == {"baseline": 8, "arm": 8, "resolved": 8,
                                  "matched": True}
    assert summary[0]["nominal_q"] == 8

    arm_mismatch = [dict(row, policy="t2t_q16_f0_m0_prefix") for row in arm]
    with pytest.raises(ValueError, match="nominal Q"):
        run_study(tmp_path, monkeypatch, cases=cases, baseline_rows=baseline,
                  arm_rows=arm_mismatch)
    with pytest.raises(ValueError, match="nominal Q"):
        run_study(tmp_path, monkeypatch, cases=cases, baseline_rows=baseline,
                  arm_rows=arm, nominal_q=16)


def test_exact_diffusiongemma_alias_and_unknown_models_rejected(tmp_path, monkeypatch):
    cases = make_cases()
    baseline = make_requests(cases, False)
    arm = make_requests(cases, True)
    for row in baseline + arm:
        row["model"] = "diffusiongemma"
        row["max_q"] = 256
    summary, _, audit, _ = run_study(tmp_path, monkeypatch, cases=cases,
                                     baseline_rows=baseline, arm_rows=arm)
    assert summary[0]["model"] == "gemma"
    assert audit["model"] == "gemma"
    assert prefix_cohorts.canonical_model("diffusiongemma") == "gemma"

    bad = [dict(row, model="diffusion-gemma") for row in baseline]
    with pytest.raises(ValueError, match="unsupported model 'diffusion-gemma'"):
        run_study(tmp_path, monkeypatch, cases=cases,
                  baseline_rows=bad, arm_rows=arm)
    bad = [dict(row, model="DiffusionGemma") for row in baseline]
    with pytest.raises(ValueError, match="unsupported model 'DiffusionGemma'"):
        run_study(tmp_path, monkeypatch, cases=cases,
                  baseline_rows=bad, arm_rows=arm)


def test_reject_missing_case(tmp_path, monkeypatch):
    cases = make_cases()
    with pytest.raises(ValueError, match="missing case indices"):
        run_study(tmp_path, monkeypatch, cases=cases,
                  baseline_rows=make_requests(cases, False)[:-1])


def test_reject_duplicate_case(tmp_path, monkeypatch):
    cases = make_cases()
    rows = make_requests(cases, False)
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate case index"):
        run_study(tmp_path, monkeypatch, cases=cases, baseline_rows=rows)


def test_reject_failed_case(tmp_path, monkeypatch):
    cases = make_cases()
    rows = make_requests(cases, False)
    rows[5]["status"] = "error"
    with pytest.raises(ValueError, match="failed status"):
        run_study(tmp_path, monkeypatch, cases=cases, baseline_rows=rows)


def test_reject_metadata_mismatch(tmp_path, monkeypatch):
    cases = make_cases()
    rows = make_requests(cases, False)
    rows[7]["seed"] += 1
    with pytest.raises(ValueError, match="seed mismatch"):
        run_study(tmp_path, monkeypatch, cases=cases, baseline_rows=rows)


def test_reject_work_metadata_mismatch(tmp_path, monkeypatch):
    cases = make_cases()
    rows = make_requests(cases, False)
    rows[3]["generation_physical_padded_rows"] += 1
    with pytest.raises(ValueError, match="Unexpected unpadded work"):
        run_study(tmp_path, monkeypatch, cases=cases, baseline_rows=rows)


def test_reject_provided_correct_mismatch(tmp_path, monkeypatch):
    cases = make_cases()
    rows = make_requests(cases, False)
    rows[0]["correct"] = 1
    with pytest.raises(ValueError, match="provided correct disagrees"):
        run_study(tmp_path, monkeypatch, cases=cases, baseline_rows=rows)


def test_directory_input_policy_filter_and_metadata_files(tmp_path, monkeypatch):
    cases = make_cases()
    arm_rows = make_requests(cases, True)
    decoy = dict(arm_rows[0], policy="other")
    arm_rows = arm_rows + [decoy]
    summary, _, audit, _ = run_study(
        tmp_path, monkeypatch, cases=cases, baseline_dir=True, arm_dir=True,
        arm_rows=arm_rows, baseline_policy="zero_arm", arm_policy="arm")
    assert len(summary) == 12
    assert audit["arm_policy"] == "arm"


def test_multiple_policies_require_explicit_selection(tmp_path, monkeypatch):
    cases = make_cases()
    arm_rows = make_requests(cases, True)
    arm_rows.append(dict(arm_rows[0], policy="other"))
    with pytest.raises(ValueError, match="Multiple policies"):
        run_study(tmp_path, monkeypatch, cases=cases, arm_dir=True,
                  arm_rows=arm_rows, arm_policy=None)


def test_read_request_rows_ignores_run_metadata(tmp_path):
    cases = make_cases()
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "run.json").write_text(json.dumps({"model": "llada21"}))
    (folder / "complete-arm.json").write_text(json.dumps({"policy": "arm", "requests": 90}))
    (folder / "marks-arm.json").write_text(json.dumps({"0": True}))
    request = make_requests(cases, True)[0]
    (folder / "00-arm.json").write_text(json.dumps(request))
    rows = prefix_cohorts.read_request_rows(folder, "arm")
    assert rows == [request]

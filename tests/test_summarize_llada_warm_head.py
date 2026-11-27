"""Unit contracts for the read-only warm-head paired summary."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest


MODULE = Path(__file__).resolve().parents[1] / "tools" / "summarize_llada_warm_head.py"
SPEC = importlib.util.spec_from_file_location("summarize_llada_warm_head", MODULE)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def traces(*, rows=(40, 41, 64, 65), added=(40, 41), old_sampled=(64, 65),
           pre_state="same", new_rows=None):
    old = [dict(tick=0, rows=list(rows), dirty_before=[], sampled=list(old_sampled),
                prefix=[], dirty=[], pre_state={"x": pre_state},
                layer_physical_q=[len(rows)] * 20, q=len(rows))]
    new = [dict(tick=0, rows=list(new_rows or rows), dirty_before=[],
                sampled=sorted(set(old_sampled) | set(added)),
                warm_head_sample=len(added), prefix=[], dirty=[],
                pre_state={"x": pre_state},
                layer_physical_q=[len(new_rows or rows)] * 20,
                q=len(new_rows or rows))]
    return old, new


def make_row(index, *, flip=False, intervened=True, fail=False, **kw):
    old_correct = index % 3 != 0
    new_correct = (index % 2 == 0) if flip else old_correct
    old_trace, new_trace = traces(**kw)
    row = dict(
        index=index,
        old=dict(correct=old_correct, N=10 + index, F=2 + index % 5,
                 Q=20 + index, trace=old_trace),
        new=dict(correct=new_correct, N=12 + index, F=3 + index % 4,
                 Q=22 + index, trace=new_trace),
        compare=dict(intervened=intervened, passed=True, pre_state_verified=True,
                     tick=0 if intervened else None,
                     added=[40, 41] if intervened else None))
    if fail:
        row["old_failure"] = "boom"
    return row


MANIFEST = dict(protocol="llada-hfull-warm-head-single-factor-v1", q=64,
                mode="half", coverage="head", checkpoint="/models/x",
                samples_sha256="samples", old_source="run")


def write_run(root, rows, manifest=None):
    (root / "rows").mkdir(parents=True)
    for row in rows:
        (root / "rows" / f"{row['index']:02d}.json").write_text(json.dumps(row))
    (root / "run.json").write_text(json.dumps(manifest or MANIFEST))
    return root


def full_rows():
    pilot = [make_row(i) for i in (0, 10, 20)]
    pinned = [make_row(i) for i in range(90) if i not in (0, 10, 20)]
    return pilot, pinned


def test_combine_and_cohorts_are_complete():
    pilot, pinned = full_rows()
    pilot_rows = {r["index"]: r for r in pilot}
    pinned_rows = {r["index"]: r for r in pinned}
    rows = mod.combine(pilot_rows, MANIFEST, pinned_rows, MANIFEST)
    assert len(rows) == 90
    cohorts = {name: mod.summarize_cohort(name, indices, rows)
               for name, indices in mod.COHORTS.items()}
    assert cohorts["original30"]["coverage"] == dict(
        expected=30, present=30, missing=[], complete=True, fraction=1.0)
    assert cohorts["new60"]["coverage"]["present"] == 60
    assert cohorts["all90"]["coverage"]["complete"] is True
    assert cohorts["all90"]["intervened"] == 90
    assert cohorts["all90"]["declared_first_passed"] == 90
    assert cohorts["all90"]["independent_first_passed"] == 90
    assert cohorts["all90"]["independent_pre_state_verified"] == 90
    assert cohorts["all90"]["old_pooled_TPF"] == pytest.approx(
        cohorts["all90"]["old_N"] / cohorts["all90"]["old_F"])
    assert cohorts["all90"]["new_pooled_TE"] == pytest.approx(
        cohorts["all90"]["new_N"] / cohorts["all90"]["new_Q"])
    old_tpf = [row["old_TPF"] for row in cohorts["all90"]["per_index"]]
    new_te = [row["new_TE"] for row in cohorts["all90"]["per_index"]]
    assert cohorts["all90"]["old_macro_TPF"] == pytest.approx(sum(old_tpf) / len(old_tpf))
    assert cohorts["all90"]["new_macro_TE"] == pytest.approx(sum(new_te) / len(new_te))
    assert cohorts["all90"]["old_executor_F"] == cohorts["all90"]["old_F"]
    assert cohorts["all90"]["old_strict_cap"] is True
    assert cohorts["all90"]["new_strict_cap"] is True
    assert cohorts["all90"]["old_max_Q_per_forward"] <= 64
    first = cohorts["all90"]["per_index"][0]
    assert first["old_TPF"] == pytest.approx(first["old_N"] / first["old_F"])
    assert first["new_TE"] == pytest.approx(first["new_N"] / first["new_Q"])
    assert first["old_executor_F"] == first["old_F"]
    assert first["old_strict_cap"] is True and first["new_strict_cap"] is True


def test_strict_cap_is_false_when_a_forward_exceeds_the_offer():
    row = make_row(4)
    row["new"]["trace"][0]["q"] = 65
    row["new"]["trace"][0]["layer_physical_q"] = [65] * 20
    metrics = mod.arm_metrics(row["new"], q=64)
    assert metrics["strict_cap"] is False
    assert metrics["max_Q_per_forward"] == 65


def test_incomplete_cohort_is_partial_and_present_only():
    pilot, pinned = full_rows()
    pinned = [r for r in pinned if r["index"] != 89]
    rows = mod.combine({r["index"]: r for r in pilot}, MANIFEST,
                       {r["index"]: r for r in pinned}, MANIFEST)
    new60 = mod.summarize_cohort("new60", mod.COHORTS["new60"], rows)
    assert new60["coverage"]["present"] == 59
    assert new60["coverage"]["missing"] == [89]
    assert new60["coverage"]["complete"] is False
    assert new60["partial"] is True and new60["metrics_scope"] == "present_only"
    all90 = mod.summarize_cohort("all90", mod.COHORTS["all90"], rows)
    assert all90["coverage"]["missing"] == [89]
    assert all90["partial"] is True


def test_independent_verification_detects_rows_and_pre_state_mismatch():
    bad_rows = make_row(1, new_rows=(39, 40, 64, 65))
    pair = mod.verify_pair(bad_rows)
    assert pair["declared_passed"] is True
    assert pair["independent_passed"] is False
    assert pair["independent_checks"]["rows_equal"] is False

    bad_pre = make_row(2)
    bad_pre["old"]["trace"][0]["pre_state"] = {"x": "different"}
    pair = mod.verify_pair(bad_pre)
    assert pair["independent_passed"] is False
    assert pair["independent_checks"]["pre_state_equal"] is False


def test_non_intervened_pair_is_not_verified():
    row = make_row(3, intervened=False)
    pair = mod.verify_pair(row)
    assert pair["intervened"] is False
    assert pair["independent_passed"] is None
    assert pair["independent_pre_state_verified"] is None


def test_excluded_failed_run_is_metadata_only(tmp_path):
    pilot, pinned = full_rows()
    pilot_dir = write_run(tmp_path / "pilot", pilot)
    pinned_dir = write_run(tmp_path / "pinned", pinned)
    failed = tmp_path / "failed"
    (failed / "rows").mkdir(parents=True)
    (failed / "rows" / "10.json").write_text("{not json")
    (failed / "summary.json").write_text(json.dumps(
        dict(requests=3, old_failures=1, new_failures=1,
             intervened=2, first_checks_passed=2)))
    out = tmp_path / "out"
    assert mod.main(["--pilot", str(pilot_dir), "--pinned", str(pinned_dir),
                     "--out-dir", str(out), "--exclude", str(failed)]) == 0
    result = json.loads((out / "summary-warm-head-single-factor.json").read_text())
    assert result["source"]["excluded_rows"] == 1
    assert result["source"]["excluded_summary"]["old_failures"] == 1
    assert result["source"]["combined_indices"] == list(range(90))
    assert result["cohorts"]["all90"]["coverage"]["complete"] is True


def test_output_has_no_raw_fields(tmp_path):
    pilot, pinned = full_rows()
    pilot_dir = write_run(tmp_path / "pilot", pilot)
    pinned_dir = write_run(tmp_path / "pinned", pinned)
    out = tmp_path / "out"
    mod.main(["--pilot", str(pilot_dir), "--pinned", str(pinned_dir),
              "--out-dir", str(out)])
    result = json.loads((out / "summary-warm-head-single-factor.json").read_text())
    text = json.dumps(result)
    for key in mod.RAW_KEYS:
        assert f'"{key}"' not in text
    mod._assert_no_raw(result)
    assert (out / "summary-warm-head-single-factor.csv").exists()
    assert (out / "paired-warm-head-single-factor.csv").exists()


def test_cli_csv_rows(tmp_path):
    pilot, pinned = full_rows()
    pilot_dir = write_run(tmp_path / "pilot", pilot)
    pinned_dir = write_run(tmp_path / "pinned", pinned)
    out = tmp_path / "out"
    mod.main(["--pilot", str(pilot_dir), "--pinned", str(pinned_dir),
              "--out-dir", str(out)])
    with (out / "summary-warm-head-single-factor.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["cohort"] for r in rows] == ["original30", "new60", "all90"]
    assert rows[-1]["present"] == "90" and rows[-1]["complete"] == "True"
    with (out / "paired-warm-head-single-factor.csv").open(newline="") as handle:
        pairs = list(csv.DictReader(handle))
    assert len(pairs) == 90


def test_protocol_mismatch_and_duplicate_raise(tmp_path):
    pilot, pinned = full_rows()
    with pytest.raises(mod.SummaryError, match="protocol"):
        mod.combine({r["index"]: r for r in pilot}, MANIFEST,
                    {r["index"]: r for r in pinned}, dict(MANIFEST, protocol="other"))
    dup = make_row(10)
    with pytest.raises(mod.SummaryError, match="both pilot and pinned"):
        mod.combine({10: dup}, MANIFEST, {10: dup}, MANIFEST)

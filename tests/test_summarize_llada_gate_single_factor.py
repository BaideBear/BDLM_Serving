"""Unit contracts for the gate single-factor read-only summary."""
import csv
import json
from pathlib import Path

import pytest

from experiments.t2t.serving_slots.llada_hfull_gate_single_factor import compare_pair
from tools import summarize_llada_gate_single_factor as mod


def trace(rows=range(32, 64), q=None):
    rows = list(rows)
    q = len(rows) if q is None else q
    return [dict(tick=0, rows=rows, q=q, layer_physical_q=[q] * 20,
                 tail_opened=False, head_mask_at_plan=8,
                 pre_state={"x": "same"})]


def arm(correct, n, f, q, rows=range(32, 64)):
    return dict(correct=correct, N=n, F=f, Q=q, trace=trace(rows),
                output_ids=[1, 2, 3], failure=None)


def pair(index, old_correct, new_correct, *, old_n=100, new_n=110,
         old_f=10, new_f=9, old_q=500, new_q=450):
    old = arm(old_correct, old_n, old_f, old_q)
    new = arm(new_correct, new_n, new_f, new_q)
    return dict(index=index, old=old, new=new, compare=compare_pair(old, new))


def write(root, rows, prefix=None):
    (root / "rows").mkdir(parents=True, exist_ok=True)
    for row in rows:
        name = f"{prefix}-{row['index']:02d}.json" if prefix else f"{row['index']:02d}.json"
        (root / "rows" / name).write_text(json.dumps(row))
    return root


def make_tree(tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    (control / "rows").mkdir()
    pilot = tmp_path / "pilot"
    rest = tmp_path / "rest"
    for i in range(90):
        old = arm(i % 3 != 0, 100 + i, 10 + i % 4, 500 + i)
        new = arm(i % 2 == 0, 110 + i, 9 + i % 3, 450 + i)
        if i < 60:
            (control / "rows" / f"c64-full-half-{i:02d}.json").write_text(json.dumps(old))
            (control / "rows" / f"c64-full-quarter-{i:02d}.json").write_text(json.dumps(new))
        else:
            write(rest, [dict(index=i, old=old, new=new,
                              compare=compare_pair(old, new))])
        if i in (0, 10, 20):
            write(pilot, [dict(index=i, old=old, new=new,
                               compare=compare_pair(old, new))])
    return control, pilot, rest


def test_build_rows_and_cohorts_complete(tmp_path):
    control, pilot, rest = make_tree(tmp_path)
    rows = mod.build_rows(control / "rows", rest, pilot)
    assert sorted(rows) == list(range(90))
    cohorts = {name: mod.summarize_cohort(name, indices, rows)
               for name, indices in mod.COHORTS.items()}
    assert cohorts["original30"]["coverage"]["complete"] is True
    assert cohorts["new60"]["coverage"]["complete"] is True
    assert cohorts["all90"]["coverage"]["complete"] is True
    assert cohorts["all90"]["old_strict_cap"] is True
    assert cohorts["all90"]["new_strict_cap"] is True
    old_tpf = [row["old_TPF"] for row in cohorts["all90"]["per_index"]]
    assert cohorts["all90"]["old_macro_TPF"] == pytest.approx(sum(old_tpf) / len(old_tpf))
    assert cohorts["all90"]["old_executor_F"] == cohorts["all90"]["old_F"]
    checks = mod.pilot_check(pilot, control / "rows")
    assert len(checks) == 6
    assert all(all(value for key, value in check.items() if key not in ("index", "arm", "gate"))
               for check in checks)


def test_missing_control_row_is_partial(tmp_path):
    control, pilot, rest = make_tree(tmp_path)
    (control / "rows" / "c64-full-half-59.json").unlink()
    rows = mod.build_rows(control / "rows", rest, pilot)
    stats = mod.summarize_cohort("all90", mod.COHORTS["all90"], rows)
    assert stats["coverage"]["missing"] == [59]
    assert stats["coverage"]["complete"] is False
    assert stats["partial"] is True and stats["metrics_scope"] == "present_only"


def test_pilot_mismatch_is_reported(tmp_path):
    control, pilot, rest = make_tree(tmp_path)
    row = json.loads((pilot / "rows" / "00.json").read_text())
    row["old"]["output_ids"] = [9, 9]
    (pilot / "rows" / "00.json").write_text(json.dumps(row))
    checks = mod.pilot_check(pilot, control / "rows")
    bad = [c for c in checks if c["index"] == 0 and c["arm"] == "old"][0]
    assert bad["output_ids_equal"] is False


def test_cli_writes_aggregates_without_raw_fields(tmp_path):
    control, pilot, rest = make_tree(tmp_path)
    out = tmp_path / "out"
    assert mod.main(["--control-rows", str(control / "rows"), "--gate-rest", str(rest),
                     "--gate-pilot", str(pilot), "--out-dir", str(out)]) == 0
    result = json.loads((out / "summary-gate-single-factor.json").read_text())
    text = json.dumps(result)
    for key in mod.RAW_KEYS:
        assert f'"{key}"' not in text
    mod._assert_no_raw(result)
    assert result["cohorts"]["all90"]["coverage"]["complete"] is True
    assert len(result["source"]["pilot_check"]) == 6
    with (out / "summary-gate-single-factor.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["cohort"] for r in rows] == ["original30", "new60", "all90"]

"""Unit contracts for the read-only full-head sampled audit (no model, no GPU)."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest


MODULE = Path(__file__).resolve().parents[1] / "tools" / "audit_hfull_sampled.py"
SPEC = importlib.util.spec_from_file_location("audit_hfull_sampled", MODULE)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def event(*, block=1, rows=(), mask=(), sampled=(), cold=(), plan_masks=None,
          head_q=None, tail_q=None, drain_only=False, phase="sample", tick=0,
          q=None, head_full=True):
    split = (block + 1) * 32
    rows = list(rows)
    if q is None:
        q = len(rows)
    if head_q is None:
        head_q = sum(1 for p in rows if p < split)
    if tail_q is None:
        tail_q = sum(1 for p in rows if p >= split)
    if plan_masks is None:
        plan_masks = (sum(1 for p in mask if p < split)
                      + sum(1 for p in cold if p < split))
    return dict(block=block, tick=tick, q=q, phase=phase, rows=rows, mask=list(mask),
                sampled=list(sampled), cold=list(cold), head_mask_at_plan=plan_masks,
                head_q=head_q, tail_q=tail_q, drain_only=drain_only, head_full=head_full)


def row(arm, index, trace, dataset="gsm8k", gate=None, coverage="head"):
    return dict(arm=arm, index=index, dataset=dataset, task_id=f"{dataset}/{index}",
                seed=index, cross_open=gate or arm.rsplit("-", 1)[-1],
                coverage=coverage, samples_sha256="samples-test", trace=trace)


def write_rows(root, rows):
    root.mkdir(parents=True, exist_ok=True)
    for r in rows:
        (root / f"{r['arm']}-{r['index']:02d}.json").write_text(json.dumps(r))
    return root


def affected_event(*, tick=0, warm=2, head_full=True):
    """A full-head forward whose only cold rows are in the tail block."""
    return event(rows=range(32, 96), mask=list(range(40, 40 + warm)),
                 sampled=[64, 65], cold=[64, 65], tick=tick, head_full=head_full)


def test_tail_only_cold_is_affected_and_pays_warm_head_masks():
    metrics = mod.event_metrics(affected_event(warm=3))
    assert metrics["cold_tail_only"] and metrics["affected"]
    assert metrics["head_sampled"] == 0 and metrics["head_sample0_with_head_mask"]
    assert metrics["paid_warm_head_mask_rows"] == 3


def test_a_cold_head_mask_is_sampled_and_not_affected():
    metrics = mod.event_metrics(event(rows=range(32, 96), mask=[], sampled=[40, 64],
                                      cold=[40, 64]))
    assert not metrics["cold_tail_only"] and not metrics["affected"]
    assert metrics["head_sampled"] == 1
    assert metrics["cold_head"] == 1 and metrics["cold_tail"] == 1


def test_a_full_debt_drain_is_not_the_cold_precedence_case():
    metrics = mod.event_metrics(event(rows=range(32, 65), mask=[40], sampled=[],
                                      cold=[64], drain_only=True))
    assert metrics["cold_tail_only"] and metrics["head_sample0_with_head_mask"]
    assert not metrics["affected"] and metrics["paid_warm_head_mask_rows"] == 0


def test_head_full_is_required_for_the_paid_warm_head_label():
    metrics = mod.event_metrics(affected_event(head_full=False))
    assert not metrics["affected"] and metrics["paid_warm_head_mask_rows"] == 0


def test_first_tick_and_per_question_aggregates(tmp_path):
    rows = [
        row("c64-full-eager", 0, [affected_event(tick=2, warm=2),
                                  affected_event(tick=5, warm=3)]),
        row("c64-full-eager", 1, [affected_event(tick=1, warm=1)]),
    ]
    root = write_rows(tmp_path / "rows", rows)
    result = mod.audit(str(root), arms=("c64-full-eager",))
    gate = result["gates"]["c64-full-eager"]
    assert gate["requests"] == 2 and gate["total_F"] == 3
    assert gate["affected_F"] == 3 and gate["affected_questions"] == 2
    assert gate["paid_warm_head_mask_rows"] == 6
    assert gate["first_tick"]["min"] == 1
    assert gate["first_tick"]["median"] == 1.5
    assert gate["first_tick"]["max"] == 2
    assert gate["first_tick"]["per_question"] == {"0": 2, "1": 1}
    assert gate["affected_F_per_question"] == {"0": 2, "1": 1}
    assert gate["head_sample0_F"] == 3
    assert gate["head_sample0_with_head_mask_F"] == 3
    assert gate["cold_tail_only_F"] == 3
    assert gate["by_dataset"]["gsm8k"]["affected_F"] == 3


def test_common_batch_excludes_unpaired_questions(tmp_path):
    rows = [
        row("a", 0, [affected_event(tick=0)]),
        row("a", 1, [affected_event(tick=1)]),
        row("a", 2, [affected_event(tick=2)]),
        row("b", 1, [affected_event(tick=3)]),
        row("b", 2, [affected_event(tick=4)]),
        row("b", 3, [affected_event(tick=5)]),
    ]
    root = write_rows(tmp_path / "rows", rows)
    result = mod.audit(str(root), arms=("a", "b"))
    assert result["source"]["common_indices"] == [1, 2]
    assert result["gates"]["a"]["requests"] == 2
    assert result["gates"]["a"]["total_F"] == 2
    assert result["gates"]["b"]["first_tick"]["max"] == 4
    restricted = mod.audit(str(root), arms=("a", "b"), indices=[1])
    assert restricted["gates"]["a"]["requests"] == 1
    with pytest.raises(mod.AuditError):
        mod.audit(str(root), arms=("a", "b"), indices=[0])


def test_snapshot_metadata_distinguishes_pinned_and_all_common(tmp_path):
    rows = [row("a", 0, [affected_event()]), row("a", 1, [affected_event()]),
            row("b", 0, [affected_event()]), row("b", 1, [affected_event()])]
    root = write_rows(tmp_path / "rows", rows)
    result = mod.audit(str(root), arms=("a", "b"), indices=[0])
    assert result["source"]["all_common_indices_at_capture"] == [0, 1]
    assert result["source"]["requested_indices"] == [0]
    assert result["source"]["common_indices"] == [0]
    assert result["gates"]["a"]["requests"] == 1


def test_aggregate_output_contains_no_raw_fields(tmp_path):
    rows = [row("c64-full-eager", 0, [affected_event(tick=0)])]
    root = write_rows(tmp_path / "rows", rows)
    result = mod.audit(str(root), arms=("c64-full-eager",))
    text = json.dumps(result)
    for key in mod.FORBIDDEN_RAW_KEYS:
        assert f'"{key}"' not in text
    mod._assert_no_raw(result)  # does not raise


def test_load_rows_skips_lock_and_rejects_duplicate(tmp_path):
    root = tmp_path / "rows"
    root.mkdir()
    (root / "a-00.json").write_text(json.dumps(row("a", 0, [])))
    (root / "a-00.lock").write_text("not json")
    (root / "a-00-copy.json").write_text(json.dumps(row("a", 0, [])))
    with pytest.raises(mod.AuditError, match="duplicate"):
        mod.load_rows(str(root))


@pytest.mark.parametrize("mutate,match", [
    (lambda e: e.update(head_q=99), "head_q"),
    (lambda e: e.update(q=99), "q !="),
    (lambda e: e.update(head_mask_at_plan=9), "head_mask_at_plan"),
    (lambda e: e.update(cold=[999]), "escapes rows"),
])
def test_structural_inconsistencies_fail_loudly(mutate, match):
    bad = affected_event()
    mutate(bad)
    with pytest.raises(mod.AuditError, match=match):
        mod.event_metrics(bad, "bad/trace[0]")


def test_cli_writes_json_and_csv(tmp_path):
    root = write_rows(tmp_path / "rows",
                      [row("c64-full-eager", 0, [affected_event(tick=4)])])
    out = tmp_path / "out" / "audit.json"
    table = tmp_path / "out" / "audit.csv"
    assert mod.main(["--rows", str(root), "--out", str(out), "--csv", str(table),
                     "--arms", "c64-full-eager", "--quiet"]) == 0
    result = json.loads(out.read_text())
    assert result["protocol"] == mod.PROTOCOL
    assert result["gates"]["c64-full-eager"]["affected_F"] == 1
    with table.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 and rows[0]["gate"] == "eager"
    assert rows[0]["first_tick_min"] == "4"

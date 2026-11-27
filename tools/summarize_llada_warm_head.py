#!/usr/bin/env python3
"""Read-only paired summary of the LLaDA warm-head Q64 single factor.

Inputs are cluster-side receipts only.  The failed first run is excluded by
path and never loaded; the passed pilot (0/10/20) and the pinned remaining 87
are combined.  Reports original30 / new60 / all90 coverage and paired old/new
metrics. Macro TPF is ``mean(N_i/F_i)``; token efficiency is ``sum(N_i)/sum(Q_i)``.
Macro TE and pooled TPF are auxiliary. If a cohort is incomplete, its
metrics are explicitly marked ``partial=True`` and present-only, never an
overall score.  ``F`` is the self-developed executor's actual forward count,
not an extra native-decoder measurement.

Standard library only: no torch, no model, no inference, no traces or
``output_ids`` in the output.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

PROTOCOL = "llada-warm-head-paired-summary-v1"
COHORTS = {"original30": tuple(range(30)), "new60": tuple(range(30, 90)),
           "all90": tuple(range(90))}
RAW_KEYS = frozenset({"trace", "output_ids", "rows", "sampled", "mask",
                      "cold", "dirty", "prefix", "dirty_before", "dirty_after"})


class SummaryError(ValueError):
    """Receipts are structurally unusable; fail loudly instead of mis-summarizing."""


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SummaryError(f"cannot read {path}: {exc}") from exc


def load_run(run_dir, label):
    """Load one run's rows and manifest; validate the paired row schema."""
    run_dir = Path(run_dir)
    rows_dir = run_dir / "rows"
    if not rows_dir.is_dir():
        raise SummaryError(f"{label}: missing rows directory {rows_dir}")
    rows = {}
    for path in sorted(rows_dir.glob("*.json")):
        row = _read(path)
        if not isinstance(row, dict):
            raise SummaryError(f"{path}: row is not an object")
        index = row.get("index")
        if type(index) is not int or not 0 <= index < 90:
            raise SummaryError(f"{path}: invalid index {index!r}")
        if index in rows:
            raise SummaryError(f"{path}: duplicate index {index}")
        for key in ("old", "new", "compare"):
            if key not in row:
                raise SummaryError(f"{path}: missing {key}")
        rows[index] = row
    manifest = _read(run_dir / "run.json") if (run_dir / "run.json").exists() else {}
    return rows, manifest


def combine(pilot_rows, pilot_run, pinned_rows, pinned_run):
    """Combine the pilot and pinned rows; require one protocol and no overlap."""
    for key in ("protocol", "q", "mode", "coverage", "checkpoint",
                "samples_sha256", "old_source"):
        if pilot_run.get(key) != pinned_run.get(key):
            raise SummaryError(f"pilot/pinned manifest mismatch on {key}")
    rows = {}
    for source, data in (("pilot", pilot_rows), ("pinned", pinned_rows)):
        for index, row in data.items():
            if index in rows:
                raise SummaryError(f"index {index} appears in both pilot and pinned")
            row = dict(row)
            row["_source"] = source
            rows[index] = row
    if not set(rows) <= set(range(90)):
        raise SummaryError("combined indices escape 0..89")
    return rows


def verify_pair(row):
    """Independently re-check the first intervention from the two traces."""
    old = row.get("old") or {}
    new = row.get("new") or {}
    compared = row.get("compare") or {}
    intervened = bool(compared.get("intervened"))
    out = dict(index=row["index"], source=row.get("_source"), intervened=intervened,
               declared_passed=bool(compared.get("passed")),
               declared_pre_state_verified=bool(compared.get("pre_state_verified")))
    if not intervened:
        out["independent_passed"] = None
        out["independent_pre_state_verified"] = None
        return out
    tick = compared.get("tick")
    old_by_tick = {event.get("tick"): event for event in old.get("trace") or []}
    new_by_tick = {event.get("tick"): event for event in new.get("trace") or []}
    old_event = old_by_tick.get(tick)
    new_event = new_by_tick.get(tick)
    added = list(compared.get("added") or [])
    checks = {}
    if old_event is None or new_event is None:
        checks["old_tick_present"] = old_event is not None
        checks["new_tick_present"] = new_event is not None
    else:
        checks["old_tick_present"] = True
        checks["new_tick_present"] = True
        checks["rows_equal"] = new_event.get("rows") == old_event.get("rows")
        checks["dirty_before_equal"] = (new_event.get("dirty_before")
                                        == old_event.get("dirty_before"))
        old_pre = old_event.get("pre_state")
        checks["pre_state_equal"] = ((new_event.get("pre_state") == old_pre)
                                     if old_pre is not None else None)
        checks["added_in_rows"] = set(added) <= set(new_event.get("rows") or [])
        checks["sampled_union"] = (sorted(set(old_event.get("sampled") or []) | set(added))
                                   == (new_event.get("sampled") or []))
        checks["added_count"] = new_event.get("warm_head_sample") == len(added)
        checks["no_extra_rows"] = (set(new_event.get("rows") or [])
                                   == set(old_event.get("rows") or []))
        checks["no_prefix_dirty"] = not (
            set(added) & (set(new_event.get("prefix") or [])
                          | set(new_event.get("dirty") or [])))
        checks["same_forward"] = (
            new_event.get("layer_physical_q") == [len(new_event.get("rows") or [])] * 20
            and new_event.get("q") == len(new_event.get("rows") or [])
            and len(new_event.get("sampled") or []) <= len(new_event.get("rows") or []))
    out["independent_passed"] = all(value is True for value in checks.values()
                                    if value is not None)
    out["independent_pre_state_verified"] = checks.get("pre_state_equal") is not None
    out["independent_checks"] = checks
    out["first_tick"] = tick
    out["added_count"] = len(added)
    return out


def arm_metrics(arm, q=64):
    """Common per-arm metrics: per-request TPF, pooled TE, strict cap, native F."""
    n = int(arm.get("N", 0))
    f = int(arm.get("F", 0))
    physical_q = int(arm.get("Q", 0))
    trace = arm.get("trace") or []
    max_q = max((int(event.get("q", 0)) for event in trace), default=0)
    strict = all(int(event.get("q", 0)) <= q
                 and event.get("layer_physical_q", [event.get("q", 0)] * 20)
                 == [event.get("q", 0)] * 20
                 for event in trace)
    return dict(N=n, F=f, Q=physical_q, executor_F=f,
                TPF=(n / f) if f else None, TE=(n / physical_q) if physical_q else None,
                max_Q_per_forward=max_q, strict_cap=bool(strict))


def summarize_cohort(name, indices, rows):
    present = [index for index in indices if index in rows]
    missing = [index for index in indices if index not in rows]
    complete = not missing
    stats = dict(
        requests=len(present), old_correct=0, new_correct=0,
        old_N=0, new_N=0, old_F=0, new_F=0, old_Q=0, new_Q=0,
        old_failures=0, new_failures=0, intervened=0,
        declared_first_passed=0, independent_first_passed=0,
        independent_pre_state_verified=0, both_true=0, both_false=0,
        old_true_new_false=0, old_false_new_true=0,
        delta_N=0, delta_F=0, delta_Q=0, per_index=[])
    first_ticks, added_counts = [], []
    for index in present:
        row = rows[index]
        old = row.get("old") or {}
        new = row.get("new") or {}
        old_fail = "old_failure" in row or bool(old.get("failure"))
        new_fail = "new_failure" in row or bool(new.get("failure"))
        old_correct, new_correct = bool(old.get("correct")), bool(new.get("correct"))
        pair = verify_pair(row)
        old_m = arm_metrics(old)
        new_m = arm_metrics(new)
        stats["old_failures"] += int(old_fail)
        stats["new_failures"] += int(new_fail)
        stats["old_correct"] += int(old_correct)
        stats["new_correct"] += int(new_correct)
        stats["old_N"] += old_m["N"]
        stats["new_N"] += new_m["N"]
        stats["old_F"] += old_m["F"]
        stats["new_F"] += new_m["F"]
        stats["old_Q"] += old_m["Q"]
        stats["new_Q"] += new_m["Q"]
        stats["delta_N"] += new_m["N"] - old_m["N"]
        stats["delta_F"] += new_m["F"] - old_m["F"]
        stats["delta_Q"] += new_m["Q"] - old_m["Q"]
        stats["both_true"] += int(old_correct and new_correct)
        stats["both_false"] += int(not old_correct and not new_correct)
        stats["old_true_new_false"] += int(old_correct and not new_correct)
        stats["old_false_new_true"] += int(not old_correct and new_correct)
        stats["intervened"] += int(pair["intervened"])
        stats["declared_first_passed"] += int(pair["declared_passed"])
        stats["independent_first_passed"] += int(pair["independent_passed"] is True)
        stats["independent_pre_state_verified"] += int(
            pair["independent_pre_state_verified"] is True)
        if pair["intervened"]:
            first_ticks.append(pair["first_tick"])
            added_counts.append(pair["added_count"])
        stats["per_index"].append(dict(
            index=index, source=row.get("_source"),
            old_correct=old_correct, new_correct=new_correct,
            old_N=old.get("N"), new_N=new.get("N"),
            old_F=old.get("F"), new_F=new.get("F"),
            old_Q=old.get("Q"), new_Q=new.get("Q"),
            old_executor_F=old_m["executor_F"], new_executor_F=new_m["executor_F"],
            old_TPF=old_m["TPF"], new_TPF=new_m["TPF"],
            old_TE=old_m["TE"], new_TE=new_m["TE"],
            old_max_Q=old_m["max_Q_per_forward"], new_max_Q=new_m["max_Q_per_forward"],
            old_strict_cap=old_m["strict_cap"], new_strict_cap=new_m["strict_cap"],
            old_failure=bool(old_fail), new_failure=bool(new_fail),
            intervened=pair["intervened"], first_tick=pair.get("first_tick"),
            added_count=pair.get("added_count"),
            declared_first_passed=pair["declared_passed"],
            independent_first_passed=pair["independent_passed"],
            independent_pre_state_verified=pair["independent_pre_state_verified"]))
    stats["coverage"] = dict(
        expected=len(indices), present=len(present), missing=missing,
        complete=complete,
        fraction=(len(present) / len(indices)) if indices else 1.0)
    stats["partial"] = not complete
    stats["metrics_scope"] = "all" if complete else "present_only"
    stats["old_pooled_TPF"] = (stats["old_N"] / stats["old_F"]) if stats["old_F"] else None
    stats["new_pooled_TPF"] = (stats["new_N"] / stats["new_F"]) if stats["new_F"] else None
    stats["delta_pooled_TPF"] = (
        stats["new_pooled_TPF"] - stats["old_pooled_TPF"]
        if stats["old_pooled_TPF"] is not None and stats["new_pooled_TPF"] is not None
        else None)
    stats["old_pooled_TE"] = (stats["old_N"] / stats["old_Q"]) if stats["old_Q"] else None
    stats["new_pooled_TE"] = (stats["new_N"] / stats["new_Q"]) if stats["new_Q"] else None
    stats["delta_pooled_TE"] = (
        stats["new_pooled_TE"] - stats["old_pooled_TE"]
        if stats["old_pooled_TE"] is not None and stats["new_pooled_TE"] is not None
        else None)
    old_macro_tpf = [row["old_TPF"] for row in stats["per_index"] if row["old_TPF"] is not None]
    new_macro_tpf = [row["new_TPF"] for row in stats["per_index"] if row["new_TPF"] is not None]
    old_macro_te = [row["old_TE"] for row in stats["per_index"] if row["old_TE"] is not None]
    new_macro_te = [row["new_TE"] for row in stats["per_index"] if row["new_TE"] is not None]
    stats["old_macro_TPF"] = sum(old_macro_tpf) / len(old_macro_tpf) if old_macro_tpf else None
    stats["new_macro_TPF"] = sum(new_macro_tpf) / len(new_macro_tpf) if new_macro_tpf else None
    stats["delta_macro_TPF"] = (
        stats["new_macro_TPF"] - stats["old_macro_TPF"]
        if stats["old_macro_TPF"] is not None and stats["new_macro_TPF"] is not None else None)
    stats["old_macro_TE"] = sum(old_macro_te) / len(old_macro_te) if old_macro_te else None
    stats["new_macro_TE"] = sum(new_macro_te) / len(new_macro_te) if new_macro_te else None
    stats["delta_macro_TE"] = (
        stats["new_macro_TE"] - stats["old_macro_TE"]
        if stats["old_macro_TE"] is not None and stats["new_macro_TE"] is not None else None)
    stats["old_executor_F"] = stats["old_F"]
    stats["new_executor_F"] = stats["new_F"]
    stats["old_strict_cap"] = all(row["old_strict_cap"] for row in stats["per_index"])
    stats["new_strict_cap"] = all(row["new_strict_cap"] for row in stats["per_index"])
    stats["old_max_Q_per_forward"] = max(
        (row["old_max_Q"] for row in stats["per_index"]), default=0)
    stats["new_max_Q_per_forward"] = max(
        (row["new_max_Q"] for row in stats["per_index"]), default=0)
    stats["first_tick"] = dict(
        min=min(first_ticks) if first_ticks else None,
        median=statistics.median(first_ticks) if first_ticks else None,
        max=max(first_ticks) if first_ticks else None)
    stats["added_count"] = dict(
        total=sum(added_counts),
        mean=(sum(added_counts) / len(added_counts)) if added_counts else None,
        max=max(added_counts) if added_counts else None)
    return stats


def _assert_no_raw(value, path="root"):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in RAW_KEYS:
                raise SummaryError(f"refusing to emit raw key {key!r} at {path}")
            _assert_no_raw(item, f"{path}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _assert_no_raw(item, f"{path}[{i}]")


def write_cohort_csv(result, path):
    columns = ("cohort", "expected", "present", "complete", "partial",
               "old_correct", "new_correct", "old_N", "new_N", "old_F", "new_F",
               "old_Q", "new_Q", "old_failures", "new_failures", "intervened",
               "declared_first_passed", "independent_first_passed",
               "independent_pre_state_verified", "both_true", "both_false",
               "old_true_new_false", "old_false_new_true", "delta_N", "delta_F",
               "delta_Q", "old_executor_F", "new_executor_F",
               "old_macro_TPF", "new_macro_TPF", "delta_macro_TPF",
               "old_macro_TE", "new_macro_TE", "delta_macro_TE",
               "old_pooled_TPF", "new_pooled_TPF", "delta_pooled_TPF",
               "old_pooled_TE", "new_pooled_TE", "delta_pooled_TE",
               "old_max_Q_per_forward", "new_max_Q_per_forward",
               "old_strict_cap", "new_strict_cap",
               "first_tick_min", "first_tick_median", "first_tick_max")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for name, stats in result["cohorts"].items():
            row = {key: stats.get(key) for key in columns if key in stats}
            row.update(cohort=name, expected=stats["coverage"]["expected"],
                       present=stats["coverage"]["present"],
                       complete=stats["coverage"]["complete"], partial=stats["partial"],
                       first_tick_min=stats["first_tick"]["min"],
                       first_tick_median=stats["first_tick"]["median"],
                       first_tick_max=stats["first_tick"]["max"])
            writer.writerow(row)


def write_pair_csv(result, path):
    columns = ("index", "source", "old_correct", "new_correct", "old_N", "new_N",
               "old_F", "new_F", "old_Q", "new_Q", "old_failure", "new_failure",
               "old_executor_F", "new_executor_F",
               "old_TPF", "new_TPF", "old_TE", "new_TE", "old_max_Q", "new_max_Q",
               "old_strict_cap", "new_strict_cap", "intervened", "first_tick",
               "added_count", "declared_first_passed", "independent_first_passed",
               "independent_pre_state_verified")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in result["cohorts"]["all90"]["per_index"]:
            writer.writerow({key: row.get(key) for key in columns})


def print_summary(result):
    def fmt(value, spec=".3f"):
        return format(value, spec) if value is not None else "-"

    print("cohort       present  complete  old_correct new_correct  "
          "intervened declared independent  oldF newF  oldQ newQ  deltaQ  "
          "macroTPF(old/new)  macroTE(old/new)  pooledTPF(old/new)  strict(old/new)")
    for name, stats in result["cohorts"].items():
        cov = stats["coverage"]
        print(f"{name:<12} {cov['present']:>3}/{cov['expected']:<3} "
              f"{str(cov['complete']):>8}  {stats['old_correct']:>10} "
              f"{stats['new_correct']:>10}  {stats['intervened']:>10} "
              f"{stats['declared_first_passed']:>8} {stats['independent_first_passed']:>11}  "
              f"{stats['old_F']:>4} {stats['new_F']:>4} {stats['old_Q']:>6} "
              f"{stats['new_Q']:>6} {stats['delta_Q']:>7}  "
              f"{fmt(stats['old_macro_TPF'])}/{fmt(stats['new_macro_TPF'])}  "
              f"{fmt(stats['old_macro_TE'], '.4f')}/{fmt(stats['new_macro_TE'], '.4f')}  "
              f"{fmt(stats['old_pooled_TPF'])}/{fmt(stats['new_pooled_TPF'])}  "
              f"{stats['old_strict_cap']}/{stats['new_strict_cap']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--pinned", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, default=None,
                        help="failed first run to exclude; metadata only, never loaded")
    args = parser.parse_args(argv)
    if args.exclude and args.exclude.resolve() in (args.pilot.resolve(), args.pinned.resolve()):
        raise SummaryError("--exclude must not be an input run")
    pilot_rows, pilot_run = load_run(args.pilot, "pilot")
    pinned_rows, pinned_run = load_run(args.pinned, "pinned")
    rows = combine(pilot_rows, pilot_run, pinned_rows, pinned_run)
    cohorts = {name: summarize_cohort(name, indices, rows)
               for name, indices in COHORTS.items()}
    source = dict(
        pilot=str(args.pilot), pinned=str(args.pinned),
        excluded=str(args.exclude) if args.exclude else None,
        pilot_rows=len(pilot_rows), pinned_rows=len(pinned_rows),
        pilot_indices=sorted(pilot_rows), pinned_indices=sorted(pinned_rows),
        combined_indices=sorted(rows),
        missing_all90=[i for i in range(90) if i not in rows],
        protocol=pilot_run.get("protocol"), q=pilot_run.get("q"),
        mode=pilot_run.get("mode"), coverage=pilot_run.get("coverage"),
        checkpoint=pilot_run.get("checkpoint"),
        samples_sha256=pilot_run.get("samples_sha256"),
        old_source=pilot_run.get("old_source"))
    if args.exclude and (args.exclude / "summary.json").exists():
        excluded = _read(args.exclude / "summary.json")
        source["excluded_summary"] = {
            key: excluded.get(key) for key in
            ("requests", "old_failures", "new_failures", "intervened",
             "first_checks_passed")}
        excluded_rows = args.exclude / "rows"
        source["excluded_rows"] = (len(list(excluded_rows.glob("*.json")))
                                   if excluded_rows.is_dir() else 0)
    result = dict(protocol=PROTOCOL, source=source, cohorts=cohorts)
    _assert_no_raw(result)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary-warm-head-single-factor.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_cohort_csv(result, args.out_dir / "summary-warm-head-single-factor.csv")
    write_pair_csv(result, args.out_dir / "paired-warm-head-single-factor.csv")
    print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

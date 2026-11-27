#!/usr/bin/env python3
"""Read-only paired summary for the Q64 gate single factor (half vs quarter).

Combines the trusted control receipts (0..59, ``c64-full-half`` vs
``c64-full-quarter``) with the GPU5 gate run (60..89).  The same-device 0/10/20
pilot is checked against the control before it is trusted.  Reports
original30 / new60 / all90 with per-request N/F, pooled TPF/TE, strict cap and
executor F. Standard library only; no traces or ``output_ids`` in the output.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.t2t.serving_slots.llada_hfull_gate_single_factor import (
    compare_pair, summarize_cohort)

PROTOCOL = "llada-gate-single-factor-summary-v1"
COHORTS = {"original30": tuple(range(30)), "new60": tuple(range(30, 90)),
           "all90": tuple(range(90))}
RAW_KEYS = frozenset({"trace", "output_ids", "rows", "sampled", "mask",
                      "cold", "dirty", "prefix", "dirty_before", "dirty_after"})


class SummaryError(ValueError):
    pass


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SummaryError(f"cannot read {path}: {exc}") from exc


def _strip_pre_state(trace):
    return [{k: v for k, v in event.items() if k != "pre_state"} for event in trace]


def control_pair(control_rows, index):
    old = _read(control_rows / f"c64-full-half-{index:02d}.json")
    new = _read(control_rows / f"c64-full-quarter-{index:02d}.json")
    return dict(index=index, old=old, new=new,
                compare=compare_pair(old, new, 64), source="control")


def gate_pair(run_dir, index):
    row = _read(Path(run_dir) / "rows" / f"{index:02d}.json")
    row = dict(row)
    row["source"] = "gate"
    row["compare"] = compare_pair(row["old"], row["new"], 64)
    return row


def pilot_check(pilot_dir, control_rows, indices=(0, 10, 20)):
    """Check the same-device 0/10/20 pilot against the trusted control."""
    checks = []
    for index in indices:
        row = _read(Path(pilot_dir) / "rows" / f"{index:02d}.json")
        old = _read(control_rows / f"c64-full-half-{index:02d}.json")
        new = _read(control_rows / f"c64-full-quarter-{index:02d}.json")
        for arm, control, label in (("old", old, "half"), ("new", new, "quarter")):
            gate = row[arm]
            checks.append(dict(
                index=index, arm=arm, gate=label,
                output_ids_equal=gate["output_ids"] == control["output_ids"],
                trace_equal=(_strip_pre_state(gate["trace"])
                             == _strip_pre_state(control["trace"])),
                F_equal=gate["F"] == control["F"],
                Q_equal=gate["Q"] == control["Q"],
                correct_equal=gate["correct"] == control["correct"]))
    return checks


def build_rows(control_rows, gate_rest, gate_pilot=None):
    rows = {}
    for index in range(60):
        path = Path(control_rows) / f"c64-full-half-{index:02d}.json"
        if path.exists():
            rows[index] = control_pair(Path(control_rows), index)
        elif gate_pilot is not None and (Path(gate_pilot) / "rows" / f"{index:02d}.json").exists():
            rows[index] = gate_pair(gate_pilot, index)
    for index in range(60, 90):
        path = Path(gate_rest) / "rows" / f"{index:02d}.json"
        if path.exists():
            rows[index] = gate_pair(gate_rest, index)
    return rows


def _assert_no_raw(value, path="root"):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in RAW_KEYS:
                raise SummaryError(f"refusing to emit raw key {key!r} at {path}")
            _assert_no_raw(item, f"{path}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _assert_no_raw(item, f"{path}[{i}]")


def write_csv(summary, path):
    columns = ("cohort", "expected", "present", "complete", "partial",
               "old_correct", "new_correct", "old_N", "new_N", "old_F",
               "new_F", "old_Q", "new_Q", "old_pooled_TPF", "new_pooled_TPF",
               "old_pooled_TE", "new_pooled_TE", "old_macro_TPF",
               "new_macro_TPF", "old_macro_TE", "new_macro_TE",
               "old_executor_F", "new_executor_F", "old_strict_cap",
               "new_strict_cap", "old_max_Q_per_forward",
               "new_max_Q_per_forward", "pre_state_equal_tick0", "diverged")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for name, stats in summary["cohorts"].items():
            row = {key: stats.get(key) for key in columns if key in stats}
            row.update(cohort=name, expected=stats["coverage"]["expected"],
                       present=stats["coverage"]["present"],
                       complete=stats["coverage"]["complete"], partial=stats["partial"])
            writer.writerow(row)


def write_pair_csv(summary, path):
    columns = ("index", "source", "old_correct", "new_correct", "old_N",
               "new_N", "old_F", "new_F", "old_Q", "new_Q", "old_executor_F",
               "new_executor_F", "old_TPF", "new_TPF", "old_TE", "new_TE",
               "old_max_Q", "new_max_Q",
               "old_strict_cap", "new_strict_cap", "old_failure",
               "new_failure", "first_divergence")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in summary["cohorts"]["all90"]["per_index"]:
            writer.writerow({key: row.get(key) for key in columns})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-rows", type=Path, required=True,
                        help="trusted control rows dir (c64-full-half/quarter)")
    parser.add_argument("--gate-rest", type=Path, required=True,
                        help="GPU5 gate run for 60..89")
    parser.add_argument("--gate-pilot", type=Path, default=None,
                        help="same-device 0/10/20 gate pilot, checked against control")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = build_rows(args.control_rows, args.gate_rest, args.gate_pilot)
    cohorts = {name: summarize_cohort(name, indices, rows)
               for name, indices in COHORTS.items()}
    checks = pilot_check(args.gate_pilot, args.control_rows) if args.gate_pilot else []
    summary = dict(
        protocol=PROTOCOL,
        source=dict(control_rows=str(args.control_rows),
                    gate_rest=str(args.gate_rest),
                    gate_pilot=str(args.gate_pilot) if args.gate_pilot else None,
                    indices=sorted(rows), missing_all90=[i for i in range(90) if i not in rows],
                    old_mode="half", new_mode="quarter",
                    executor_F_note="F is the self-developed executor's actual forward count; Q is physical rows under strict cap",
                    pilot_check=checks),
        cohorts=cohorts)
    _assert_no_raw(summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary-gate-single-factor.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_csv(summary, args.out_dir / "summary-gate-single-factor.csv")
    write_pair_csv(summary, args.out_dir / "paired-gate-single-factor.csv")
    print(json.dumps({name: {k: stats[k] for k in (
        "requests", "old_correct", "new_correct", "old_F", "new_F",
        "old_Q", "new_Q", "old_pooled_TPF", "new_pooled_TPF",
        "old_pooled_TE", "new_pooled_TE", "old_strict_cap", "new_strict_cap")}
        for name, stats in cohorts.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

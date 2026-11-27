#!/usr/bin/env python3
"""Read-only audit of warm-head MASK sampling in LLaDA full-head receipts.

Question
--------
``_head_full_plan`` in ``bdlm_serving/methods/t2t/llada_budget_request.py``
uses ``sampled = cold_sel or rows & eligible``.  When every cold row is in the
next block (``cold_sel`` is tail-only), the warm head MASK rows that are still
projected into the physical Q are not sampled in that forward.  This module
quantifies that from already-written receipts.

Scope
-----
Standard library only.  It never imports ``bdlm_serving``, ``torch``, or a
model, and it never runs inference.  It reads row JSON files and writes
aggregates only (no rows, masks, sampled lists, traces, or output IDs).

Definitions
-----------
For one trace event with ``block``:

* ``split = (block + 1) * 32`` is the boundary between the current head block
  and the next block.
* ``head rows`` are ``rows < split``; ``warm head MASK rows`` are
  ``mask rows < split`` (``plan.mask`` excludes cold and dirty rows).
* ``head sampled`` is ``len([p for p in sampled if p < split])``.
* ``cold_tail_only`` means ``cold`` is non-empty and every cold row is
  ``>= split``.
* ``affected`` means a full-head forward with ``cold_tail_only``, at least one
  head MASK at plan time, no head row sampled, and not a pure drain.  In this
  branch the warm head MASK rows were projected (paid) but not sampled.
* ``paid warm head MASK rows`` is the number of warm head MASK rows in an
  affected forward.
* ``head sample=0 F`` is the number of forwards with no head row sampled; the
  subset that still had head MASKs is reported separately.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

B = 32
FULL_GATES = ("off", "eager", "half", "quarter")
DEFAULT_ARMS = tuple(f"c64-full-{gate}" for gate in FULL_GATES)
PROTOCOL = "hfull-sampled-audit-v1"

# Never emit raw per-forward state; only counts and first-tick aggregates.
FORBIDDEN_RAW_KEYS = frozenset({
    "rows", "mask", "sampled", "cold", "trace", "output_ids", "dirty",
    "prefix", "dirty_before", "dirty_after", "written_absolute", "absolute_positions",
})


class AuditError(ValueError):
    """A receipt is structurally unusable; fail loudly instead of mis-counting."""


def _int_list(value, name, where):
    if not isinstance(value, list) or any(type(v) is not int for v in value):
        raise AuditError(f"{where}: {name} must be a list of ints")
    return value


def split_of(event):
    block = event.get("block")
    if type(block) is not int or block < 0:
        raise AuditError("trace event has no valid block")
    return (block + 1) * B


def event_metrics(event, where="trace"):
    """Classify one forward.  Raises on any structural inconsistency."""
    required = ("block", "tick", "q", "phase", "rows", "mask", "sampled",
                "cold", "head_mask_at_plan", "head_q", "tail_q")
    missing = [key for key in required if key not in event]
    if missing:
        raise AuditError(f"{where}: missing {missing}")
    block, tick = event["block"], event["tick"]
    if type(block) is not int or block < 0 or type(tick) is not int or tick < 0:
        raise AuditError(f"{where}: invalid block/tick")
    rows = _int_list(event["rows"], "rows", where)
    mask = _int_list(event["mask"], "mask", where)
    sampled = _int_list(event["sampled"], "sampled", where)
    cold = _int_list(event["cold"], "cold", where)
    q = event["q"]
    if type(q) is not int or q != len(rows):
        raise AuditError(f"{where}: q != len(rows)")
    plan_masks = event["head_mask_at_plan"]
    if type(plan_masks) is not int or plan_masks < 0:
        raise AuditError(f"{where}: invalid head_mask_at_plan")
    if not (set(mask) <= set(rows) and set(sampled) <= set(rows)
            and set(cold) <= set(rows)):
        raise AuditError(f"{where}: selected list escapes rows")
    split = split_of(event)
    head_rows = [p for p in rows if p < split]
    tail_rows = [p for p in rows if p >= split]
    if event["head_q"] != len(head_rows) or event["tail_q"] != len(tail_rows):
        raise AuditError(f"{where}: head_q/tail_q disagree with split")
    warm_head = [p for p in mask if p < split]
    cold_head = [p for p in cold if p < split]
    cold_tail = [p for p in cold if p >= split]
    head_sampled = [p for p in sampled if p < split]
    head_full = event.get("head_full") is True
    if head_full and plan_masks != len(warm_head) + len(cold_head):
        raise AuditError(f"{where}: head_mask_at_plan != warm + cold head MASKs")
    cold_tail_only = bool(cold) and not cold_head
    drain_only = bool(event.get("drain_only", False))
    affected = (head_full and cold_tail_only and plan_masks > 0
                and not head_sampled and not drain_only)
    return dict(
        tick=tick,
        block=block,
        split=split,
        q=q,
        phase=event["phase"],
        head_rows=len(head_rows),
        tail_rows=len(tail_rows),
        warm_head_mask_rows=len(warm_head),
        cold_head=len(cold_head),
        cold_tail=len(cold_tail),
        head_sampled=len(head_sampled),
        tail_sampled=len(sampled) - len(head_sampled),
        head_mask_at_plan=plan_masks,
        head_full=head_full,
        cold_tail_only=cold_tail_only,
        drain_only=drain_only,
        head_sample0=not head_sampled,
        head_sample0_with_head_mask=(not head_sampled and plan_masks > 0),
        affected=affected,
        paid_warm_head_mask_rows=(len(warm_head) if affected else 0),
    )


def load_rows(rows_dir):
    """Read ``rows/*.json`` into ``{arm: {index: row}}``; skip ``.lock`` files."""
    if not os.path.isdir(rows_dir):
        raise AuditError(f"rows directory does not exist: {rows_dir}")
    files = sorted(glob.glob(os.path.join(rows_dir, "*.json")))
    if not files:
        raise AuditError(f"no row JSON files in {rows_dir}")
    data = defaultdict(dict)
    for path in files:
        try:
            with open(path, encoding="utf-8") as handle:
                row = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise AuditError(f"cannot read {path}: {exc}") from exc
        if not isinstance(row, dict):
            raise AuditError(f"{path}: row is not an object")
        for key in ("arm", "index", "trace"):
            if key not in row:
                raise AuditError(f"{path}: missing {key}")
        arm, index = row["arm"], row["index"]
        if not isinstance(arm, str) or type(index) is not int or index < 0:
            raise AuditError(f"{path}: invalid arm/index")
        if not isinstance(row["trace"], list):
            raise AuditError(f"{path}: trace is not a list")
        if index in data[arm]:
            raise AuditError(f"{path}: duplicate index {index} for arm {arm}")
        data[arm][index] = row
    return {arm: dict(rows) for arm, rows in data.items()}, files


def _blank():
    return dict(
        total_F=0,
        head_full_F=0,
        not_head_full_F=0,
        total_warm_head_mask_rows=0,
        head_sample0_F=0,
        head_sample0_with_head_mask_F=0,
        cold_tail_only_F=0,
        cold_mixed_F=0,
        cold_head_only_F=0,
        no_cold_F=0,
        affected_F=0,
        paid_warm_head_mask_rows=0,
    )


def _add(stats, metrics):
    stats["total_F"] += 1
    stats["head_full_F"] += int(metrics["head_full"])
    stats["not_head_full_F"] += int(not metrics["head_full"])
    stats["total_warm_head_mask_rows"] += metrics["warm_head_mask_rows"]
    stats["head_sample0_F"] += int(metrics["head_sample0"])
    stats["head_sample0_with_head_mask_F"] += int(metrics["head_sample0_with_head_mask"])
    stats["cold_tail_only_F"] += int(metrics["cold_tail_only"])
    stats["cold_mixed_F"] += int(metrics["cold_head"] and metrics["cold_tail"])
    stats["cold_head_only_F"] += int(metrics["cold_head"] and not metrics["cold_tail"])
    stats["no_cold_F"] += int(not metrics["cold_head"] and not metrics["cold_tail"])
    if metrics["affected"]:
        stats["affected_F"] += 1
        stats["paid_warm_head_mask_rows"] += metrics["paid_warm_head_mask_rows"]


def _finalize(stats, questions):
    """Attach per-question first-tick aggregates to a counter dict."""
    ticks = sorted(q["first_tick"] for q in questions if q["first_tick"] is not None)
    stats["requests"] = len(questions)
    stats["affected_questions"] = sum(1 for q in questions if q["affected_F"])
    stats["first_tick"] = {
        "min": ticks[0] if ticks else None,
        "median": statistics.median(ticks) if ticks else None,
        "max": ticks[-1] if ticks else None,
        "per_question": {str(q["index"]): q["first_tick"] for q in questions
                         if q["first_tick"] is not None},
    }
    stats["affected_F_per_question"] = {str(q["index"]): q["affected_F"]
                                        for q in questions if q["affected_F"]}
    stats["affected_F_share"] = (stats["affected_F"] / stats["total_F"]
                                 if stats["total_F"] else 0.0)
    stats["paid_share"] = (stats["paid_warm_head_mask_rows"]
                           / stats["total_warm_head_mask_rows"]
                           if stats["total_warm_head_mask_rows"] else 0.0)
    return stats


def audit(rows_dir, arms=DEFAULT_ARMS, indices=None):
    """Audit the common question batch for each arm; return aggregates only."""
    data, files = load_rows(rows_dir)
    missing = [arm for arm in arms if arm not in data]
    if missing:
        raise AuditError(f"arms absent from rows: {missing}")
    all_common = sorted(set.intersection(*(set(data[arm]) for arm in arms)))
    common = all_common
    if indices is not None:
        wanted = sorted({int(i) for i in indices})
        absent = [i for i in wanted if i not in all_common]
        if absent:
            raise AuditError(f"requested indices not common to all arms: {absent}")
        common = wanted
    if not common:
        raise AuditError("arms share no question index")

    samples = sorted({row.get("samples_sha256") for arm in arms
                      for row in data[arm].values() if row.get("samples_sha256")})
    datasets = {i: data[arms[0]][i].get("dataset") for i in common}
    result = {
        "protocol": PROTOCOL,
        "source": {
            "captured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "rows_dir": os.path.abspath(rows_dir),
            "row_files": len(files),
            "arms": {arm: len(data[arm]) for arm in arms},
            "arms_audited": list(arms),
            "samples_sha256": samples,
            "common_indices": common,
            "all_common_indices_at_capture": all_common,
            "requested_indices": sorted({int(i) for i in indices}) if indices is not None else None,
            "common_datasets": dict(Counter(datasets.values())),
        },
        "gates": {},
    }
    for arm in arms:
        first_row = data[arm][common[0]]
        gate = first_row.get("cross_open")
        coverage = first_row.get("coverage")
        for i in common:
            row = data[arm][i]
            if row.get("cross_open") != gate or row.get("coverage") != coverage:
                raise AuditError(f"{arm}/{i}: gate/coverage differs within arm")
        questions = []
        for i in common:
            row = data[arm][i]
            stats = _blank()
            first_tick = None
            for pos, event in enumerate(row["trace"]):
                metrics = event_metrics(event, f"{arm}/{i}/trace[{pos}]")
                _add(stats, metrics)
                if metrics["affected"]:
                    first_tick = metrics["tick"] if first_tick is None else min(first_tick, metrics["tick"])
            stats["index"] = i
            stats["dataset"] = row.get("dataset")
            stats["first_tick"] = first_tick
            questions.append(stats)
        overall = _blank()
        for stats in questions:
            for key in overall:
                overall[key] += stats[key]
        _finalize(overall, questions)
        overall["arm"] = arm
        overall["gate"] = gate
        overall["coverage"] = coverage
        by_dataset = {}
        for dataset in sorted({q["dataset"] for q in questions}):
            subset = [q for q in questions if q["dataset"] == dataset]
            dstats = _blank()
            for stats in subset:
                for key in dstats:
                    dstats[key] += stats[key]
            _finalize(dstats, subset)
            by_dataset[dataset] = dstats
        overall["by_dataset"] = by_dataset
        result["gates"][arm] = overall
    _assert_no_raw(result)
    return result


def _assert_no_raw(value, path="root"):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_RAW_KEYS:
                raise AuditError(f"refusing to emit raw key {key!r} at {path}")
            _assert_no_raw(item, f"{path}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _assert_no_raw(item, f"{path}[{i}]")


def write_csv(result, path):
    columns = ("arm", "gate", "coverage", "requests", "total_F", "affected_F",
               "affected_questions", "first_tick_min", "first_tick_median",
               "first_tick_max", "paid_warm_head_mask_rows",
               "total_warm_head_mask_rows", "paid_share", "head_sample0_F",
               "head_sample0_with_head_mask_F", "cold_tail_only_F",
               "cold_mixed_F", "cold_head_only_F", "no_cold_F", "affected_F_share")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for arm in result["source"]["arms_audited"]:
            stats = result["gates"][arm]
            row = {key: stats.get(key) for key in columns}
            row["first_tick_min"] = stats["first_tick"]["min"]
            row["first_tick_median"] = stats["first_tick"]["median"]
            row["first_tick_max"] = stats["first_tick"]["max"]
            writer.writerow(row)


def print_summary(result):
    print("gate           req  total_F  aff_F  aff_q  first(min/med/max)  "
          "paid_warm_head_mask  head_s0_F  head_s0_hm_F")
    for arm in result["source"]["arms_audited"]:
        s = result["gates"][arm]
        tick = s["first_tick"]
        triad = "-/-/-" if tick["min"] is None else f"{tick['min']}/{tick['median']}/{tick['max']}"
        print(f"{s['gate']:<14} {s['requests']:>3} {s['total_F']:>8} {s['affected_F']:>6} "
              f"{s['affected_questions']:>6} {triad:>17} {s['paid_warm_head_mask_rows']:>19} "
              f"{s['head_sample0_F']:>10} {s['head_sample0_with_head_mask_F']:>13}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="/gs/results/llada-hfull-paired/rows")
    parser.add_argument("--out", default="/gs/results/llada-hfull-paired/audit-hfull-sampled.json")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--indices", default=None,
                        help="optional comma-separated subset of the common question indices")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    indices = [int(x) for x in args.indices.split(",")] if args.indices else None
    result = audit(args.rows, arms=tuple(args.arms), indices=indices)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if args.csv:
        write_csv(result, args.csv)
    if not args.quiet:
        print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())

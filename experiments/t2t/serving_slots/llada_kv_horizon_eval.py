"""Independent evaluation for the research LLaDA KV-horizon branch (head-only Q8).

Old arm: the main executor with ``RecordingRequest`` (same selected rows and
sampler rule).  New arm: ``KVHorizonRequest`` only.  Both reuse
``evaluate_fixed``; no other agent module is edited.  The first-divergence
ledger records the first tick where outputs/plans differ, the horizon gap, and
whether the selection geometry stayed identical until the visibility effect.
Receipts stay on the cluster; the archive is compact aggregates only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import statistics
import time
from pathlib import Path

from .llada_hfull_gate_single_factor import arm_metrics
from .llada_hfull_single_factor import request_classes
from .llada_kv_horizon_research import KVHorizonRequest

PROTOCOL = "llada-kv-horizon-single-factor-v1"
CROSS = ("off", "eager", "half", "quarter", "head")
DIFF_FIELDS = ("rows", "sampled", "dirty_before", "written_absolute",
               "admitted", "edited", "revoked")
OUTPUT_FIELDS = ("written_absolute", "admitted", "edited", "revoked")
PLAN_FIELDS = ("rows", "sampled")


def run_arm(model, tok, task, prompt, q, law, mode, borrow, coverage, horizon):
    """Reuse the registered evaluator; swap only the request class."""
    from bdlm_serving.methods.t2t import llada_budget_request as request_module
    from .llada_fixed_family_eval import evaluate_fixed

    if horizon:
        cls = KVHorizonRequest
    else:
        _module, recording, _warm = request_classes()
        cls = recording
    original = request_module.LLaDABudgetRequest
    request_module.LLaDABudgetRequest = cls
    try:
        return evaluate_fixed(model, tok, task, prompt, q, law,
                              cross=(mode, borrow), coverage=coverage)
    finally:
        request_module.LLaDABudgetRequest = original


def first_divergence(old_row, new_row):
    """First tick where visibility changes outputs/plans; selection ledger."""
    old_trace = old_row.get("trace") or []
    new_trace = new_row.get("trace") or []
    first_any = first_output = first_plan = first_horizon = None
    for tick, (old, new) in enumerate(zip(old_trace, new_trace)):
        differs = {field: old.get(field) != new.get(field) for field in DIFF_FIELDS}
        if first_any is None and any(differs.values()):
            first_any = tick
        if first_output is None and any(old.get(f) != new.get(f) for f in OUTPUT_FIELDS):
            first_output = tick
        if first_plan is None and any(old.get(f) != new.get(f) for f in PLAN_FIELDS):
            first_plan = tick
        if first_horizon is None and (new.get("kv_horizon_gap") or 0) > 0:
            first_horizon = tick
        if all(value is not None for value in
               (first_any, first_output, first_plan, first_horizon)):
            break
    if first_any is None:
        return dict(
            diverged=False, visibility_only=False, same_selection_before=True,
            same_input=old_row.get("input_sha256") == new_row.get("input_sha256"),
            first_horizon_tick=first_horizon,
            horizon_gap_at_first_horizon=(
                new_trace[first_horizon].get("kv_horizon_gap")
                if first_horizon is not None else None),
            first_output_tick=None, first_plan_tick=None,
            horizon_gap_at_first_output=None, zero_gap_divergence=True)
    old = old_trace[first_any]
    new = new_trace[first_any]
    checks = {f"{field}_equal_at_first": old.get(field) == new.get(field)
              for field in DIFF_FIELDS}
    old_pre, new_pre = old.get("pre_state"), new.get("pre_state")
    checks["pre_state_equal_at_first"] = (
        old_pre == new_pre if old_pre is not None and new_pre is not None else None)
    gap_at_horizon = (new_trace[first_horizon].get("kv_horizon_gap")
                      if first_horizon is not None else None)
    gap_at_output = (new_trace[first_output].get("kv_horizon_gap")
                     if first_output is not None else None)
    same_selection_before = all(
        old_trace[t].get("rows") == new_trace[t].get("rows")
        and old_trace[t].get("sampled") == new_trace[t].get("sampled")
        for t in range(first_output if first_output is not None else first_any))
    visibility_only = bool(
        first_horizon is not None and first_output is not None
        and first_output >= first_horizon
        and (first_plan is None or first_plan >= first_output)
        and same_selection_before)
    return dict(
        diverged=True, visibility_only=visibility_only,
        same_selection_before=same_selection_before,
        same_input=old_row.get("input_sha256") == new_row.get("input_sha256"),
        first_any_tick=first_any, first_output_tick=first_output,
        first_plan_tick=first_plan, first_horizon_tick=first_horizon,
        horizon_gap_at_first_horizon=gap_at_horizon,
        horizon_gap_at_first_output=gap_at_output,
        zero_gap_divergence=(first_horizon is None or gap_at_horizon == 0),
        checks=checks,
        kv_horizon=(new_trace[first_horizon].get("kv_horizon")
                    if first_horizon is not None else None),
        kv_high_water=(new_trace[first_horizon].get("kv_high_water")
                       if first_horizon is not None else None))


def summarize_cohort(name, indices, rows, q):
    present = [index for index in indices if index in rows]
    missing = [index for index in indices if index not in rows]
    complete = not missing
    stats = dict(
        requests=len(present), old_correct=0, new_correct=0,
        old_N=0, new_N=0, old_F=0, new_F=0, old_Q=0, new_Q=0,
        old_failures=0, new_failures=0, diverged=0, visibility_only=0,
        same_selection_before=0, zero_gap_divergence=0,
        first_output_ticks=[], first_plan_ticks=[], first_horizon_ticks=[],
        horizon_gaps_at_horizon=[], horizon_gaps_at_output=[],
        per_index=[])
    for index in present:
        row = rows[index]
        old, new = row.get("old") or {}, row.get("new") or {}
        compared = row.get("compare") or {}
        old_m, new_m = arm_metrics(old, q), arm_metrics(new, q)
        old_correct, new_correct = bool(old.get("correct")), bool(new.get("correct"))
        old_fail = "old_failure" in row or bool(old.get("failure"))
        new_fail = "new_failure" in row or bool(new.get("failure"))
        stats["old_correct"] += int(old_correct)
        stats["new_correct"] += int(new_correct)
        stats["old_failures"] += int(old_fail)
        stats["new_failures"] += int(new_fail)
        stats["old_N"] += old_m["N"]
        stats["new_N"] += new_m["N"]
        stats["old_F"] += old_m["F"]
        stats["new_F"] += new_m["F"]
        stats["old_Q"] += old_m["Q"]
        stats["new_Q"] += new_m["Q"]
        stats["diverged"] += int(bool(compared.get("diverged")))
        stats["visibility_only"] += int(bool(compared.get("visibility_only")))
        stats["same_selection_before"] += int(bool(compared.get("same_selection_before")))
        stats["zero_gap_divergence"] += int(bool(compared.get("zero_gap_divergence")))
        if compared.get("first_output_tick") is not None:
            stats["first_output_ticks"].append(compared["first_output_tick"])
        if compared.get("first_plan_tick") is not None:
            stats["first_plan_ticks"].append(compared["first_plan_tick"])
        if compared.get("first_horizon_tick") is not None:
            stats["first_horizon_ticks"].append(compared["first_horizon_tick"])
        if compared.get("horizon_gap_at_first_horizon") is not None:
            stats["horizon_gaps_at_horizon"].append(compared["horizon_gap_at_first_horizon"])
        if compared.get("horizon_gap_at_first_output") is not None:
            stats["horizon_gaps_at_output"].append(compared["horizon_gap_at_first_output"])
        stats["per_index"].append(dict(
            index=index, old_correct=old_correct, new_correct=new_correct,
            old_N=old.get("N"), new_N=new.get("N"),
            old_F=old.get("F"), new_F=new.get("F"),
            old_Q=old.get("Q"), new_Q=new.get("Q"),
            old_TPF=old_m["TPF"], new_TPF=new_m["TPF"],
            old_TE=old_m["TE"], new_TE=new_m["TE"],
            old_executor_F=old_m["executor_F"], new_executor_F=new_m["executor_F"],
            old_strict_cap=old_m["strict_cap"], new_strict_cap=new_m["strict_cap"],
            old_failure=bool(old_fail), new_failure=bool(new_fail),
            diverged=bool(compared.get("diverged")),
            visibility_only=bool(compared.get("visibility_only")),
            same_selection_before=bool(compared.get("same_selection_before")),
            first_any_tick=compared.get("first_any_tick"),
            first_output_tick=compared.get("first_output_tick"),
            first_plan_tick=compared.get("first_plan_tick"),
            first_horizon_tick=compared.get("first_horizon_tick"),
            horizon_gap_at_first_horizon=compared.get("horizon_gap_at_first_horizon"),
            horizon_gap_at_first_output=compared.get("horizon_gap_at_first_output"),
            zero_gap_divergence=bool(compared.get("zero_gap_divergence")),
            checks=compared.get("checks") or {}))
    stats["coverage"] = dict(
        expected=len(indices), present=len(present), missing=missing,
        complete=complete,
        fraction=(len(present) / len(indices)) if indices else 1.0)
    stats["partial"] = not complete
    stats["metrics_scope"] = "all" if complete else "present_only"
    stats["old_macro_TPF"] = (
        sum(row["old_TPF"] for row in stats["per_index"] if row["old_TPF"] is not None)
        / max(1, sum(1 for row in stats["per_index"] if row["old_TPF"] is not None))
        if any(row["old_TPF"] is not None for row in stats["per_index"]) else None)
    stats["new_macro_TPF"] = (
        sum(row["new_TPF"] for row in stats["per_index"] if row["new_TPF"] is not None)
        / max(1, sum(1 for row in stats["per_index"] if row["new_TPF"] is not None))
        if any(row["new_TPF"] is not None for row in stats["per_index"]) else None)
    stats["old_macro_TE"] = (
        sum(row["old_TE"] for row in stats["per_index"] if row["old_TE"] is not None)
        / max(1, sum(1 for row in stats["per_index"] if row["old_TE"] is not None))
        if any(row["old_TE"] is not None for row in stats["per_index"]) else None)
    stats["new_macro_TE"] = (
        sum(row["new_TE"] for row in stats["per_index"] if row["new_TE"] is not None)
        / max(1, sum(1 for row in stats["per_index"] if row["new_TE"] is not None))
        if any(row["new_TE"] is not None for row in stats["per_index"]) else None)
    stats["old_pooled_TPF"] = stats["old_N"] / stats["old_F"] if stats["old_F"] else None
    stats["new_pooled_TPF"] = stats["new_N"] / stats["new_F"] if stats["new_F"] else None
    stats["old_pooled_TE"] = stats["old_N"] / stats["old_Q"] if stats["old_Q"] else None
    stats["new_pooled_TE"] = stats["new_N"] / stats["new_Q"] if stats["new_Q"] else None
    stats["old_executor_F"] = stats["old_F"]
    stats["new_executor_F"] = stats["new_F"]
    stats["old_strict_cap"] = all(row["old_strict_cap"] for row in stats["per_index"])
    stats["new_strict_cap"] = all(row["new_strict_cap"] for row in stats["per_index"])
    stats["first_output_tick"] = dict(
        min=min(stats["first_output_ticks"]) if stats["first_output_ticks"] else None,
        median=statistics.median(stats["first_output_ticks"]) if stats["first_output_ticks"] else None,
        max=max(stats["first_output_ticks"]) if stats["first_output_ticks"] else None)
    stats["first_plan_tick"] = dict(
        min=min(stats["first_plan_ticks"]) if stats["first_plan_ticks"] else None,
        median=statistics.median(stats["first_plan_ticks"]) if stats["first_plan_ticks"] else None,
        max=max(stats["first_plan_ticks"]) if stats["first_plan_ticks"] else None)
    stats["first_horizon_tick"] = dict(
        min=min(stats["first_horizon_ticks"]) if stats["first_horizon_ticks"] else None,
        median=statistics.median(stats["first_horizon_ticks"]) if stats["first_horizon_ticks"] else None,
        max=max(stats["first_horizon_ticks"]) if stats["first_horizon_ticks"] else None)
    stats["horizon_gap_at_horizon"] = dict(
        min=min(stats["horizon_gaps_at_horizon"]) if stats["horizon_gaps_at_horizon"] else None,
        median=statistics.median(stats["horizon_gaps_at_horizon"]) if stats["horizon_gaps_at_horizon"] else None,
        max=max(stats["horizon_gaps_at_horizon"]) if stats["horizon_gaps_at_horizon"] else None)
    stats["horizon_gap_at_output"] = dict(
        min=min(stats["horizon_gaps_at_output"]) if stats["horizon_gaps_at_output"] else None,
        median=statistics.median(stats["horizon_gaps_at_output"]) if stats["horizon_gaps_at_output"] else None,
        max=max(stats["horizon_gaps_at_output"]) if stats["horizon_gaps_at_output"] else None)
    return stats


def run_index(model, tok, tasks, prompts, index, args):
    from bdlm_serving.methods.t2t.budget_family import Law

    law = Law(*[int(x) for x in args.law.split(",")])
    row = dict(index=index, dataset=tasks[index]["dataset"],
               task_id=tasks[index]["task_id"], seed=tasks[index]["seed"])
    try:
        row["old"] = run_arm(model, tok, tasks[index], prompts[index],
                             args.q, law, args.mode, args.borrow,
                             args.coverage, False)
    except Exception as exc:  # recorded, never replaced
        row["old_failure"] = repr(exc)
    try:
        row["new"] = run_arm(model, tok, tasks[index], prompts[index],
                             args.q, law, args.mode, args.borrow,
                             args.coverage, True)
    except Exception as exc:  # recorded, never replaced
        row["new_failure"] = repr(exc)
    if "old" in row and "new" in row:
        row["compare"] = first_divergence(row["old"], row["new"])
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline", type=int, required=True)
    parser.add_argument("--indices", default=",".join(map(str, range(30))))
    parser.add_argument("--q", type=int, choices=(8,), default=8)
    parser.add_argument("--mode", default="head", choices=CROSS)
    parser.add_argument("--coverage", default="sparse", choices=("sparse",))
    parser.add_argument("--law", default="1,8")
    parser.add_argument("--borrow", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    import torch
    import transformers
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from .cross_race_llada import load_gpu

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one protected GPU")
    tasks = load_tasks(args.samples)
    if [t["index"] for t in tasks] != list(range(90)):
        raise ValueError("Not the registered frozen90 ordering")
    indices = [int(x) for x in args.indices.split(",")]
    if len(set(indices)) != len(indices) or any(i not in range(90) for i in indices):
        raise ValueError("Invalid index list")
    prompts = {}
    for i in indices:
        ref = json.loads((args.reference / f"{i:02d}-native32.json").read_text())
        if (ref.get("index") != i or not ref.get("prompt_ids")
                or ref.get("generation_cap") != 2048):
            raise ValueError(f"Unregistered native reference for {i}")
        prompts[i] = ref["prompt_ids"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rows").mkdir(exist_ok=True)
    run = dict(
        protocol=PROTOCOL, checkpoint=str(args.checkpoint),
        samples_sha256=hashlib.sha256(args.samples.read_bytes()).hexdigest(),
        reference=str(args.reference), deadline=args.deadline, indices=indices,
        q=args.q, mode=args.mode, coverage=args.coverage, law=args.law,
        borrow=args.borrow, batch=1, visibility_branch="KVHorizonRequest",
        strict_cap_expected=True,
        executor_F_note="F is the self-developed executor's actual forward count; Q is physical rows",
        gpu=torch.cuda.get_device_name(), host=socket.gethostname(),
        torch=str(torch.__version__), transformers=transformers.__version__,
        device_uuid=os.environ.get("CUDA_VISIBLE_DEVICES", ""), started=time.time())
    (args.out / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    tok, model = load_gpu(args, lambda: None)
    deadline = args.deadline - 60
    rows = []
    with torch.inference_mode():
        for i in indices:
            dst = args.out / "rows" / f"{i:02d}.json"
            if dst.exists() and not args.force:
                rows.append(json.loads(dst.read_text()))
                continue
            if time.time() >= deadline:
                (args.out / "stopped.json").write_text(json.dumps(
                    dict(reason="deadline", at=time.time(), completed=len(rows))) + "\n")
                break
            row = run_index(model, tok, tasks, prompts, i, args)
            tmp = dst.with_suffix(".tmp")
            tmp.write_text(json.dumps(row, sort_keys=True) + "\n")
            os.replace(tmp, dst)
            rows.append(row)
            compared = row.get("compare") or {}
            print(f"HORIZON {i:02d} old={bool((row.get('old') or {}).get('correct'))} "
                  f"new={bool((row.get('new') or {}).get('correct'))} "
                  f"div={bool(compared.get('diverged'))} "
                  f"visOnly={bool(compared.get('visibility_only'))} "
                  f"firstOut={compared.get('first_output_tick')} "
                  f"gap={compared.get('horizon_gap_at_first')} "
                  f"oldFail={row.get('old_failure')} newFail={row.get('new_failure')}",
                  flush=True)
    cohorts = {name: summarize_cohort(name, indices, {r["index"]: r for r in rows}, args.q)
               for name, indices in (("original30", tuple(range(30))),
                                     ("new60", tuple(range(30, 90))),
                                     ("all90", tuple(range(90))))}
    summary = dict(protocol=PROTOCOL, q=args.q, mode=args.mode,
                   coverage=args.coverage, borrow=args.borrow,
                   cohorts=cohorts)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (args.out / "summary.csv").open("w", newline="") as handle:
        columns = ("cohort", "requests", "old_correct", "new_correct", "old_N",
                   "new_N", "old_F", "new_F", "old_Q", "new_Q",
                   "old_macro_TPF", "new_macro_TPF", "old_macro_TE",
                   "new_macro_TE", "old_pooled_TPF", "new_pooled_TPF",
                   "old_pooled_TE", "new_pooled_TE", "old_strict_cap",
                   "new_strict_cap", "diverged", "visibility_only",
                   "same_selection_before", "zero_gap_divergence",
                   "first_horizon_tick_min", "first_horizon_tick_median",
                   "horizon_gap_at_horizon_median",
                   "horizon_gap_at_output_median")
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, stats in cohorts.items():
            row = dict(cohort=name, **{k: stats[k] for k in columns if k in stats})
            row.update(first_horizon_tick_min=stats["first_horizon_tick"]["min"],
                       first_horizon_tick_median=stats["first_horizon_tick"]["median"],
                       horizon_gap_at_horizon_median=stats["horizon_gap_at_horizon"]["median"],
                       horizon_gap_at_output_median=stats["horizon_gap_at_output"]["median"])
            writer.writerow(row)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Paired Q64 gate single factor: old successor gate vs new successor gate.

Both arms use ``coverage="head"``, ``q=64``, ``borrow=False`` and
``warm_head_sample=False``; the only difference is the registered gate
(e.g. ``half -> quarter``).  Reuses ``evaluate_fixed`` and ``load_gpu``; no
other agent module is edited.  Records strict cap, the self-developed executor's
actual F, macro/pooled TPF/TE and the first physical-row divergence.  Receipts
stay on the cluster.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import time
from pathlib import Path

from .llada_hfull_single_factor import request_classes

PROTOCOL = "llada-hfull-gate-single-factor-v1"
CROSS = ("off", "eager", "half", "quarter")


def arm_metrics(arm, q=64):
    """Common per-arm metrics: TPF, TE, strict cap and executor actual F."""
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


def run_arm(model, tok, task, prompt, q, law, mode, coverage):
    """Reuse the registered evaluator with the recording request class."""
    from .llada_fixed_family_eval import evaluate_fixed

    request_module, recording, _warm = request_classes()
    original = request_module.LLaDABudgetRequest
    request_module.LLaDABudgetRequest = recording
    try:
        return evaluate_fixed(model, tok, task, prompt, q, law,
                              cross=(mode, False), coverage=coverage)
    finally:
        request_module.LLaDABudgetRequest = original


def compare_pair(old_row, new_row, q=64):
    """Paired gate contrast; the first differing physical rows are evidence."""
    old = old_row or {}
    new = new_row or {}
    old_m = arm_metrics(old, q)
    new_m = arm_metrics(new, q)
    old_trace = old.get("trace") or []
    new_trace = new.get("trace") or []
    first_divergence = None
    for tick, (old_event, new_event) in enumerate(zip(old_trace, new_trace)):
        if (old_event.get("rows") != new_event.get("rows")
                or old_event.get("tail_opened") != new_event.get("tail_opened")):
            first_divergence = dict(
                tick=tick,
                old_rows=len(old_event.get("rows") or []),
                new_rows=len(new_event.get("rows") or []),
                old_tail_opened=bool(old_event.get("tail_opened")),
                new_tail_opened=bool(new_event.get("tail_opened")),
                old_head_mask_at_plan=old_event.get("head_mask_at_plan"),
                new_head_mask_at_plan=new_event.get("head_mask_at_plan"))
            break
    old_pre = old_trace[0].get("pre_state") if old_trace else None
    new_pre = new_trace[0].get("pre_state") if new_trace else None
    return dict(
        old_correct=bool(old.get("correct")), new_correct=bool(new.get("correct")),
        old_N=old_m["N"], new_N=new_m["N"], old_F=old_m["F"], new_F=new_m["F"],
        old_Q=old_m["Q"], new_Q=new_m["Q"], old_TPF=old_m["TPF"], new_TPF=new_m["TPF"],
        old_TE=old_m["TE"], new_TE=new_m["TE"],
        old_executor_F=old_m["executor_F"], new_executor_F=new_m["executor_F"],
        old_max_Q=old_m["max_Q_per_forward"], new_max_Q=new_m["max_Q_per_forward"],
        old_strict_cap=old_m["strict_cap"], new_strict_cap=new_m["strict_cap"],
        pre_state_equal_at_tick0=((old_pre == new_pre)
                                  if old_pre is not None and new_pre is not None else None),
        first_divergence=first_divergence,
        old_failure=old.get("failure"), new_failure=new.get("failure"))


def summarize_cohort(name, indices, rows):
    present = [index for index in indices if index in rows]
    missing = [index for index in indices if index not in rows]
    complete = not missing
    stats = dict(
        requests=len(present), old_correct=0, new_correct=0,
        old_N=0, new_N=0, old_F=0, new_F=0, old_Q=0, new_Q=0,
        old_failures=0, new_failures=0, both_true=0, both_false=0,
        old_true_new_false=0, old_false_new_true=0, delta_N=0, delta_F=0,
        delta_Q=0, pre_state_equal_tick0=0, diverged=0, per_index=[])
    for index in present:
        row = rows[index]
        old = row.get("old") or {}
        new = row.get("new") or {}
        compared = row.get("compare") or {}
        old_correct = bool(old.get("correct"))
        new_correct = bool(new.get("correct"))
        old_fail = "old_failure" in row or bool(old.get("failure"))
        new_fail = "new_failure" in row or bool(new.get("failure"))
        stats["old_correct"] += int(old_correct)
        stats["new_correct"] += int(new_correct)
        stats["old_failures"] += int(old_fail)
        stats["new_failures"] += int(new_fail)
        stats["old_N"] += int(old.get("N", 0))
        stats["new_N"] += int(new.get("N", 0))
        stats["old_F"] += int(old.get("F", 0))
        stats["new_F"] += int(new.get("F", 0))
        stats["old_Q"] += int(old.get("Q", 0))
        stats["new_Q"] += int(new.get("Q", 0))
        stats["delta_N"] += int(new.get("N", 0)) - int(old.get("N", 0))
        stats["delta_F"] += int(new.get("F", 0)) - int(old.get("F", 0))
        stats["delta_Q"] += int(new.get("Q", 0)) - int(old.get("Q", 0))
        stats["both_true"] += int(old_correct and new_correct)
        stats["both_false"] += int(not old_correct and not new_correct)
        stats["old_true_new_false"] += int(old_correct and not new_correct)
        stats["old_false_new_true"] += int(not old_correct and new_correct)
        stats["pre_state_equal_tick0"] += int(compared.get("pre_state_equal_at_tick0") is True)
        stats["diverged"] += int(compared.get("first_divergence") is not None)
        stats["per_index"].append(dict(
            index=index, old_correct=old_correct, new_correct=new_correct,
            old_N=old.get("N"), new_N=new.get("N"), old_F=old.get("F"),
            new_F=new.get("F"), old_Q=old.get("Q"), new_Q=new.get("Q"),
            old_TPF=compared.get("old_TPF"), new_TPF=compared.get("new_TPF"),
            old_TE=compared.get("old_TE"), new_TE=compared.get("new_TE"),
            old_executor_F=compared.get("old_executor_F"),
            new_executor_F=compared.get("new_executor_F"),
            old_max_Q=compared.get("old_max_Q"), new_max_Q=compared.get("new_max_Q"),
            old_strict_cap=compared.get("old_strict_cap"),
            new_strict_cap=compared.get("new_strict_cap"),
            old_failure=bool(old_fail), new_failure=bool(new_fail),
            first_divergence=compared.get("first_divergence")))
    stats["coverage"] = dict(
        expected=len(indices), present=len(present), missing=missing,
        complete=complete, fraction=(len(present) / len(indices)) if indices else 1.0)
    stats["partial"] = not complete
    stats["metrics_scope"] = "all" if complete else "present_only"
    stats["old_pooled_TPF"] = stats["old_N"] / stats["old_F"] if stats["old_F"] else None
    stats["new_pooled_TPF"] = stats["new_N"] / stats["new_F"] if stats["new_F"] else None
    stats["old_pooled_TE"] = stats["old_N"] / stats["old_Q"] if stats["old_Q"] else None
    stats["new_pooled_TE"] = stats["new_N"] / stats["new_Q"] if stats["new_Q"] else None
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
        (row["old_max_Q"] or 0 for row in stats["per_index"]), default=0)
    stats["new_max_Q_per_forward"] = max(
        (row["new_max_Q"] or 0 for row in stats["per_index"]), default=0)
    return stats


def run_index(model, tok, tasks, prompts, index, args):
    from bdlm_serving.methods.t2t.budget_family import Law

    law = Law(*[int(x) for x in args.law.split(",")])
    row = dict(index=index, dataset=tasks[index]["dataset"],
               task_id=tasks[index]["task_id"], seed=tasks[index]["seed"])
    try:
        row["old"] = run_arm(model, tok, tasks[index], prompts[index],
                             args.q, law, args.old_mode, args.coverage)
    except Exception as exc:  # recorded, never replaced
        row["old_failure"] = repr(exc)
    try:
        row["new"] = run_arm(model, tok, tasks[index], prompts[index],
                             args.q, law, args.new_mode, args.coverage)
    except Exception as exc:  # recorded, never replaced
        row["new_failure"] = repr(exc)
    if "old" in row and "new" in row:
        row["compare"] = compare_pair(row["old"], row["new"], args.q)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline", type=int, required=True)
    parser.add_argument("--indices", default="0,10,20")
    parser.add_argument("--q", type=int, default=64)
    parser.add_argument("--coverage", default="head", choices=("head",))
    parser.add_argument("--old-mode", default="half", choices=CROSS)
    parser.add_argument("--new-mode", default="quarter", choices=CROSS)
    parser.add_argument("--law", default="1,8")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.old_mode == args.new_mode:
        raise ValueError("old and new gate must differ")
    if args.q != 64:
        raise ValueError("This single factor is registered at Q64")

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
        q=args.q, coverage=args.coverage, old_mode=args.old_mode,
        new_mode=args.new_mode, law=args.law, batch=1,
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
            print(f"GATE {i:02d} old={bool((row.get('old') or {}).get('correct'))} "
                  f"new={bool((row.get('new') or {}).get('correct'))} "
                  f"oldF={(row.get('old') or {}).get('F')} "
                  f"newF={(row.get('new') or {}).get('F')} "
                  f"div={compared.get('first_divergence')} "
                  f"oldFail={row.get('old_failure')} newFail={row.get('new_failure')}",
                  flush=True)
    cohorts = {name: summarize_cohort(name, indices, {r["index"]: r for r in rows})
               for name, indices in (("original30", tuple(range(30))),
                                     ("new60", tuple(range(30, 90))),
                                     ("all90", tuple(range(90))))}
    summary = dict(protocol=PROTOCOL, old_mode=args.old_mode,
                   new_mode=args.new_mode, cohorts=cohorts)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (args.out / "summary.csv").open("w", newline="") as handle:
        columns = ("cohort", "requests", "old_correct", "new_correct", "old_N",
                   "new_N", "old_F", "new_F", "old_Q", "new_Q", "old_pooled_TPF",
                   "new_pooled_TPF", "old_pooled_TE", "new_pooled_TE",
                   "old_macro_TPF", "new_macro_TPF", "old_macro_TE",
                   "new_macro_TE", "old_executor_F", "new_executor_F",
                   "old_strict_cap", "new_strict_cap", "pre_state_equal_tick0",
                   "diverged")
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, stats in cohorts.items():
            writer.writerow(dict(cohort=name, **{k: stats[k] for k in columns if k in stats}))
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

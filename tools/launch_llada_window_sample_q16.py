"""Q16 active-window paid-warm-MASK single factor, with a technical gate.

Old arm is the measured active-window control (the same pinned geometry as the
Q16 window run).  New arm adds already-paid waiting warm MASK positions on
mixed cold calls via ``llada_window_sample.WindowPaidMask``.  The arm order and
``ORDER`` are frozen in a preregistration before any score is computed.  The
run starts with 0/10/20; if at least one real paid MASK exposure occurs, it
continues in the same output directory to full90 (original30 + new60).  All
failures are recorded and never replaced.  The compact archive uses canonical
pooled TE (sum(N)/sum(Q)).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = "/gs/work/llada-window-sample-q16-sf"
sys.path[:] = [p for p in sys.path if p not in (REPO, "/gs/history-repo", "")]
sys.path.insert(0, REPO)
ANCHORS = (0, 10, 20)
ORDER = (0, 10, 20, *(i for i in range(90) if i not in (0, 10, 20)))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_control(path, index):
    path = Path(path)
    row = json.loads((path / "rows" / f"{index:02d}.json").read_text())
    return row.get("window") or row.get("paid") or {}


def summarize(rows, indices, q=16):
    present = [i for i in indices if i in rows]
    missing = [i for i in indices if i not in rows]
    stats = dict(
        requests=len(present), old_correct=0, new_correct=0,
        old_N=0, new_N=0, old_F=0, new_F=0, old_Q=0, new_Q=0,
        old_failures=0, new_failures=0, exposed=0, first_checks=0,
        first_failures=0, control_replay=0, control_replay_failures=0,
        paid_waiting_calls=0, paid_waiting_rows=0, paid_waiting_admits=0,
        first_ticks=[], per_index=[])
    for index in present:
        row = rows[index]
        old, new = row.get("old") or {}, row.get("paid") or {}
        first = row.get("first")
        exposed = first is not None
        stats["old_correct"] += int(bool(old.get("correct")))
        stats["new_correct"] += int(bool(new.get("correct")))
        stats["old_failures"] += int(bool(old.get("failure")) or "old_failure" in row)
        stats["new_failures"] += int(bool(new.get("failure")) or "new_failure" in row)
        stats["old_N"] += int(old.get("N", 0))
        stats["new_N"] += int(new.get("N", 0))
        stats["old_F"] += int(old.get("F", 0))
        stats["new_F"] += int(new.get("F", 0))
        stats["old_Q"] += int(old.get("Q", 0))
        stats["new_Q"] += int(new.get("Q", 0))
        stats["paid_waiting_calls"] += int(new.get("paid_waiting_calls", 0))
        stats["paid_waiting_rows"] += int(new.get("paid_waiting_rows", 0))
        stats["paid_waiting_admits"] += int(new.get("paid_waiting_admits", 0))
        stats["exposed"] += int(exposed)
        stats["first_checks"] += int(exposed and not row.get("first_failure"))
        stats["first_failures"] += int(bool(row.get("first_failure")))
        stats["control_replay"] += int(bool(row.get("control_replay")))
        stats["control_replay_failures"] += int(row.get("control_replay") is False)
        if exposed:
            stats["first_ticks"].append(first.get("tick"))
        stats["per_index"].append(dict(
            index=index, old_correct=bool(old.get("correct")),
            new_correct=bool(new.get("correct")),
            old_N=old.get("N"), new_N=new.get("N"),
            old_F=old.get("F"), new_F=new.get("F"),
            old_Q=old.get("Q"), new_Q=new.get("Q"),
            old_failure=bool(old.get("failure")) or "old_failure" in row,
            new_failure=bool(new.get("failure")) or "new_failure" in row,
            first_failure=bool(row.get("first_failure")),
            control_replay=row.get("control_replay"),
            exposed=exposed, first_tick=first.get("tick") if exposed else None,
            added=list(first.get("added") or []) if exposed else [],
            paid_waiting_rows=new.get("paid_waiting_rows", 0),
            paid_waiting_admits=new.get("paid_waiting_admits", 0)))
    stats["coverage"] = dict(
        expected=len(indices), present=len(present), missing=missing,
        complete=not missing,
        fraction=(len(present) / len(indices)) if indices else 1.0)
    stats["old_pooled_TE"] = stats["old_N"] / stats["old_Q"] if stats["old_Q"] else None
    stats["new_pooled_TE"] = stats["new_N"] / stats["new_Q"] if stats["new_Q"] else None
    old_rates = [r["old_N"] / r["old_F"] for r in stats["per_index"] if r["old_F"]]
    new_rates = [r["new_N"] / r["new_F"] for r in stats["per_index"] if r["new_F"]]
    stats["old_macro_TPF"] = statistics.mean(old_rates) if old_rates else None
    stats["new_macro_TPF"] = statistics.mean(new_rates) if new_rates else None
    stats["first_tick"] = dict(
        min=min(stats["first_ticks"]) if stats["first_ticks"] else None,
        median=statistics.median(stats["first_ticks"]) if stats["first_ticks"] else None,
        max=max(stats["first_ticks"]) if stats["first_ticks"] else None)
    return stats


def write_summary(out, rows, order, q=16):
    out = Path(out)
    cohorts = {
        "pilot3": summarize(rows, [i for i in order if i in (0, 10, 20)], q),
        "original30": summarize(rows, tuple(range(30)), q),
        "new60": summarize(rows, tuple(range(30, 90)), q),
        "all90": summarize(rows, tuple(range(90)), q),
    }
    result = dict(protocol="llada-window-paid-mask-v1", q=q, order=order,
                  cohorts=cohorts)
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    columns = ("cohort", "requests", "old_correct", "new_correct", "old_N",
               "new_N", "old_F", "new_F", "old_Q", "new_Q", "old_pooled_TE",
               "new_pooled_TE", "old_macro_TPF", "new_macro_TPF",
               "old_failures", "new_failures", "exposed", "first_checks",
               "first_failures", "control_replay", "control_replay_failures",
               "paid_waiting_calls", "paid_waiting_rows", "paid_waiting_admits")
    with (out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for name, stats in cohorts.items():
            writer.writerow(dict(cohort=name, **{k: stats[k] for k in columns if k in stats}))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("reference", "samples", "checkpoint", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--deadline", type=int, required=True)
    parser.add_argument("--limit", type=int, choices=(3, 90), default=3)
    parser.add_argument("--auto-full90", action="store_true")
    parser.add_argument("--control-30", type=Path, default=None)
    parser.add_argument("--control-60", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    import torch
    import transformers
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.methods.t2t.budget_family import Law
    from experiments.t2t.serving_slots.llada_fixed_family_eval import (
        CAP, KEYS, SCORER_SHA, evaluate_fixed, require)
    from experiments.t2t.serving_slots.llada_frontier_seat import classes, compact
    from experiments.t2t.serving_slots.llada_window_sample import first as first_paid
    from experiments.t2t.serving_slots.cross_race_llada import load_gpu

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one protected GPU")
    tasks = load_tasks(args.samples)
    if [t["index"] for t in tasks] != list(range(90)):
        raise ValueError("Not the registered frozen90 ordering")
    order = list(ORDER[:args.limit])
    order_full90 = list(ORDER)
    prompts = {}
    for i in order:
        ref = json.loads((args.reference / f"{i:02d}-native32.json").read_text())
        if (ref.get("index") != i or not ref.get("prompt_ids")
                or ref.get("generation_cap") != CAP):
            raise ValueError(f"Unregistered native reference for {i}")
        prompts[i] = ref["prompt_ids"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rows").mkdir(exist_ok=True)
    prereg = dict(
        protocol="llada-window-paid-mask-v1", factor="paid", q=16,
        arms={"old": "active-window control", "new": "window paid warm MASK"},
        order=order_full90, pilot_order=[0, 10, 20], limit=args.limit,
        auto_full90=bool(args.auto_full90),
        expand_rule="continue to full90 only if pilot exposure > 0",
        checkpoint=str(args.checkpoint),
        samples_sha256=hashlib.sha256(args.samples.read_bytes()).hexdigest(),
        scorer=SCORER_SHA,
        geometry_sha256=sha(Path(classes.__code__.co_filename).with_name(
            "llada_window_geometry.py")),
        sampling_sha256=sha(Path(first_paid.__code__.co_filename)),
        planner_sha256=sha(Path(classes.__code__.co_filename)),
        control_30=str(args.control_30) if args.control_30 else None,
        control_60=str(args.control_60) if args.control_60 else None,
        device=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        created_at=time.time(), deadline=args.deadline)
    (args.out / "preregistration.json").write_text(
        json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    tok, model = load_gpu(args, lambda: None)
    control_cls, paid_cls = classes("paid")
    rows = {}
    for path in sorted((args.out / "rows").glob("*.json")):
        row = json.loads(path.read_text())
        rows[row["index"]] = row
    deadline = args.deadline - 60
    with torch.inference_mode():
        for i in order:
            dst = args.out / "rows" / f"{i:02d}.json"
            if dst.exists() and not args.force:
                rows[i] = json.loads(dst.read_text())
                continue
            if time.time() >= deadline:
                (args.out / "stopped.json").write_text(json.dumps(
                    dict(reason="deadline", at=time.time(), completed=len(rows))) + "\n")
                break
            ref = json.loads((args.reference / f"{i:02d}-native32.json").read_text())
            row = dict(index=i, dataset=tasks[i]["dataset"],
                       task_id=tasks[i]["task_id"], seed=tasks[i]["seed"])
            try:
                arms = [evaluate_fixed(model, tok, tasks[i], ref["prompt_ids"], 16,
                                       Law(), cross=("head", False), request_cls=cls)
                        for cls in (control_cls, paid_cls)]
                old, new = arms[:2]
                require(old["input_sha256"] == new["input_sha256"], "unpaired input")
                for arm in arms:
                    require(0 < arm["max_decoder_Q"] <= 16
                            and arm["Q"] == sum(e["q"] for e in arm["trace"])
                            and all(e["q"] <= 16 and e["layer_physical_q"] == [e["q"]] * 20
                                    for e in arm["trace"]), "invalid physical bill")
                old_c = compact(old)
                row.update(old=old_c, paid=compact(new))
                try:
                    row["first"] = first_paid(old, new)
                except Exception as exc:
                    row["first"] = None
                    row["first_failure"] = repr(exc)
                control_path = args.control_30 if i < 30 else args.control_60
                if control_path is not None:
                    control = load_control(control_path, i)
                    same = all(old_c.get(k) == control.get(k) for k in
                               ("output_digest", "N", "F", "Q", "failure", "correct"))
                    row["control_replay"] = bool(same)
                    if not same:
                        row["control_replay_diff"] = {
                            k: [old_c.get(k), control.get(k)] for k in
                            ("output_digest", "N", "F", "Q", "failure", "correct")
                            if old_c.get(k) != control.get(k)}
            except Exception as exc:
                row["old_failure"] = repr(exc)
            tmp = dst.with_suffix(".tmp")
            tmp.write_text(json.dumps(row, sort_keys=True) + "\n")
            os.replace(tmp, dst)
            rows[i] = row
            first = row.get("first")
            print(f"PAID16 {i:02d} old={bool((row.get('old') or {}).get('correct'))} "
                  f"paid={bool((row.get('paid') or {}).get('correct'))} "
                  f"exposed={bool(first)} added={len((first or {}).get('added') or [])} "
                  f"replay={row.get('control_replay')} "
                  f"fail={row.get('old_failure') or row.get('first_failure')}",
                  flush=True)
    pilot = summarize(rows, [i for i in (0, 10, 20) if i in rows], 16)
    if args.limit == 3 and args.auto_full90 and pilot["exposed"] > 0 and time.time() < deadline:
        rest = [i for i in ORDER if i not in rows]
        # Reuse the loaded model; rows 0/10/20 already exist and are skipped.
        for i in rest:
            dst = args.out / "rows" / f"{i:02d}.json"
            if dst.exists() and not args.force:
                rows[i] = json.loads(dst.read_text())
                continue
            if time.time() >= deadline:
                (args.out / "stopped.json").write_text(json.dumps(
                    dict(reason="deadline", at=time.time(), completed=len(rows))) + "\n")
                break
            ref = json.loads((args.reference / f"{i:02d}-native32.json").read_text())
            row = dict(index=i, dataset=tasks[i]["dataset"],
                       task_id=tasks[i]["task_id"], seed=tasks[i]["seed"])
            try:
                arms = [evaluate_fixed(model, tok, tasks[i], ref["prompt_ids"], 16,
                                       Law(), cross=("head", False), request_cls=cls)
                        for cls in (control_cls, paid_cls)]
                old, new = arms[:2]
                require(old["input_sha256"] == new["input_sha256"], "unpaired input")
                for arm in arms:
                    require(0 < arm["max_decoder_Q"] <= 16
                            and arm["Q"] == sum(e["q"] for e in arm["trace"])
                            and all(e["q"] <= 16 and e["layer_physical_q"] == [e["q"]] * 20
                                    for e in arm["trace"]), "invalid physical bill")
                old_c = compact(old)
                row.update(old=old_c, paid=compact(new))
                try:
                    row["first"] = first_paid(old, new)
                except Exception as exc:
                    row["first"] = None
                    row["first_failure"] = repr(exc)
                control_path = args.control_30 if i < 30 else args.control_60
                if control_path is not None:
                    control = load_control(control_path, i)
                    same = all(old_c.get(k) == control.get(k) for k in
                               ("output_digest", "N", "F", "Q", "failure", "correct"))
                    row["control_replay"] = bool(same)
            except Exception as exc:
                row["old_failure"] = repr(exc)
            tmp = dst.with_suffix(".tmp")
            tmp.write_text(json.dumps(row, sort_keys=True) + "\n")
            os.replace(tmp, dst)
            rows[i] = row
            first = row.get("first")
            print(f"PAID16 {i:02d} old={bool((row.get('old') or {}).get('correct'))} "
                  f"paid={bool((row.get('paid') or {}).get('correct'))} "
                  f"exposed={bool(first)} added={len((first or {}).get('added') or [])} "
                  f"replay={row.get('control_replay')} "
                  f"fail={row.get('old_failure') or row.get('first_failure')}",
                  flush=True)
    write_summary(args.out, rows, order)
    print(json.dumps({k: summarize(rows, v, 16) for k, v in
                      (("pilot3", (0, 10, 20)), ("original30", tuple(range(30))),
                       ("new60", tuple(range(30, 90))), ("all90", tuple(range(90))))},
                     sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

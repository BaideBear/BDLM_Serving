"""Single-factor Q64 full-head half-gate arm: old vs warm-head sample.

The only difference between arms is ``LLaDABudgetRequest(warm_head_sample=...)``.
Both arms reuse ``llada_fixed_family_eval.evaluate_fixed`` and
``cross_race_llada.load_gpu``; no other agent module is edited.  The old arm
records a pre-state digest on every forward, so the first intervention can be
checked for identical pre-state and identical physical Q before trajectories are
allowed to diverge.

Receipts stay on the cluster.  Failures are recorded and never replaced: an
existing per-index receipt is skipped unless ``--force`` is explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path

PROTOCOL = "llada-hfull-warm-head-single-factor-v1"
CROSS = ("off", "eager", "half", "quarter")


def state_digest(req):
    """Hash the exact pre-step state that determines the next plan."""
    h = hashlib.sha256()
    h.update(json.dumps(req.x.detach().cpu().tolist()).encode())
    h.update(f"|{req.head}|{req.end}|{req.origin}|{req.sample_clock}"
             f"|{req.bank.last_end}|{req.status}|{int(req.stopping)}".encode())
    h.update(("|" + ",".join(map(str, sorted(req.dirty)))).encode())
    return h.hexdigest()


def request_classes():
    """Return (module, old recording class, opt-in class) with the base class."""
    from bdlm_serving.methods.t2t import llada_budget_request as request_module

    class RecordingRequest(request_module.LLaDABudgetRequest):
        """Old control: unchanged plan, plus a pre-state digest per record."""

        def _step(self, q):
            self._pre_state = dict(
                x_sha256=state_digest(self), head=self.head, end=self.end,
                origin=self.origin, dirty=sorted(self.dirty),
                last_end=self.bank.last_end, sample_clock=self.sample_clock,
                status=self.status, stopping=bool(self.stopping))
            event = super()._step(q)
            event["pre_state"] = self._pre_state
            return event

    class WarmHeadSampleRequest(RecordingRequest):
        """The single factor: identical request with warm_head_sample=True."""

        def __init__(self, *args, **kwargs):
            kwargs["warm_head_sample"] = True
            super().__init__(*args, **kwargs)

    return request_module, RecordingRequest, WarmHeadSampleRequest


def compare_first(old_row, new_row, q=64):
    """Check the first warm-head intervention against the old control."""
    old_by_tick = {e["tick"]: e for e in old_row.get("trace", [])}
    for event in new_row.get("trace", []):
        if event.get("warm_head_sample", 0) <= 0:
            continue
        old = old_by_tick.get(event["tick"])
        added = list(event.get("warm_head_rows", []))
        checks = {}
        if old is None:
            checks["old_tick_present"] = False
            return dict(intervened=True, passed=False, checks=checks,
                        tick=event["tick"], added=added)
        checks["old_tick_present"] = True
        checks["rows_equal"] = event["rows"] == old["rows"]
        checks["q_equal"] = event["q"] == old["q"] == len(event["rows"])
        checks["q_cap"] = len(event["rows"]) <= q
        checks["dirty_before_equal"] = event["dirty_before"] == old["dirty_before"]
        old_pre = old.get("pre_state")
        checks["pre_state_equal"] = ((event.get("pre_state") == old_pre)
                                     if old_pre is not None else None)
        checks["added_in_rows"] = set(added) <= set(event["rows"])
        checks["sampled_union"] = (sorted(set(old["sampled"]) | set(added))
                                   == event["sampled"])
        checks["added_count"] = event["warm_head_sample"] == len(added)
        checks["no_extra_rows"] = set(event["rows"]) == set(old["rows"])
        checks["no_prefix_dirty"] = not (set(added)
                                         & (set(event["prefix"]) | set(event["dirty"])))
        checks["same_forward"] = (
            event.get("layer_physical_q") == [len(event["rows"])] * 20
            and event["q"] == len(event["rows"])
            and len(event["sampled"]) <= len(event["rows"]))
        passed = all(value is True for value in checks.values() if value is not None)
        return dict(intervened=True, passed=passed, checks=checks,
                    pre_state_verified=old_pre is not None,
                    tick=event["tick"], added=added,
                    old_sampled=old["sampled"], new_sampled=event["sampled"],
                    rows=event["rows"], dirty_before=event["dirty_before"],
                    head_mask_at_plan=event["head_mask_at_plan"])
    return dict(intervened=False, passed=False, checks={})


def summarize(rows):
    out = dict(requests=0, intervened=0, first_checks_passed=0,
               old_failures=0, new_failures=0, old_correct=0, new_correct=0,
               old_N=0, new_N=0, old_F=0, new_F=0, old_Q=0, new_Q=0,
               per_index=[])
    for row in rows:
        out["requests"] += 1
        old, new = row.get("old") or {}, row.get("new") or {}
        out["old_failures"] += int("old_failure" in row)
        out["new_failures"] += int("new_failure" in row)
        out["old_correct"] += int(bool(old.get("correct")))
        out["new_correct"] += int(bool(new.get("correct")))
        out["old_N"] += int(old.get("N", 0))
        out["new_N"] += int(new.get("N", 0))
        out["old_F"] += int(old.get("F", 0))
        out["new_F"] += int(new.get("F", 0))
        out["old_Q"] += int(old.get("Q", 0))
        out["new_Q"] += int(new.get("Q", 0))
        compared = row.get("compare") or {}
        intervened = bool(compared.get("intervened"))
        out["intervened"] += int(intervened)
        out["first_checks_passed"] += int(intervened and bool(compared.get("passed")))
        out["per_index"].append(dict(
            index=row.get("index"), old_correct=bool(old.get("correct")),
            new_correct=bool(new.get("correct")), intervened=intervened,
            first_passed=bool(compared.get("passed")),
            old_failure=row.get("old_failure"), new_failure=row.get("new_failure"),
            old_F=old.get("F"), new_F=new.get("F"),
            old_Q=old.get("Q"), new_Q=new.get("Q"),
            first_tick=compared.get("tick"), added=compared.get("added")))
    return out


def run_arm(model, tok, task, prompt, q, law, mode, coverage, warm_head_sample):
    """Reuse the registered evaluator; swap only the request class."""
    from .llada_fixed_family_eval import evaluate_fixed

    request_module, recording, warm = request_classes()
    cls = warm if warm_head_sample else recording
    original = request_module.LLaDABudgetRequest
    request_module.LLaDABudgetRequest = cls
    try:
        return evaluate_fixed(model, tok, task, prompt, q, law,
                              cross=(mode, False), coverage=coverage)
    finally:
        request_module.LLaDABudgetRequest = original


def run_index(model, tok, tasks, prompts, index, args, old_row=None):
    from bdlm_serving.methods.t2t.budget_family import Law

    law = Law(*[int(x) for x in args.law.split(",")])
    row = dict(index=index, dataset=tasks[index]["dataset"],
               task_id=tasks[index]["task_id"], seed=tasks[index]["seed"])
    if old_row is None:
        try:
            row["old"] = run_arm(model, tok, tasks[index], prompts[index],
                                 args.q, law, args.mode, args.coverage, False)
        except Exception as exc:  # recorded, never replaced
            row["old_failure"] = repr(exc)
    else:
        row["old"] = old_row
    try:
        row["new"] = run_arm(model, tok, tasks[index], prompts[index],
                             args.q, law, args.mode, args.coverage, True)
    except Exception as exc:  # recorded, never replaced
        row["new_failure"] = repr(exc)
    if "old" in row and "new" in row:
        row["compare"] = compare_first(row["old"], row["new"], args.q)
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
    parser.add_argument("--mode", default="half", choices=CROSS)
    parser.add_argument("--coverage", default="head", choices=("head",))
    parser.add_argument("--law", default="1,8")
    parser.add_argument("--old-source", choices=("run", "reuse"), default="run")
    parser.add_argument("--reuse-rows", type=Path, default=None)
    parser.add_argument("--force", action="store_true",
                        help="replace an existing index receipt; default never replaces")
    parser.add_argument("--check-deps", action="store_true",
                        help="print sys.path and math_verify import status, then exit")
    args = parser.parse_args(argv)

    import torch
    import transformers
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from .cross_race_llada import load_gpu

    if args.check_deps:
        import sys
        print("sys.path", sys.path, flush=True)
        import math_verify
        print("math_verify OK", math_verify.__file__, flush=True)
        return 0
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
        old_source=args.old_source, batch=1, warm_head_sample_default=False,
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
            old_row = None
            if args.old_source == "reuse":
                if args.reuse_rows is None:
                    raise ValueError("--reuse-rows is required with --old-source reuse")
                src = args.reuse_rows / f"c64-full-half-{i:02d}.json"
                if not src.exists():
                    raise RuntimeError(f"Missing trusted old control {src}")
                old_row = json.loads(src.read_text())
            row = run_index(model, tok, tasks, prompts, i, args, old_row=old_row)
            tmp = dst.with_suffix(".tmp")
            tmp.write_text(json.dumps(row, sort_keys=True) + "\n")
            os.replace(tmp, dst)
            rows.append(row)
            compared = row.get("compare") or {}
            print(f"SINGLE {i:02d} old={bool((row.get('old') or {}).get('correct'))} "
                  f"new={bool((row.get('new') or {}).get('correct'))} "
                  f"intervened={bool(compared.get('intervened'))} "
                  f"first={bool(compared.get('passed'))} "
                  f"oldF={(row.get('old') or {}).get('F')} "
                  f"newF={(row.get('new') or {}).get('F')} "
                  f"oldFail={row.get('old_failure')} newFail={row.get('new_failure')}",
                  flush=True)
    summary = summarize(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

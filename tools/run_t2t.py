"""Run measured Q policies after qualifying full-width output and work on a frozen suite."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
import os
from pathlib import Path
from time import perf_counter_ns

from bdlm_serving.evaluation.frozen import load_tasks, sha
from bdlm_serving.methods.t2t.admission_gate import read_admission_fit
from bdlm_serving.methods.t2t.context_gate import ADMISSION, CONTEXT
from bdlm_serving.methods.t2t.physical_prefix import PREFIX_MODES
from bdlm_serving.methods.t2t.protocol import Arm, matrix, verify_bridge, verify_work
from bdlm_serving.methods.t2t.slot_trace import parse_offer, slots_for


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=("llada21", "gemma"))
    p.add_argument("--suite", choices=("frozen30", "frozen90"), default="frozen30")
    for key in ("checkpoint", "samples", "output", "reference"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--arm", action="append", help=(
        "Q:fresh:mask:selection; repeatable. Physical prefix: Q:0:Q-K:MODE, "
        f"modes: {', '.join(PREFIX_MODES)}. K=0 only for hard_prefix_zero. "
        "Q <= block; B32 Q64 explicitly aliases Q32 with the same K."))
    p.add_argument("--hazard-weights", type=Path, help="One frozen positional ranker shared by all requests")
    p.add_argument("--context-weights", type=Path, help="Private calibration24 input-context coefficients (B32)")
    p.add_argument("--admission-fit", type=Path, help="Frozen first-approved Q8 fit (fallback: T2T_ADMISSION_FIT)")
    p.add_argument("--exchange-rates", type=Path, help="Frozen old-edit and MASK-admission rates (B32)")
    p.add_argument("--seed-offset", type=int, default=0,
                   help="Frozen30 paired seed repeat; requires matching native references")
    p.add_argument("--slot-control", action="append", help="OFFER=qualified frozen90 fixed policy directory; e.g. 32 or 32r8")
    p.add_argument("--time-requests", action="store_true",
                   help="Store synchronized warm batch-one backend wall time separately from qualification")
    return p.parse_args()


def durable(path, data):
    temp = path.with_suffix(".partial")
    with temp.open("x") as f:
        json.dump(data, f, ensure_ascii=False)
    temp.replace(path)


def slot_controls(arms, args, extra, block):
    traces = [slots_for(a.selection, block, a.budget, a.fresh, a.mask) for a in arms]
    required = set().union(*traces)
    controls = {}
    for spec in getattr(args, "slot_control", None) or ():
        key, folder = spec.split("=", 1)
        choice, folder = parse_offer(key,block), Path(folder)
        q,k,mode = choice.plan(block)
        p = f"t2t_q{q}_f0_m{q-k}_{mode}"
        meta = json.loads((folder/"run.json").read_text())
        if choice.native_full:
            p = "fullwidth"
            ready = args.model == "gemma"
        else:
            ready = (p in meta["policies"] and json.loads((folder/f"complete-{p}.json").read_text())
                     == dict(policy=p, requests=90))
        if (choice in controls or extra["suite"] != "frozen90" or meta["suite"] != "frozen90"
                or meta["model"] != args.model or meta["samples_sha256"] != extra["samples_sha256"]
                or not ready
                or json.loads((folder/"qualified.json").read_text()) !=
                   dict(model=args.model, full_width=block, matched=90)):
            raise ValueError(f"Invalid fixed slot control: {spec}")
        controls[choice] = folder, p
    if required != controls.keys():
        raise ValueError("Require one qualified control per slot in the trace")
    flat = set()
    for t in traces:
        if len(t) == 1:
            flat.add(t[0])
        elif not set(t) <= flat:
            raise ValueError("Reproduce each fixed slot before mixed traces")
    return controls


def check_flat(row, old):
    copy = {**row, "policy": old["policy"], "events": [
        {k:v for k,v in e.items() if k not in ("slot_tick", "slot_offer", "slot_prefix_mode")}
        for e in row["events"]]}
    if old["policy"] == "fullwidth":
        # Only scheduler bookkeeping is new. Compare every native event field,
        # phase count (after relabeling), raw output ID and physical model call.
        extra = ("reserved_extra_edit_positions", "reserved_strict_edit_positions", "hard_strict_edits",
                 "mandatory_fresh", "fresh_selected", "edit_slots", "overflow", "hard_clock",
                 "prefix_empty_verify_plan")
        tick = 0
        for original, e in zip(row["events"], copy["events"], strict=True):
            stage = e["stage"]
            if stage == "native_offer":
                if original["slot_tick"] != tick or original["slot_offer"] != e["q"]:
                    raise ValueError("Invalid native offer clock")
                tick += 1
            elif "slot_tick" in original or stage not in ("verify", "bootstrap"):
                raise ValueError("Unexpected action in fixed native schedule")
            if (e["mandatory_fresh"] != e["fresh_selected"] or e["fresh_selected"] != e["available_fresh"]
                    or e["hard_strict_edits"] != e["edited"] or e["overflow"] or e["edit_slots"]
                    or e["reserved_extra_edit_positions"] or e["reserved_strict_edit_positions"]):
                raise ValueError("Native offer lost a mandatory action or used a sparse quota")
            if stage != "bootstrap":
                e["stage"] = "dense"
            for k in extra:
                e.pop(k, None)
        for col in ("forwards_by_phase", "q_by_phase"):
            counts = dict(copy[col])
            for k in ("verify", "native_offer"):
                if k in counts:
                    counts["dense"] = counts.get("dense", 0) + counts.pop(k)
            copy[col] = counts
    if json.loads(json.dumps(copy)) != old:
        raise ValueError(f"Flat slot trajectory disagrees with fixed control at {row['index']}")


def main(args, prepare=None):
    ll = args.model == "llada21"
    block = 32 if ll else 256
    offset = getattr(args, "seed_offset", 0)
    suite = getattr(args, "suite", "frozen30")
    if suite not in ("frozen30", "frozen90"):
        raise ValueError(f"Unknown frozen suite: {suite}")
    if suite == "frozen90" and offset != 0:
        raise ValueError("Frozen90 requires --seed-offset 0")
    samples_sha256 = sha(args.samples)
    if suite == "frozen90":
        from bdlm_serving.evaluation.frozen90 import load_tasks as load_tasks90
        tasks = load_tasks90(args.samples)
    else:
        tasks = load_tasks(args.samples, seed_offset=offset) if offset else load_tasks(args.samples)
    if sha(args.samples) != samples_sha256:
        raise ValueError("Samples changed while loading")
    arms = [Arm.parse(s, block) for s in args.arm] if args.arm else matrix(args.model)
    if len(set(arms)) != len(arms) or any(
            a.budget >= block and a.selection not in ("hard_noedit", *PREFIX_MODES)
            and not slots_for(a.selection, block, a.budget, a.fresh, a.mask) for a in arms):
        raise ValueError("Require distinct arms; at or above block width use a prefix arm or hard_noedit")
    extra = dict(suite=suite, samples_sha256=samples_sha256)
    admission_fit = None
    fit_path = getattr(args, 'admission_fit', None)
    if any(a.selection in ADMISSION for a in arms):
        fit_path = fit_path or os.environ.get('T2T_ADMISSION_FIT')
        if not fit_path:
            raise ValueError('Admission policies require --admission-fit or T2T_ADMISSION_FIT')
        path = Path(fit_path)
        digest = sha(path)
        admission_fit = read_admission_fit(path)
        if sha(path) != digest:
            raise ValueError('Admission fit changed while loading')
        extra['admission_fit_sha256'] = digest
    elif fit_path is not None:
        raise ValueError('--admission-fit requires an admission arm')
    if any(a.selection.startswith("hard_exchange_") for a in arms):
        from bdlm_serving.methods.t2t.exchange import read_rates
        read_rates(getattr(args, "exchange_rates", None))
        extra["exchange_rates_sha256"] = sha(args.exchange_rates)
    if any(a.selection.startswith("hard_hazard_") for a in arms):
        from bdlm_serving.methods.t2t.hazard import read_weights
        read_weights(getattr(args,"hazard_weights",None))
        extra["hazard_weights_sha256"] = sha(args.hazard_weights)
    with ExitStack() as stack:
        ranker = None
        if any(a.selection in CONTEXT for a in arms):
            from bdlm_serving.methods.t2t.embedding_context import (
                ContextRanker, Projector, input_table, read_context_weights)
            usage = any(a.selection == "hard_committed_usage" for a in arms)
            if usage and any(a.selection in CONTEXT and a.selection != "hard_committed_usage" for a in arms):
                raise ValueError("Cannot mix different context calibrations in one invocation")
            if usage:
                from bdlm_serving.methods.t2t.committed_usage import UsageRanker, read_usage_weights
            w = (read_usage_weights if usage else read_context_weights)(getattr(args, 'context_weights', None))
            extra['context_weights_sha256'] = sha(args.context_weights)
            if admission_fit is not None and extra['context_weights_sha256'] != admission_fit['ranker_sha256']:
                raise ValueError('Admission study requires the frozen gate ranker')
            projector = Projector(stack.enter_context(input_table(args.checkpoint)))
            ranker = (UsageRanker if usage else ContextRanker)(projector.unit, w)
        _run_matrix(args, prepare, tasks, arms, extra, block, offset, ranker, admission_fit)


def _run_matrix(args, prepare, tasks, arms, extra, block, offset, ranker, admission_fit=None):
    from bdlm_serving.methods.t2t.online import load_backend
    controls = slot_controls(arms, args, extra, block)
    args.output.mkdir(exist_ok=False, parents=True)
    durable(args.output / "run.json", dict(model=args.model, batch_size=1, cap=2048,
        block=block, policies=[a.policy for a in arms], arms=[asdict(a) for a in arms],
        seed_offset=offset, **extra))
    run = load_backend(args, prepare, context_ranker=ranker, admission_fit=admission_fit)
    timed = getattr(args, "time_requests", False)
    if timed:
        import torch
        timing = args.output / "timing"
        timing.mkdir()
        durable(timing/"scope.json", dict(model=args.model, suite=extra["suite"],
            samples_sha256=extra["samples_sha256"], batch_size=1, device=torch.cuda.get_device_name(),
            torch=str(torch.__version__), visible_device=os.environ.get("CUDA_VISIBLE_DEVICES"),
            warmup="one unmeasured task 0 for each non-qualification arm with the same seed",
            interval="synchronized backend run: prompt processing, generation, trace and packaging; "
                     "LLaDA includes reading its frozen prompt reference; excludes disk output, scoring, "
                     "queueing, initial model load, qualification and warmup"))
    for arm in [None, *arms]:
        policy = "fullwidth" if arm is None else arm.policy
        saturated = arm is not None and arm.selection == "hard_noedit" and arm.budget >= block
        effective_arm = (Arm(block, 0, block, "hard_noedit") if saturated else
                         arm.prefix_executable(block) if arm is not None else None)
        measure = timed and arm is not None
        if measure:
            warm = run(tasks[0], effective_arm)
            signature = (warm["output_ids"],warm["generation_forwards"],warm["generation_query_rows"])
            del warm
            times = []
        for task in tasks:
            if measure:
                torch.cuda.synchronize()
                start = perf_counter_ns()
            row = run(task, effective_arm)
            if measure:
                torch.cuda.synchronize()
                elapsed = perf_counter_ns() - start
                if task is tasks[0] and signature != (row["output_ids"],row["generation_forwards"],row["generation_query_rows"]):
                    raise ValueError("Timed first request disagrees with warmup")
                times.append(dict(index=task["index"],task_id=task["task_id"],seed=task["seed"],
                                  wall_ns=elapsed,N=row["output_tokens"],F=row["generation_forwards"],
                                  physical_Q=row["generation_physical_padded_rows"]))
            row.update({k: task[k] for k in ("index", "task_id", "dataset", "seed")},
                       model=args.model, policy=policy)
            verify_work(row)
            idx = task["index"]
            if arm is not None and policy.startswith("slottrace_"):
                spec = slots_for(arm.selection, block, arm.budget, arm.fresh, arm.mask)
                if len(spec) == 1:
                    directory, pp = controls[spec[0]]
                    check_flat(row, json.loads((directory / f"{idx:02d}-{pp}.json").read_text()))
            if arm is None or saturated:
                ref = json.loads((args.reference / f"{idx:02d}-native{block}.json").read_text())
                verify_bridge(row, ref)
            if saturated:
                row["single_block_saturation"] = dict(nominal_q=arm.budget, executable_width=block)
            elif effective_arm != arm:
                row["single_block_prefix_capacity"] = dict(nominal_q=arm.budget,
                    executable_width=effective_arm.budget, quota=effective_arm.edit)
            durable(args.output / f"{idx:02d}-{policy}.json", row)
            print(f"{policy} {idx:02d} {row['status']} N={row['output_tokens']} "
                  f"F={row['generation_forwards']} Q={row['generation_query_rows']}", flush=True)
        if arm is None:
            durable(args.output / "qualified.json", dict(model=args.model, full_width=block, matched=len(tasks)))
        else:
            if measure:
                durable(timing/f"{policy}.json",dict(policy=policy,requests=times))
            durable(args.output / f"complete-{policy}.json", dict(policy=policy, requests=len(tasks)))


if __name__ == "__main__":
    main(parse_args())

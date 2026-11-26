"""Run measured Q policies after qualifying full-width output and work on a frozen suite."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
import os
from pathlib import Path

from bdlm_serving.evaluation.frozen import load_tasks, sha
from bdlm_serving.methods.t2t.admission_gate import read_admission_fit
from bdlm_serving.methods.t2t.context_gate import ADMISSION, CONTEXT
from bdlm_serving.methods.t2t.physical_prefix import PREFIX_MODES
from bdlm_serving.methods.t2t.protocol import Arm, matrix, verify_bridge, verify_work


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
    return p.parse_args()


def durable(path, data):
    temp = path.with_suffix(".partial")
    with temp.open("x") as f:
        json.dump(data, f, ensure_ascii=False)
    temp.replace(path)


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
            a.budget >= block and a.selection not in ("hard_noedit", *PREFIX_MODES) for a in arms):
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
    args.output.mkdir(exist_ok=False, parents=True)
    durable(args.output / "run.json", dict(model=args.model, batch_size=1, cap=2048,
        block=block, policies=[a.policy for a in arms], arms=[asdict(a) for a in arms],
        seed_offset=offset, **extra))
    run = load_backend(args, prepare, context_ranker=ranker, admission_fit=admission_fit)
    for arm in [None, *arms]:
        policy = "fullwidth" if arm is None else arm.policy
        saturated = arm is not None and arm.selection == "hard_noedit" and arm.budget >= block
        effective_arm = (Arm(block, 0, block, "hard_noedit") if saturated else
                         arm.prefix_executable(block) if arm is not None else None)
        for task in tasks:
            row = run(task, effective_arm)
            row.update({k: task[k] for k in ("index", "task_id", "dataset", "seed")},
                       model=args.model, policy=policy)
            verify_work(row)
            idx = task["index"]
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
            durable(args.output / f"complete-{policy}.json", dict(policy=policy, requests=len(tasks)))


if __name__ == "__main__":
    main(parse_args())

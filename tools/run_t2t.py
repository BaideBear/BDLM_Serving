"""Run measured Q policies after qualifying full-width output AND work on frozen30."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from bdlm_serving.evaluation.frozen import load_tasks
from bdlm_serving.methods.t2t.protocol import Arm, matrix, verify_bridge, verify_work


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=("llada21", "gemma"))
    for key in ("checkpoint", "samples", "output", "reference"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--arm", action="append", help="Q:fresh:mask:selection; repeatable")
    return p.parse_args()


def durable(path, data):
    temp = path.with_suffix(".partial")
    with temp.open("x") as f:
        json.dump(data, f, ensure_ascii=False)
    temp.replace(path)


def main(args, prepare=None):
    from bdlm_serving.methods.t2t.online import load_backend
    ll = args.model == "llada21"
    block = 32 if ll else 256
    tasks = load_tasks(args.samples)
    arms = [Arm.parse(s, block) for s in args.arm] if args.arm else matrix(args.model)
    if len(set(arms)) != len(arms) or any(a.budget >= block for a in arms):
        raise ValueError("Require distinct sparse arms with budgets smaller than the full-width gate")
    args.output.mkdir(exist_ok=False, parents=True)
    durable(args.output / "run.json", dict(model=args.model, batch_size=1, cap=2048,
        block=block, policies=[a.policy for a in arms], arms=[asdict(a) for a in arms]))
    run = load_backend(args, prepare)
    for arm in [None, *arms]:
        policy = "fullwidth" if arm is None else arm.policy
        for task in tasks:
            row = run(task, arm)
            row.update({k: task[k] for k in ("index", "task_id", "dataset", "seed")},
                       model=args.model, policy=policy)
            verify_work(row)
            idx = task["index"]
            if arm is None:
                ref = json.loads((args.reference / f"{idx:02d}-native{block}.json").read_text())
                verify_bridge(row, ref)
            durable(args.output / f"{idx:02d}-{policy}.json", row)
            print(f"{policy} {idx:02d} {row['status']} N={row['output_tokens']} "
                  f"F={row['generation_forwards']} Q={row['generation_query_rows']}", flush=True)
        if arm is None:
            durable(args.output / "qualified.json", dict(model=args.model, full_width=block, matched=30))
        else:
            durable(args.output / f"complete-{policy}.json", dict(policy=policy, requests=30))


if __name__ == "__main__":
    main(parse_args())

"""CPU selector timing and same-state equivalence, not a serving benchmark."""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
from statistics import median
import subprocess
from time import perf_counter_ns

from bdlm_serving.evaluation.frozen import MANIFEST
from bdlm_serving.methods.t2t.frontier import select_hard


PARENT = "f71c3120ddbd297cc90fcbe8e2c4aca3b35b673f"


def states(folder, model):
    width = 32 if model == "llada21" else 256
    for task in json.loads(MANIFEST.read_text())["samples"]:
        with gzip.open(folder / f"{task['index']:02d}.trace.jsonl.gz", "rt") as f:
            steps = map(json.loads, f)
            meta = next(steps)
            assert all(meta[k] == task[k] for k in ("index","dataset","task_id","seed"))
            if width == 32:
                info = next(steps)
                prompt, mask = info["prompt_length"], info["mask_id"]
            canvas = None
            for step in steps:
                b = step["block"]
                if b != canvas:
                    canvas, tick = b, 1
                    eligible = range(max(0, prompt-b) if width == 32 else 0, width)
                    prev = [x != mask for x in step["before"]] if width == 32 else [False]*width
                    write = [-1]*width
                else:
                    tick += 1
                    yield (eligible, prev, fresh, write.copy(), [tick-1]*width, tick), strict, task["seed"]
                after = step["accepted"]
                changed = {i for i in eligible if prev[i] != after[i] or
                           (after[i] and step["before"][i] != step["after"][i])}
                fresh = {i for i in changed if after[i]}
                strict = {i for i in fresh if prev[i]}
                for i in changed:
                    write[i] = tick
                prev = after


def measure(cases, old, q, mode):
    times = [[], []]
    for j, (args, strict, seed) in enumerate(cases):
        def invoke(fn):
            return fn(*args, q, max(1,q//8) if mode != "hard_noedit" else 0, mode,
                      seed, previous_strict=strict)
        outputs = []
        for k in ((0,1) if j % 2 == 0 else (1,0)):
            fn = (old, select_hard)[k]
            start = perf_counter_ns()
            selected = invoke(fn)
            times[k].append((perf_counter_ns()-start) / 1000)
            outputs.append(selected)
        if outputs[0] != outputs[1]:
            raise ValueError(f"{mode} changed rows at state {j}")
    result = dict(policy=mode, q=q, equal_states=len(cases))
    for name, values in zip(("before", "after"), times, strict=True):
        result[name+"_median_us"] = round(median(values),2)
        result[name+"_p95_us"] = round(sorted(values)[(len(values)*95-1)//100],2)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    root = Path(__file__).resolve().parents[1]
    src = subprocess.check_output(["git","-C",str(root),"show",
                                  PARENT+":bdlm_serving/methods/t2t/frontier.py"])
    ns = dict(__name__="bdlm_serving.methods.t2t.previous_frontier",
              __package__="bdlm_serving.methods.t2t")
    exec(compile(src, "<archived allocator>", "exec"), ns)
    rows = []
    for model, budgets in (("llada21", (4,16)), ("gemma", (32,64))):
        cases = list(states(args.native_root/model, model))
        modes = ["hard_noedit","hard_frontier","hard_slack","hard_trigger","hard_slack_placebo",
                 "hard_trigger_placebo"] + (["hard_slack_geometry","hard_geometry"] if model=="llada21" else [])
        for q in budgets:
            for mode in modes:
                rows.append(dict(model=model, **measure(cases, ns["select_hard"], q, mode)))
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

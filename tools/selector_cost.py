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


def hazard_cost(native_root, weights_path=None, exchange_rates=None, q=16):
    from bdlm_serving.methods.t2t.hazard import choose, read_weights
    from bdlm_serving.methods.t2t.exchange import MODES, choose as exchange, read_rates
    weights = read_weights(weights_path) if weights_path else None
    rates = read_rates(exchange_rates) if exchange_rates else None
    modes = ("hard_noedit", *MODES) if rates else ("hard_noedit", "hard_hazard_fixed", "hard_hazard_gate")
    cases = [(args, prev, seed) for args, prev, seed in states(native_root/"llada21", "llada21")
             if any(not args[1][i] for i in args[0])]
    def invoke(mode, args, prev, seed):
        valid, resolved, fresh, writes, _, clock = args
        if mode == "hard_noedit":
            return select_hard(*args, q, 0, mode, seed, previous_strict=prev)
        if rates:
            picked, reserved, _ = exchange(valid, resolved, fresh, prev, writes, clock, q,
                                           max(1,q//8), rates, MODES[mode])
            return picked, reserved
        return choose(valid, resolved, fresh, writes, prev, clock, 16, 2, weights,
                      .05 if mode.endswith("gate") else 0.)
    timings = {mode: [] for mode in modes}
    for j, (args, prev, seed) in enumerate(cases):
        for mode in modes:
            invoke(mode, args, prev, seed)
        for rep in range(3):
            shift = (j + rep) % 3
            for mode in modes[shift:] + modes[:shift]:
                t0 = perf_counter_ns()
                picked, reserved = invoke(mode, args, prev, seed)
                elapsed = perf_counter_ns() - t0
                if not args[2] <= set(picked) or len(picked) != min(len(args[0]), max(q,len(args[2]))):
                    raise ValueError("physical seat or mandatory refresh mismatch")
                timings[mode].append(elapsed / 1000)
    return [dict(policy=mode, nominal_q=q, states=len(cases), calls=len(ts),
                 median_us=round(median(ts), 2), p95_us=round(sorted(ts)[(95*len(ts)-1)//100], 2))
            for mode, ts in timings.items()]


def age_draw_cost(run):
    """Same predecoded pool and clocks; time just the additional candidate draw."""
    from random import Random
    from bdlm_serving.methods.t2t.exchange import PLACEBOS, oldest_half
    meta = json.loads((run/"run.json").read_text())
    tasks = json.loads(MANIFEST.read_text())["samples"]
    if json.loads((run/"qualified.json").read_text()) != dict(model="llada21",full_width=32,matched=30):
        raise ValueError("Unqualified placebo source")
    for arm,policy in zip(meta["arms"],meta["policies"],strict=True):
        salt = PLACEBOS[arm["selection"]]
        if json.loads((run/f"complete-{policy}.json").read_text()) != dict(policy=policy,requests=30):
            raise ValueError("Incomplete placebo source")
        cases = []
        for task in tasks:
            row = json.loads((run/f"{task['index']:02d}-{policy}.json").read_text())
            assert all(row[k] == task[k] for k in ("index","dataset","task_id","seed"))
            for e in row["events"]:
                if e["stage"] == "bootstrap":
                    block, clocks = e["block"], [-1]*32
                if "placebo_exchange" in e:
                    a = e["placebo_exchange"]
                    key = f"exchange-placebo:{salt}:{row['seed']}:{block}:{e['hard_clock']}"
                    draw = Random(key).sample(a["pool"],1)
                    assert set(draw) == set(a["executed"])-set(a["base"])
                    cases.append((a["pool"],clocks.copy(),key))
                if e["stage"] in ("bootstrap","selected"):
                    for i in e["absolute_positions"]:
                        clocks[i-32*block] = e["hard_clock"]
        timings = [[],[]]
        def invoke(mode,pool,clocks,key):
            return Random(key).sample(oldest_half(pool,clocks,key) if mode else pool,1)
        for j,(pool,clocks,key) in enumerate(cases):
            for mode in (0,1):
                invoke(mode,pool,clocks,key)
            for rep in range(4):
                for mode in ((1,0) if (j+rep)%2 else (0,1)):
                    start = perf_counter_ns()
                    invoke(mode,pool,clocks,key)
                    timings[mode].append((perf_counter_ns()-start)/1000.)
        for mode,ts in enumerate(timings):
            yield dict(nominal_q=arm["budget"],salt=salt,mode=("uniform","oldest_half")[mode],
                       swap_states=len(cases),calls=len(ts),median_us=round(median(ts),2),
                       p95_us=round(sorted(ts)[(95*len(ts)-1)//100],2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    choice = p.add_mutually_exclusive_group()
    choice.add_argument("--age-run", type=Path, action="append", help="time candidate draws on a qualified uniform run")
    choice.add_argument("--exchange-rates", type=Path, help="Time physical exchanges at Q4/8/16")
    choice.add_argument("--hazard-weights", type=Path,
                   help="Time Q16 risk ranking on still-masked post-bootstrap LLaDA states")
    args = p.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    if args.age_run:
        from tools.t2t_results import table
        table(args.out,[r for run in args.age_run for r in age_draw_cost(run)])
        return
    if not args.native_root:
        p.error("Provide --native-root or --age-run")
    if args.hazard_weights or args.exchange_rates:
        rows = [r for q in ((4,8,16) if args.exchange_rates else (16,)) for r in
                hazard_cost(args.native_root, args.hazard_weights, args.exchange_rates, q)]
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        return
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

import argparse
import csv
import json
from pathlib import Path

from .evaluation.frozen import ROOT, load_tasks, restore, sha

STUDIES = ("baselines", "flowblock-prefix", "optimus-masked-obs", "llada20-frozen30",
           "llada20-cached-b32")


def score_run(folder, samples):
    from .evaluation.scoring import score
    tasks = load_tasks(samples)
    metadata = json.loads((folder / "run.json").read_text())
    metrics = ["model", "policy", "index", "dataset", "task_id", "correct", "status",
               "output_tokens", "generation_forwards", "generation_query_rows",
               "generation_physical_padded_rows", "max_q", "tpf", "token_efficiency"]
    rows = []
    for policy in metadata["policies"]:
        for task in tasks:
            r = json.loads((folder / f"{task['index']:02d}-{policy}.json").read_text())
            if any(r[k] != task[k] for k in ("index", "dataset", "task_id", "seed")) or r["policy"] != policy:
                raise ValueError("Request identity mismatch")
            r["correct"] = r["status"] == "success" and score(r["output_text"], task)["correct"]
            rows.append({k: r[k] for k in metrics})
    with (folder / "scores.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, metrics)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Scored {len(rows)} requests: {folder / 'scores.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Frozen BDLM experiments")
    actions = parser.add_subparsers(dest="action", required=True)
    r = actions.add_parser("restore", help="restore the frozen sample file from three source files")
    r.add_argument("--raw-dir", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    g = actions.add_parser("run", help="run one model and experiment with a trusted local checkpoint")
    g.add_argument("--experiment", required=True, choices=STUDIES)
    g.add_argument("--model", required=True, choices=("llada20","llada21","sdar"))
    g.add_argument("--checkpoint", type=Path, required=True)
    g.add_argument("--samples", type=Path, required=True)
    g.add_argument("--output", type=Path, required=True)
    g.add_argument("--batch-size", type=int, help="defaults to the experiment's archived batch size")
    g.add_argument("--block-size", type=int, help="FlowBlock block length in tokens; independent of W")
    dg = actions.add_parser("diffusiongemma-native", help="run native DiffusionGemma on frozen30")
    for name in ("checkpoint", "samples", "output"):
        dg.add_argument("--" + name, type=Path, required=True)
    wave = actions.add_parser("diffusiongemma-wave", help="run FlowBlock and B3 on DiffusionGemma")
    for name in ("checkpoint", "samples", "output"):
        wave.add_argument("--" + name, type=Path, required=True)
    wave.add_argument("--config", type=Path, default=ROOT/"archive/diffusiongemma-wave/config.json")
    wave.add_argument("--block-size", type=int, help="override config; must match checkpoint canvas_length")
    wave.add_argument("--flowblock-windows", type=int, nargs="+", help="fixed window(s), in blocks")
    wave.add_argument("--suite", choices=("frozen30", "frozen90"), default="frozen30")
    wave.add_argument("--flow-only", action="store_true", help="omit the separate B3 arm")
    wave.add_argument("--reference30-root", type=Path, help="required for exact original30 reproduction on frozen90")
    wave.add_argument("--native-reference", type=Path, help="require W=1 output IDs and forwards to match")
    s = actions.add_parser("score", help="score a completed run; requires the Linux sandbox for HumanEval")
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--samples", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "restore":
        restore(args.raw_dir, args.output)
        print(f"Restored and verified: {args.output}")
    elif args.action == "score":
        score_run(args.output, args.samples)
    elif args.action == "diffusiongemma-native":
        from .runtime.diffusiongemma import run_native
        run_native(args)
    elif args.action == "diffusiongemma-wave":
        from .runtime.diffusiongemma_wave import run_wave
        run_wave(args)
    else:
        from .generation import run
        if (args.model == "llada20") != (args.experiment in ("llada20-frozen30", "llada20-cached-b32")):
            parser.error("Use llada20 with a llada20 experiment")
        if args.batch_size is not None and args.batch_size < 1:
            parser.error("--batch-size must be positive")
        tasks = load_tasks(args.samples)
        plan = json.loads((ROOT / "archive" / args.experiment / "config.json").read_text())
        if args.block_size is not None:
            if args.block_size < 1 or "flowblock_block_size" not in plan:
                parser.error("--block-size requires a positive value and a FlowBlock experiment")
            plan["flowblock_block_size"] = args.block_size
        for entry in json.loads((ROOT / "benchmarks/checkpoints" / (args.model+".json")).read_text()):
            file = args.checkpoint / entry["name"]
            if (file.stat().st_size != entry["bytes"] or
                    ("sha256" in entry and sha(file) != entry["sha256"])):
                parser.error(f"checkpoint mismatch: {file.name}")
        args.output.mkdir(exist_ok=False, parents=True)
        (args.output / "run.json").write_text(json.dumps(
            dict(plan, experiment=args.experiment, model=args.model,
                 batch_size=args.batch_size or plan["batch_size"]), indent=2))
        run(args, tasks, plan)


if __name__ == "__main__":
    main()

"""Freeze both parameter sets, then evaluate one paired shard with a single checkpoint load."""
import argparse
from contextlib import redirect_stdout
from fractions import Fraction
from hashlib import sha1, sha256
from importlib import import_module
import json
import os
from pathlib import Path
import subprocess
import sys

from bdlm_serving.evaluation.frozen import ROOT
from bdlm_serving.evaluation.frozen90 import MANIFEST, load_tasks
from bdlm_serving.methods.t2t.unified_rows import Params
from tools.frozen90_six_shards import BOARDS, OFFERS, split


DEVICES = dict(zip(BOARDS, (
    "GPU-ceadb3cd-e63f-1446-a2d5-a291144f8efb", "GPU-d3bdc2bc-d19d-e98f-ac79-ab9c6e887e0e",
    "GPU-d3c3fc9c-b15d-9776-9094-7d3b85b705c1", "GPU-861c9b5d-1e73-2550-f832-b73988bc64ff",
    "GPU-1d5b27ea-c99d-15a4-af2b-fd82c414a0c5", "GPU-94c61e79-d80b-cae2-d85f-6bd2a2c99a1c"),
    strict=True))
WORKERS = {"gemma": "shared_gemma_worker", "llada21": "shared_llada_worker"}
INPUTS = ("bdlm_serving", "experiments/t2t/serving_slots", "tools/frozen90_six_shards.py",
          "benchmarks/checkpoints", "benchmarks/frozen30/manifest.json", "benchmarks/frozen90/manifest.json")
KEYS = ("status", "correct", "N", "F", "Q", "decoder_F", "encoder_F")
FORMAT = "shared-fixed-shard-v1"


def identity(registration):
    data = json.dumps(registration, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(data.encode()).hexdigest()


def param_values(values):
    p = Params(*(Fraction(s) for s in values))
    return dict(prefix_share=str(p.prefix_share), open_at=str(p.open_at))


def registration(args):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    entries = subprocess.check_output(["git", "ls-tree", "-r", commit, "--", *INPUTS], cwd=ROOT, text=True)
    blobs = {}
    for line in entries.splitlines():
        meta, path = line.split("\t", 1)
        if path.endswith(".py") or (path.startswith("benchmarks/") and path.endswith(".json")):
            blobs[path] = meta.split()[2]
    for name in (*WORKERS.values(), "shared_fixed_shard"):
        if f"experiments/t2t/serving_slots/{name}.py" not in blobs:
            raise ValueError(f"Commit the worker before registering: {name}")
    return dict(format=FORMAT, commit=commit, blobs=blobs, cap=2048, stage=args.stage,
                params={key: param_values(getattr(args, key)) for key in OFFERS})


def check_sources(reg):
    if reg["format"] != FORMAT or reg["cap"] != 2048 or set(reg["params"]) != set(OFFERS):
        raise ValueError("Unknown fixed-request registration")
    for path, expected in reg["blobs"].items():
        p = (ROOT / path).resolve()
        if not p.is_relative_to(ROOT.resolve()):
            raise ValueError(f"Path outside registered tree: {path}")
        raw = p.read_bytes()
        if sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() != expected:
            raise ValueError(f"Source or benchmark changed: {path}")


def shard_cases(samples, board, stage):
    shard = split(samples)[board]
    if stage == "formal90":
        return shard
    if stage == "screen18":
        # One preregistered case from each dataset per board, no score-dependent selection.
        return sorted((next(r for r in shard if r["dataset"] == d)
                       for d in sorted({r["dataset"] for r in shard})), key=lambda r: r["index"])
    raise ValueError("Unexpected cohort")


def check_board(board):
    uuid = DEVICES[board]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise RuntimeError(f"Expected only the registered physical GPU for {board}")
    info = subprocess.check_output(["nvidia-smi", "-i", uuid, "--query-gpu=uuid",
                                    "--format=csv,noheader"], text=True).strip()
    apps = subprocess.check_output(["nvidia-smi", "-i", uuid, "--query-compute-apps=pid",
                                    "--format=csv,noheader,nounits"], text=True).split()
    if info != uuid or apps:
        raise RuntimeError("Failed exclusive board preflight; holder handoff belongs to launcher")


def run(args):
    reg = json.loads(args.registration.read_bytes())
    version = identity(reg)
    if args.version != version:
        raise ValueError("Registration digest does not match the requested version")
    check_sources(reg)
    tasks = load_tasks(args.samples)
    samples = json.loads(MANIFEST.read_bytes())["samples"]
    shard = shard_cases(samples, args.board, reg["stage"])
    offers = args.offers or OFFERS[args.model]
    if len(set(offers)) != len(offers) or any(q not in OFFERS[args.model] for q in offers):
        raise ValueError("Repeated or unregistered offer")
    params = Params(**{k: Fraction(v) for k, v in reg["params"][args.model].items()})
    print(f"Validated {args.board}: {len(shard)} cases x {tuple(offers)}, {reg['stage']}, {version}",
          file=sys.stderr, flush=True)
    if args.preflight:
        return
    check_board(args.board)
    with redirect_stdout(sys.stderr):
        worker = import_module("." + WORKERS[args.model], __package__)
        handle = worker.load(args.checkpoint)
    # Fixed offer from start to finish. Interleave offer order between requests.
    for j, row in enumerate(shard):
        k = j % len(offers)
        for q in (*offers[k:], *offers[:k]):
            with redirect_stdout(sys.stderr):
                result = worker.evaluate(handle, tasks[row["index"]], q, params)
            if result["status"] not in ("complete", "valid_algorithm_failure"):
                raise RuntimeError("Unqualified result; do not score this request")
            record = dict(**{k: row[k] for k in ("index", "seed", "task_id", "dataset", "prompt_sha256")},
                          board=args.board, model=args.model, offer=q, version=version, stage=reg["stage"],
                          **{k: result[k] for k in KEYS})
            for key in ("roles", "prefill_Q", "encoder_Q", "decoder_Q", "max_decoder_Q",
                        "failure_reason", "failure", "output_ids_sha256", "input_sha256"):
                if key in result:
                    record[key] = result[key]
            print(json.dumps(record, separators=(",", ":"), allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmds = parser.add_subparsers(dest="cmd", required=True)
    freeze = cmds.add_parser("register")
    freeze.add_argument("--stage", choices=["screen18", "formal90"], required=True)
    for model in OFFERS:
        freeze.add_argument("--" + model, nargs=2, required=True, metavar=("PREFIX_SHARE", "OPEN_AT"))
    work = cmds.add_parser("run")
    for key in ("checkpoint", "samples", "registration"):
        work.add_argument("--" + key, type=Path, required=True)
    work.add_argument("--version", required=True)
    work.add_argument("--model", choices=OFFERS, required=True)
    work.add_argument("--board", choices=BOARDS, required=True)
    work.add_argument("--offers", type=int, nargs="+")
    work.add_argument("--preflight", action="store_true", help="Only check source, dataset and shard; never load or use CUDA")
    args = parser.parse_args()
    if args.cmd == "register":
        reg = registration(args)
        print(identity(reg), file=sys.stderr)
        print(json.dumps(reg, indent=2, sort_keys=True))
    else:
        run(args)


if __name__ == "__main__":
    main()

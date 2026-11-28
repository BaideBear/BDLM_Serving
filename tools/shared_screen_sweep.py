"""Paired hyperparameter screen on the same frozen workers; never produces formal90 receipts."""
import argparse
from contextlib import redirect_stdout
from fractions import Fraction
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import sys

from experiments.t2t.serving_slots.shared_fixed_shard import (
    DEVICES, MANIFEST, OFFERS, WORKERS, check_board, check_sources, identity, load_tasks, shard_cases,
)
from bdlm_serving.methods.t2t.unified_rows import Params


CHOICES = ("0", "1/8", "1/4", "1/2")
MODELS = tuple(OFFERS)


def build(base, axis, llada_alpha=None, gemma_alpha=None):
    if base["stage"] != "formal90" or base["cap"] != 2048:
        raise ValueError("expected the frozen formal90 source registration")
    if axis == "prefix" and (llada_alpha is not None or gemma_alpha is not None):
        raise ValueError("a prefix sweep must not override either prefix share")
    if axis == "open" and any(v not in CHOICES for v in (llada_alpha, gemma_alpha)):
        raise ValueError("freeze both independently selected prefix shares before the open sweep")
    if axis not in ("open", "prefix"):
        raise ValueError("unknown sweep")
    arms = {m: {dose: dict(prefix_share=dose if axis == "prefix" else alpha,
                           open_at="1/4" if axis == "prefix" else dose)
                for dose in CHOICES} for m, alpha in zip(MODELS, (llada_alpha, gemma_alpha), strict=True)}
    return dict(format="paired-parameter-screen-v1", source=identity(base), axis=axis, arms=arms,
                stage="screen18", cap=2048, wrapper=sha256(Path(__file__).read_bytes()).hexdigest())


def order(shard, offers):
    for j, row in enumerate(shard):
        for o in range(len(offers)):
            q = offers[(j+o) % len(offers)]
            for i in range(len(CHOICES)):
                yield row, q, CHOICES[(i+j+o+row["index"]) % len(CHOICES)]


def run(a, reg):
    panel = json.loads(a.panel.read_bytes())
    read = lambda m: panel["arms"][m]["0"]["prefix_share"] if panel["axis"] == "open" else None
    if panel != build(reg, panel["axis"], read("llada21"), read("gemma")) or identity(panel) != a.version:
        raise ValueError("screen registration, source, wrapper, or arms do not match")
    check_sources(reg)
    tasks = load_tasks(a.samples)
    shard = shard_cases(json.loads(MANIFEST.read_bytes())["samples"], a.board, "screen18")
    if a.preflight:
        print(f"Validated {a.board}/{a.model}, {len(shard)} x 5 x 4: {a.version}", file=sys.stderr)
        return
    check_board(a.board)
    with redirect_stdout(sys.stderr):
        worker = import_module("experiments.t2t.serving_slots." + WORKERS[a.model])
        handle = worker.load(a.checkpoint)
    for row, q, dose in order(shard, OFFERS[a.model]):
        configured = panel["arms"][a.model][dose]
        with redirect_stdout(sys.stderr):
            result = worker.evaluate(handle, tasks[row["index"]], q, Params(**{k: Fraction(v) for k,v in configured.items()}))
        if result["status"] not in ("complete", "valid_algorithm_failure"):
            raise ValueError("unqualified worker status")
        keys = ("status", "correct", "N", "F", "Q", "decoder_F", "encoder_F", "decoder_Q",
                "encoder_Q", "max_decoder_Q", "roles")
        record = {k: result[k] for k in keys}
        record.update({k: row[k] for k in ("index", "seed", "dataset", "task_id", "prompt_sha256")})
        record.update(stage="screen18", version=a.version, source=panel["source"], board=a.board,
                      model=a.model, offer=q, axis=panel["axis"], dose=dose, params=configured)
        print(json.dumps(record, allow_nan=False, separators=(",", ":")), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    freeze = sub.add_parser("register")
    freeze.add_argument("--axis", choices=("prefix", "open"), required=True)
    freeze.add_argument("--llada-alpha", choices=CHOICES)
    freeze.add_argument("--gemma-alpha", choices=CHOICES)
    work = sub.add_parser("run")
    for key in ("samples", "panel", "checkpoint"):
        work.add_argument("--"+key, type=Path, required=True)
    for s in (work, freeze):
        s.add_argument("--base", type=Path, required=True)
    work.add_argument("--version", required=True)
    work.add_argument("--board", choices=DEVICES, required=True)
    work.add_argument("--model", choices=MODELS, required=True)
    work.add_argument("--preflight", action="store_true")
    a = p.parse_args()
    reg = json.loads(a.base.read_bytes())
    if a.cmd == "register":
        panel = build(reg, a.axis, a.llada_alpha, a.gemma_alpha)
        print(json.dumps(panel, indent=2, sort_keys=True))
        print(identity(panel), file=sys.stderr)
    else:
        run(a, reg)


if __name__ == "__main__":
    main()

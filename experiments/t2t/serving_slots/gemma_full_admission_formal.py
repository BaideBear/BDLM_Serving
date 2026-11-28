"""Run a registered fixed-Q64 formal shard after A3 qualification."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from .gemma_full_admission_qualify import (CORE, GPU, RUNTIME, digest, guard,
                                           read, run_generic, sha_file,
                                           source_manifest, write)
from .gemma_full_admission import open_full_admission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qualification", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--range", default="0:45")
    ap.add_argument("--deadline", type=int, required=True)
    ap.add_argument("--samples", type=Path, default=Path("/gs/code/samples.json"))
    ap.add_argument("--weights", type=Path, default=Path("/gs/models/diffusiongemma-cd40195"))
    ap.add_argument("--source-manifest", type=Path, default=Path("source.input.sha256"))
    ap.add_argument("--commit", required=True)
    args = ap.parse_args()
    lo, hi = map(int, args.range.split(":", 1))
    indices = list(range(lo, hi))
    guard("before")
    import torch
    import transformers
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from . import gemma_full_admission_fixed90_h0 as h0
    seal = read(args.qualification)
    if seal["schema"] != "gemma-full-admission-fixed-q64-a3-torch26-v1" or seal["offer"] != 64:
        raise RuntimeError("Formal shard does not match the qualification seal")
    manifest_sha, files = source_manifest(args.source_manifest, CORE, Path(__file__))
    if manifest_sha != seal["qualification_source_sha256"]:
        raise RuntimeError("Formal source manifest changed after qualification")
    if sha_file(CORE) != seal["executor_sha256"]:
        raise RuntimeError("Formal executor changed after qualification")
    tasks = load_tasks(args.samples)
    guard("inputs")
    proc = AutoProcessor.from_pretrained(args.weights, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.weights, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", experts_implementation="grouped_mm",
        device_map={"": "cuda:0"}).eval()
    guard("loaded")
    host = subprocess.check_output(["hostname"], text=True).strip()
    registration = {"schema": "gemma-full-admission-fixed-q64-a3-torch26-formal-v1",
                    "offer": 64, "registered": indices,
                    "qualification": str(args.qualification),
                    "qualification_sha256": sha_file(args.qualification),
                    "qualification_source_sha256": manifest_sha,
                    "executor_sha256": sha_file(CORE), "commit": args.commit,
                    "runtime": RUNTIME, "gpu": GPU, "pid": os.getpid(), "host": host}
    args.out.mkdir(parents=False, exist_ok=False)
    write(args.out / "registration.json", registration)
    rows = []
    for index in indices:
        if time.time() >= args.deadline:
            raise RuntimeError("Deadline reached before completing registered formal shard")
        guard("request")
        qualification_case = args.qualification.parent / f"case-{index:02}.json"
        if qualification_case.is_file():
            raw = read(qualification_case)
            source = "qualification"
        else:
            task = next(t for t in tasks if t["index"] == index)
            raw = run_generic(model, proc, task, None, open_full_admission, 64)
            source = "formal"
        raw["qualification_origin"] = seal["qualification_origin"]
        saved_q = h0.Q
        h0.Q = 64
        try:
            row = h0.summarize(raw)
        finally:
            h0.Q = saved_q
        path = args.out / f"case-{index:02}.json"
        write(path, raw)
        receipt = dict(row, index=index, source=source, raw_sha256=sha_file(path))
        write(args.out / f"receipt-{index:02}.json", receipt)
        rows.append(receipt)
        print("FORMAL " + json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False), flush=True)
    complete = {"schema": registration["schema"], "offer": 64, "registered": indices,
                "received": len(rows), "N": sum(x["N"] or 0 for x in rows),
                "F": sum(x["F"] for x in rows), "Q": sum(x["Q"] for x in rows),
                "correct": sum(int(bool(x["correct"])) for x in rows),
                "peak_decoder_q": max(x["peak_decoder_q"] for x in rows),
                "failures": [x["index"] for x in rows if x["failure"]]}
    write(args.out / "complete.json", complete)
    print("FORMAL_SUMMARY " + json.dumps(complete, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False), flush=True)
    guard("release")


if __name__ == "__main__":
    main()

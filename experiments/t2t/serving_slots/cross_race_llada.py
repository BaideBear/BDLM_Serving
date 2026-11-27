"""Registered LLaDA2.1 cross-block race on one protected GPU, batch=1.

C64 opens the next physical block under four registered gates (off/eager/half/
quarter). C16 and C8 keep one byte-identical head plan and differ only in
whether residual seats borrow a read-only contiguous tail build. Every call is
audited by ``llada_fixed_family_eval.evaluate_fixed``: real convergence, real
debt, real decoder rows, no CPU inference and no synthesized control.

Deployment contract: run as a module inside the GPU handoff worker; it touches
``--control/ready`` before loading weights and waits for ``--control/go`` so the
outer launcher can stop the holder first. Raw rows never leave the cluster.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time

from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.methods.t2t.budget_family import Law
from .llada_fixed_family_eval import CAP, KEYS, evaluate_fixed

ARMS = (
    ("c64-off", 64, "off", False),
    ("c64-eager", 64, "eager", False),
    ("c64-half", 64, "half", False),
    ("c64-quarter", 64, "quarter", False),
    ("c16-noborrow", 16, "head", False),
    ("c16-borrow", 16, "head", True),
    ("c8-noborrow", 8, "head", False),
    ("c8-borrow", 8, "head", True),
)
ORDER = (0, 10, 20, *(i for i in range(90) if i not in (0, 10, 20)))
COLUMNS = ("arm", "index", "dataset", "task_id", "seed", "q", "cross_open", "borrow",
           "correct", "N", "F", "Q", "TPF", "efficiency", "head_Q", "tail_Q",
           "tail_read_only_Q", "open_F", "borrow_F", "max_decoder_Q", "failure")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with tmp.open("x") as f:
        json.dump(obj, f, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read(path):
    return json.loads(path.read_bytes())


def rotate(index):
    k = index % len(ARMS)
    return ARMS[k:] + ARMS[:k]


def load_gpu(args, before_cuda):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from bdlm_serving.runtime.serial import install_llada_rectangular_mask

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    opts = dict(trust_remote_code=True, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(args.checkpoint, **opts)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, **opts, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    install_llada_rectangular_mask(model, allow_selected_cache=True)
    before_cuda()
    model.to("cuda")
    if model.device.type != "cuda":
        raise RuntimeError("refusing CPU inference; model must be on CUDA")
    return tok, model


def summary(out):
    rows = [read(p) for p in sorted((out / "rows").glob("*.json"))]
    by_arm = {}
    for r in rows:
        a = by_arm.setdefault(r["arm"], dict(arm=r["arm"], requests=0, correct=0, N=0, F=0, Q=0,
                                             head_Q=0, tail_Q=0, tail_read_only_Q=0,
                                             open_F=0, borrow_F=0, failures=0, tpf=0.0))
        a["requests"] += 1
        a["correct"] += int(bool(r["correct"]))
        a["N"] += r["N"]
        a["F"] += r["F"]
        a["Q"] += r["Q"]
        a["head_Q"] += r["head_Q"]
        a["tail_Q"] += r["tail_Q"]
        a["tail_read_only_Q"] += r["tail_read_only_Q"]
        a["open_F"] += r["open_F"]
        a["borrow_F"] += r["borrow_F"]
        a["failures"] += int(bool(r["failure"]))
        a["tpf"] += r["TPF"]
    for a in by_arm.values():
        a["macro_tpf"] = a["tpf"] / a["requests"] if a["requests"] else None
        a["efficiency"] = a["N"] / a["Q"] if a["Q"] else None
    return dict(rows=len(rows), by_arm=by_arm)


def write_summary(out):
    data = summary(out)
    save(out / "summary.json", data)
    tmp = out / "summary.csv.tmp"
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for p in sorted((out / "rows").glob("*.json")):
            r = read(p)
            w.writerow({k: r.get(k) for k in COLUMNS})
    os.replace(tmp, out / "summary.csv")
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "samples", "reference", "output", "control"):
        ap.add_argument("--" + name, required=True, type=Path)
    ap.add_argument("--deadline", required=True, type=int)
    ap.add_argument("--indices", default="order",
                    help='"order" for the registered 0,10,20-first 90-case order, or a comma list')
    a = ap.parse_args()
    import torch
    import transformers

    if torch.cuda.device_count() != 1 or not torch.cuda.is_available():
        raise RuntimeError("Expected one protected GPU")
    tasks = load_tasks(a.samples)
    if [t["index"] for t in tasks] != list(range(90)):
        raise ValueError("Not the registered frozen90 ordering")
    indices = (list(ORDER) if a.indices == "order"
               else [int(x) for x in a.indices.split(",")])
    if len(set(indices)) != len(indices) or any(i not in range(90) for i in indices):
        raise ValueError("Invalid index list")
    prompts = {}
    for i in indices:
        ref = read(a.reference / f"{i:02d}-native32.json")
        if ref.get("generation_cap") != CAP or not ref.get("prompt_ids"):
            raise ValueError(f"Unregistered native reference for {i}")
        if ref["index"] != i:
            raise ValueError(f"Reference index mismatch for {i}")
        prompts[i] = ref["prompt_ids"]

    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "rows").mkdir(exist_ok=True)
    a.control.mkdir(parents=True, exist_ok=True)
    (a.control / "ready").touch(exist_ok=False)
    while not (a.control / "go").exists():
        if time.time() >= a.deadline - 120:
            save(a.output / "stopped.json", dict(reason="no go", at=time.time()))
            return
        time.sleep(0.5)
    tok, model = load_gpu(a, lambda: None)
    save(a.output / "run.json", dict(
        protocol="llada-cross-race-v1", arms=[dict(arm=x[0], q=x[1], cross_open=x[2], borrow=x[3])
                                               for x in ARMS],
        indices=indices, checkpoint=str(a.checkpoint), checkpoint_files=len(list(a.checkpoint.glob("*.safetensors"))),
        samples_sha256=sha(a.samples), reference=str(a.reference), deadline=a.deadline,
        gpu=torch.cuda.get_device_name(), host=socket.gethostname(),
        torch=str(torch.__version__), transformers=transformers.__version__,
        sources={p.name: sha(p) for p in (Path(__file__),)},
        length_cap=CAP, law=[1, 8], batch=1, retirement="all_resident_no_masks_no_token_debt",
        case_failure_score=0))
    law = Law(1, 8)
    deadline = a.deadline - 60
    total = 0
    for i in indices:
        if time.time() >= deadline:
            save(a.output / "stopped.json", dict(reason="deadline", at=time.time(), completed=total))
            return
        for arm, q, mode, borrow in rotate(i):
            if time.time() >= deadline:
                save(a.output / "stopped.json", dict(reason="deadline", at=time.time(), completed=total))
                return
            dst = a.output / "rows" / f"{arm}-{i:02d}.json"
            if dst.exists():
                continue
            with (a.output / "rows" / f"{arm}-{i:02d}.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if dst.exists():
                    continue
                start = time.monotonic()
                row = evaluate_fixed(model, tok, tasks[i], prompts[i], q, law, cross=(mode, borrow))
                row.update(arm=arm, q=q, cross_open=mode, borrow=borrow,
                           wall_ms=int(1000 * (time.monotonic() - start)),
                           samples_sha256=sha(a.samples), host=socket.gethostname(),
                           gpu=torch.cuda.get_device_name())
                if any(row.get(k) is None for k in ("head_Q", "tail_Q")):
                    raise RuntimeError("Cross ledger missing")
                save(dst, row)
                total += 1
                print(f"SEAL {arm:12s} {i:02d} correct={row['correct']} N/F/Q={row['N']}/"
                      f"{row['F']}/{row['Q']} head={row['head_Q']} tail={row['tail_Q']} "
                      f"openF={row['open_F']} borrowF={row['borrow_F']} ms={row['wall_ms']}", flush=True)
        write_summary(a.output)
    write_summary(a.output)
    save(a.output / "complete.json", dict(rows=total, at=time.time(), run=sha(a.output / "run.json")))


if __name__ == "__main__":
    main()

"""Registered LLaDA2.1 serving-slot suites on one protected GPU, batch=1.

C64 opens the next physical block under four registered gates (off/eager/half/
quarter). C16 and C8 keep one byte-identical head plan and differ only in
whether residual seats borrow a read-only contiguous tail build. Every call is
audited by ``llada_fixed_family_eval.evaluate_fixed``: real convergence, real
debt, real decoder rows, no CPU inference and no synthesized control.

``--suite low-dose`` reuses the same hard decoder / head-only / no-borrow
executor and varies only the optional clean-prefix share on C4/8/16/32:
0, 1/8, 1/4, 1/2. C4 x 1/8 is not run: ``choose``'s prefix cap is
``min(room,32)*1//8 = 0`` for every C4 state, so it is provably the same row as
C4 x 0. The alias is recorded instead of re-running it.

Deployment contract: run as a module inside the GPU handoff worker; it touches
``--control/ready`` before loading weights and waits for ``--control/go`` so the
outer launcher can stop the holder first. Raw rows never leave the cluster.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
import time

from bdlm_serving.evaluation.frozen90 import load_tasks
from bdlm_serving.methods.t2t.budget_family import Law
from .llada_fixed_family_eval import CAP, KEYS, evaluate_fixed

ARMS = (
    ("c64-off", 64, "off", False, "sparse"),
    ("c64-eager", 64, "eager", False, "sparse"),
    ("c64-half", 64, "half", False, "sparse"),
    ("c64-quarter", 64, "quarter", False, "sparse"),
    ("c64-full-off", 64, "off", False, "head"),
    ("c64-full-eager", 64, "eager", False, "head"),
    ("c64-full-half", 64, "half", False, "head"),
    ("c64-full-quarter", 64, "quarter", False, "head"),
    ("c16-noborrow", 16, "head", False, "sparse"),
    ("c16-borrow", 16, "head", True, "sparse"),
    ("c8-noborrow", 8, "head", False, "sparse"),
    ("c8-borrow", 8, "head", True, "sparse"),
)
ORDER = (0, 10, 20, *(i for i in range(90) if i not in (0, 10, 20)))
COLUMNS = ("arm", "index", "dataset", "task_id", "seed", "q", "cross_open", "borrow", "coverage",
           "device_uuid", "worker_pid",
           "correct", "N", "F", "Q", "TPF", "efficiency", "head_Q", "tail_Q",
           "tail_read_only_Q", "open_F", "borrow_F", "head_full_F", "full64_F",
           "max_decoder_Q", "failure")

LOW_DOSE_TIERS = (4, 8, 16, 32)
LOW_DOSE_SHARES = ((0, 1), (1, 8), (1, 4), (1, 2))


def _arm_name(q, num, den):
    return f"c{q}-p0" if num == 0 else f"c{q}-p{num}of{den}"


# C4 x 1/8 is omitted: floor(min(room,32)/8) == 0 for every room <= 4.
LOW_DOSE_ARMS = tuple(
    (_arm_name(q, num, den), q, "head", False, "sparse", num, den)
    for q in LOW_DOSE_TIERS for num, den in LOW_DOSE_SHARES
    if not (q == 4 and (num, den) == (1, 8))
)
LOW_DOSE_ALIASES = (
    dict(alias="c4-p1of8", of="c4-p0", q=4, share="1/8",
         reason="prefix cap = min(room,32)*1//8 = 0 for room<=4; identical choose plan to 0/1"),
)
LOW_DOSE_COHORTS = ("original30", "new60", "all90")
LOW_DOSE_COLUMNS = (
    "arm", "cohort", "q", "prefix_share", "requests", "correct", "N", "F", "Q",
    "macro_tpf", "efficiency", "warm_F", "warm_rows", "warm_prefix_seats",
    "warm_prefix_row_share", "prefix_seats", "mask_seats", "dirty_selected_seats",
    "dirty_before_seats", "m2t_accepts", "t2t_edits", "t2t_revoked", "cold_seats",
    "drain_F", "zero_prefix_warm_F", "failures",
)


def prefix_cap(q, num, den, dirty=0):
    """The exact clean-prefix cap used by choose() on the head-only path."""
    room = max(0, q - dirty)
    return min(room, 32) * num // den


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with tmp.open("x") as f:
        json.dump(obj, f, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_once(path, obj):
    """Publish a canonical worker manifest without clobbering a peer's copy."""
    tmp = path.with_suffix(f".{os.getpid()}.once.tmp")
    with tmp.open("x") as f:
        json.dump(obj, f, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)  # atomic create-if-absent
    except FileExistsError:
        pass
    finally:
        tmp.unlink(missing_ok=True)


def read(path):
    return json.loads(path.read_bytes())


def rotate(arms, index):
    k = index % len(arms)
    return arms[k:] + arms[:k]


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
        a = by_arm.setdefault(r["arm"], dict(arm=r["arm"], coverage=r.get("coverage"), requests=0,
                                             correct=0, N=0, F=0, Q=0,
                                             head_Q=0, tail_Q=0, tail_read_only_Q=0,
                                             open_F=0, borrow_F=0, head_full_F=0, full64_F=0,
                                             failures=0, tpf=0.0))
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
        a["head_full_F"] += r.get("head_full_F", 0)
        a["full64_F"] += r.get("full64_F", 0)
        a["failures"] += int(bool(r["failure"]))
        a["tpf"] += r["TPF"]
    for a in by_arm.values():
        a["macro_tpf"] = a["tpf"] / a["requests"] if a["requests"] else None
        a["efficiency"] = a["N"] / a["Q"] if a["Q"] else None
    return dict(rows=len(rows), by_arm=by_arm)


def write_summary(out):
    data = summary(out)
    save(out / "summary.json", data)
    tmp = out / f"summary.csv.{os.getpid()}.tmp"
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for p in sorted((out / "rows").glob("*.json")):
            r = read(p)
            w.writerow({k: r.get(k) for k in COLUMNS})
    os.replace(tmp, out / "summary.csv")
    return data


def low_dose_metrics(row):
    """Seats actually used on warm calls, plus MASK/dirty/T2T/revocation totals."""
    trace = row["trace"]
    warm = [e for e in trace if e["sampled"] and not e["cold"]]
    warm_rows = sum(e["q"] for e in warm)
    warm_prefix = sum(len(e["prefix"]) for e in warm)
    return dict(
        warm_F=len(warm), warm_rows=warm_rows, warm_prefix_seats=warm_prefix,
        warm_prefix_row_share=(warm_prefix / warm_rows) if warm_rows else None,
        prefix_seats=sum(len(e["prefix"]) for e in trace),
        mask_seats=sum(len(e["mask"]) for e in trace),
        dirty_selected_seats=sum(len(e["dirty"]) for e in trace),
        dirty_before_seats=sum(len(e["dirty_before"]) for e in trace),
        m2t_accepts=sum(len(e["admitted"]) for e in trace),
        t2t_edits=sum(len(e["edited"]) for e in trace),
        t2t_revoked=sum(len(e["revoked"]) for e in trace),
        cold_seats=sum(len(e["cold"]) for e in trace),
        drain_F=sum(bool(e["drain_only"]) for e in trace),
        zero_prefix_warm_F=sum(1 for e in warm if not e["prefix"]),
    )


def audit_low_dose(row, num, den):
    """Batch check the executor's real seats against the registered prefix cap."""
    for e in row["trace"]:
        rows = set(e["rows"])
        if len(e["sampled"]) > e["q"] or not set(e["sampled"]) <= rows:
            raise RuntimeError("Low-dose sampled seats exceed the physical offer")
        if not set(e["prefix"]) <= rows - set(e["mask"]) - set(e["cold"]) - set(e["dirty"]):
            raise RuntimeError("Low-dose prefix overlaps MASK/cold/dirty")
        if e["sampled"] and not e["cold"]:
            cap = prefix_cap(e["q"], num, den, len(e["dirty"]))
            if len(e["prefix"]) > cap:
                raise RuntimeError(
                    f"Warm prefix {len(e['prefix'])} exceeds cap {cap} "
                    f"(q={e['q']} dirty={len(e['dirty'])} {num}/{den})")
        if not set(e["admitted"] + e["edited"] + e["revoked"]) <= set(e["sampled"]):
            raise RuntimeError("Low-dose T2T/revocation was not measured")


def cohort(index):
    return "original30" if index < 30 else "new60"


def low_dose_summary(out):
    rows = [read(p) for p in sorted((out / "rows").glob("*.json"))]
    by = {}
    for r in rows:
        for coh in (cohort(r["index"]), "all90"):
            a = by.setdefault((r["arm"], coh), dict(
                arm=r["arm"], cohort=coh, q=r["q"], prefix_share=r.get("prefix_share"),
                requests=0, correct=0, N=0, F=0, Q=0, tpf_sum=0.0,
                warm_F=0, warm_rows=0, warm_prefix_seats=0, prefix_seats=0,
                mask_seats=0, dirty_selected_seats=0, dirty_before_seats=0,
                m2t_accepts=0, t2t_edits=0, t2t_revoked=0, cold_seats=0,
                drain_F=0, zero_prefix_warm_F=0, failures=0))
            a["requests"] += 1
            a["correct"] += int(bool(r["correct"]))
            a["N"] += r["N"]; a["F"] += r["F"]; a["Q"] += r["Q"]
            a["tpf_sum"] += r["TPF"]
            for k in ("warm_F", "warm_rows", "warm_prefix_seats", "prefix_seats",
                      "mask_seats", "dirty_selected_seats", "dirty_before_seats",
                      "m2t_accepts", "t2t_edits", "t2t_revoked", "cold_seats",
                      "drain_F", "zero_prefix_warm_F"):
                a[k] += int(r.get(k, 0))
            a["failures"] += int(bool(r["failure"]))
    for a in by.values():
        a["macro_tpf"] = a["tpf_sum"] / a["requests"] if a["requests"] else None
        a["efficiency"] = a["N"] / a["Q"] if a["Q"] else None
        a["warm_prefix_row_share"] = (a["warm_prefix_seats"] / a["warm_rows"]
                                      if a["warm_rows"] else None)
        a.pop("tpf_sum")
    groups = sorted(by.values(), key=lambda x: (
        int(x["arm"].split("-")[0][1:]), x["arm"], LOW_DOSE_COHORTS.index(x["cohort"])))
    return dict(rows=len(rows), aliases=list(LOW_DOSE_ALIASES), by_arm_cohort=groups)


def write_low_dose_summary(out):
    data = low_dose_summary(out)
    save(out / "summary-low-dose.json", data)
    tmp = out / f"summary-low-dose.csv.{os.getpid()}.tmp"
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOW_DOSE_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for a in data["by_arm_cohort"]:
            w.writerow(a)
    os.replace(tmp, out / "summary-low-dose.csv")
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "samples", "reference", "output", "control"):
        ap.add_argument("--" + name, required=True, type=Path)
    ap.add_argument("--deadline", required=True, type=int)
    ap.add_argument("--indices", default="order",
                    help='"order" for the registered 0,10,20-first 90-case order, or a comma list')
    ap.add_argument("--suite", choices=("cross", "low-dose"), default="cross",
                    help="cross = C64 gates x coverage plus C16/C8 borrow; "
                         "low-dose = C4/8/16/32 x clean-prefix share, head-only/no borrow")
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
    low = a.suite == "low-dose"
    arms = LOW_DOSE_ARMS if low else ARMS
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
    (a.output / "workers").mkdir(exist_ok=True)
    a.control.mkdir(parents=True, exist_ok=True)
    (a.control / "ready").touch(exist_ok=False)
    device_uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    worker = a.output / "workers" / f"worker-{os.getpid()}.json"
    has_coverage = "coverage" in inspect.signature(evaluate_fixed).parameters
    while not (a.control / "go").exists():
        if time.time() >= a.deadline - 120:
            save(worker.with_name(f"stopped-{os.getpid()}.json"),
                 dict(reason="no go", at=time.time()))
            return
        time.sleep(0.5)
    save(worker, dict(pid=os.getpid(), device_uuid=device_uuid,
                      gpu=torch.cuda.get_device_name(), host=socket.gethostname(),
                      indices=indices, deadline=a.deadline, started=time.time()))
    tok, model = load_gpu(a, lambda: None)
    # Canonical protocol manifest only; per-worker indices/device live in workers/.
    if low:
        manifest = dict(
            protocol="llada-low-dose-v1", suite="low-dose",
            arms=[dict(arm=x[0], q=x[1], cross_open=x[2], borrow=x[3], coverage="sparse",
                       law=[x[5], x[6]]) for x in arms],
            aliases=list(LOW_DOSE_ALIASES), tiers=list(LOW_DOSE_TIERS),
            shares=[f"{n}/{d}" for n, d in LOW_DOSE_SHARES], law="per-arm")
    else:
        manifest = dict(
            protocol="llada-cross-race-v2", suite="cross",
            arms=[dict(arm=x[0], q=x[1], cross_open=x[2], borrow=x[3], coverage=x[4])
                  for x in arms], law=[1, 8])
    manifest.update(
        checkpoint=str(a.checkpoint), checkpoint_files=len(list(a.checkpoint.glob("*.safetensors"))),
        samples_sha256=sha(a.samples), reference=str(a.reference), deadline=a.deadline,
        gpu_model=torch.cuda.get_device_name(), host=socket.gethostname(),
        torch=str(torch.__version__), transformers=transformers.__version__,
        sources={p.name: sha(p) for p in (Path(__file__),)},
        length_cap=CAP, batch=1, retirement="all_resident_no_masks_no_token_debt",
        case_failure_score=0)
    if (a.output / "run.json").exists():
        prev = read(a.output / "run.json")
        if prev.get("suite", "cross") != a.suite:
            raise ValueError("Output directory already belongs to another suite")
    save_once(a.output / "run.json", manifest)
    deadline = a.deadline - 60
    total = 0
    for i in indices:
        if time.time() >= deadline:
            save(worker.with_name(f"stopped-{os.getpid()}.json"),
                 dict(reason="deadline", at=time.time(), completed=total, indices=indices))
            return
        for spec in rotate(arms, i):
            arm, q, mode, borrow, cover = spec[:5]
            num, den = (spec[5], spec[6]) if low else (1, 8)
            if time.time() >= deadline:
                save(worker.with_name(f"stopped-{os.getpid()}.json"),
                     dict(reason="deadline", at=time.time(), completed=total, indices=indices))
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
                kwargs = dict(cross=(mode, borrow))
                if cover == "head":
                    if not has_coverage:
                        raise RuntimeError("This checkout's evaluate_fixed has no coverage axis")
                    kwargs["coverage"] = cover
                row = evaluate_fixed(model, tok, tasks[i], prompts[i], q, Law(num, den), **kwargs)
                row.update(arm=arm, q=q, cross_open=mode, borrow=borrow, coverage=cover,
                           law=[num, den],
                           wall_ms=int(1000 * (time.monotonic() - start)),
                           samples_sha256=sha(a.samples), host=socket.gethostname(),
                           gpu=torch.cuda.get_device_name(), device_uuid=device_uuid,
                           worker_pid=os.getpid())
                if low:
                    audit_low_dose(row, num, den)
                    row.update(suite="low-dose", prefix_share=f"{num}/{den}", **low_dose_metrics(row))
                else:
                    if any(row.get(k) is None for k in ("head_Q", "tail_Q", "head_full_F")):
                        raise RuntimeError("Cross ledger missing")
                save(dst, row)
                total += 1
                if low:
                    print(f"SEAL {arm:12s} {i:02d} correct={row['correct']} N/F/Q={row['N']}/"
                          f"{row['F']}/{row['Q']} warmF={row['warm_F']} "
                          f"warmPrefix={row['warm_prefix_seats']}/{row['warm_rows']} "
                          f"mask={row['mask_seats']} dirty={row['dirty_selected_seats']} "
                          f"T2T={row['t2t_edits']} rev={row['t2t_revoked']} ms={row['wall_ms']}",
                          flush=True)
                else:
                    print(f"SEAL {arm:12s} {i:02d} correct={row['correct']} N/F/Q={row['N']}/"
                          f"{row['F']}/{row['Q']} head={row['head_Q']} tail={row['tail_Q']} "
                          f"cover={row['coverage']} headFullF={row['head_full_F']} "
                          f"openF={row['open_F']} borrowF={row['borrow_F']} ms={row['wall_ms']}",
                          flush=True)
        (write_low_dose_summary if low else write_summary)(a.output)
    (write_low_dose_summary if low else write_summary)(a.output)
    save(worker.with_name(f"complete-{os.getpid()}.json"),
         dict(rows=total, at=time.time(), suite=a.suite, run=sha(a.output / "run.json"),
              indices=indices))


if __name__ == "__main__":
    main()

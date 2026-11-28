"""A0 physical A0 only: verified-EOS Live Q64 right-edge vs stratified prefix.

The control is replayed against the sealed verified-EOS `on` arm before the
spread arm starts.  Both arms use Law(1,8), early=True, fused first verify,
native T2T/RNG/sampler clock and exact Q64.  The only planned difference is the
identities selected from the same strict clean-prefix pool after `choose` has
fixed the dose.  Any mismatch seals partial evidence and stops.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import ExitStack
from types import SimpleNamespace

from .gemma_ibs_certified_eos import require
from .gemma_verified_eos_eval import run

SIX = (7, 13, 25, 33, 55, 71)
A0_DEVICE_INDEX = 0
A0_UUID = "GPU-d3c3fc9c-b15d-9776-9094-7d3b85b705c1"
MODEL = Path("/gs/models/diffusiongemma-cd40195")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard_a0(stage, gpu_uuid=A0_UUID):
    apps = subprocess.check_output(["nvidia-smi", "-i", str(A0_DEVICE_INDEX),
                                    "--query-compute-apps=pid,gpu_uuid",
                                    "--format=csv,noheader,nounits"], text=True).strip()
    uuid = subprocess.check_output(["nvidia-smi", "-i", str(A0_DEVICE_INDEX),
                                    "--query-gpu=uuid", "--format=csv,noheader"],
                                   text=True).strip()
    if uuid != gpu_uuid or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
        raise RuntimeError(f"A0 UUID mismatch: {uuid!r} / {os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    if stage == "before":
        mem, util = subprocess.check_output(["nvidia-smi", "-i", str(A0_DEVICE_INDEX),
                                             "--query-gpu=memory.used,utilization.gpu",
                                             "--format=csv,noheader,nounits"], text=True).strip().split(",")
        if mem.strip() != "0" or util.strip() != "0":
            raise RuntimeError(f"A0 not idle: {mem.strip()} MiB / {util.strip()}%")
    expected = "" if stage == "before" else f"{os.getpid()}, {gpu_uuid}"
    if apps != expected:
        raise RuntimeError(f"Unexpected A0 apps at {stage}: {apps!r}")


def load_runtime(samples):
    import torch
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from bdlm_serving.evaluation import frozen, frozen90, scoring
    from bdlm_serving.runtime.diffusiongemma import check_snapshot
    require(transformers.__version__ == "5.12.0", "Wrong transformers version")
    require(sha(samples) == "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed",
            "Wrong frozen samples")
    check_snapshot(MODEL)
    require(sha(frozen.ROOT / "benchmarks/checkpoints/diffusiongemma.json") ==
            "ceb951734eb354728d7ba984093739119addbeda981b554af46d5dc105071f53",
            "Wrong model manifest")
    proc = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", experts_implementation="grouped_mm",
        device_map={"": "cuda:0"}).eval()
    return dict(torch=torch, model=model, proc=proc, tasks=frozen90.load_tasks(samples),
                scoring=scoring, transformers=transformers.__version__)


def replay_match(control, sealed):
    keys = ("output_ids", "tape", "input_sha256", "hit", "encoder_digest", "N", "correct",
            "decoder_F", "decoder_Q", "encoder_F", "encoder_Q", "F", "route_F", "route_Q")
    return {key: control[key] == sealed[key] for key in keys}


def score_delivery(handle, task, delivery):
    ids = delivery["output_ids"]
    eos = set(handle["model"].generation_config.eos_token_id)
    special = (set(handle["proc"].tokenizer.all_special_ids) | eos |
               {handle["model"].generation_config.pad_token_id})
    text = handle["proc"].tokenizer.decode(ids, skip_special_tokens=True)
    delivery["N"] = sum(token not in special for token in ids)
    delivery["correct"] = bool(handle["scoring"].score(text, task)["correct"])


def first_fork(control, treatment):
    for tick, (old, new) in enumerate(zip(control["tape"], treatment["tape"])):
        if old != new:
            ce, te = control["events"][tick], treatment["events"][tick]
            cp, tp = ce["parts"][0], te["parts"][0]
            return dict(kind="trace", tick=tick, control=ce, treatment=te,
                        control_prefix=cp["prefix"], treatment_prefix=tp["prefix"],
                        control_prefix_pool=cp.get("prefix_pool"),
                        treatment_prefix_pool=tp.get("prefix_pool"))
    if len(control["tape"]) != len(treatment["tape"]):
        return dict(kind="trace_length", control=len(control["tape"]),
                    treatment=len(treatment["tape"]))
    return None


def validate_fork(fork):
    if fork is None or fork["kind"] != "trace":
        raise RuntimeError("Spread arm has no first physical fork")
    ce, te = fork["control"], fork["treatment"]
    cp, tp = ce["parts"][0], te["parts"][0]
    require(ce["tick"] == te["tick"] and ce["q"] == te["q"] == 64
            and ce["route"] == te["route"] == "sparse"
            and len(ce["positions"]) == len(te["positions"]) == 64
            and len(cp["prefix"]) == len(tp["prefix"])
            and len(cp["sampled"]) == len(tp["sampled"]),
            "First fork changes offer, route, dose or sampling arity")
    require(fork["control_prefix_pool"] is None and isinstance(fork["treatment_prefix_pool"], list)
            and fork["treatment_prefix_pool"], "Missing strict spread prefix pool")
    require(set(tp["prefix"]) <= set(fork["treatment_prefix_pool"])
            and tp["prefix"][-1] == max(fork["treatment_prefix_pool"]),
            "Spread prefix left strict eligible pool or lost frontier")
    require(all(0 <= x < 256 for x in fork["treatment_prefix_pool"]),
            "Prefix pool crosses physical block")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--deadline", type=int, required=True)
    parser.add_argument("--samples", type=Path,
                        default=Path("/gs/code/samples.json"))
    parser.add_argument("--references", type=Path, required=True)
    args = parser.parse_args()
    if not str(args.root).startswith("/dev/shm/"):
        print("A0 prefix identity root must be under /dev/shm", file=sys.stderr)
        return 2
    try:
        guard_a0("before")
    except Exception as exc:
        print(f"A0 PREFIX IDENTITY NOT STARTED: {exc}", file=sys.stderr)
        return 4
    args.root.mkdir(parents=True, exist_ok=True)
    handle = load_runtime(args.samples)
    rows, failures = [], []
    for index in SIX:
        require(time.time() < args.deadline, "Deadline")
        guard_a0("request")
        task = next(t for t in handle["tasks"] if t["index"] == index)
        reference_path = args.references / f"C64-pair-{index:02}.json"
        sealed = json.loads(reference_path.read_bytes())
        require(sealed["index"] == index and sealed["q"] == 64
                and sealed["schema"] == "verified-eos-pair-v1"
                and sealed["samples_sha256"] == "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed",
                f"Wrong sealed reference {index}")
        error = None
        fork = None
        try:
            data = handle["proc"].apply_chat_template(
                [{"role": "user", "content": task["prompt"]}], tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda:0")
            control = run(handle["model"], data, task, 64, True, "right", True)
            score_delivery(handle, task, control)
            matches = replay_match(control, sealed["on"])
            require(all(matches.values()), f"Native right replay mismatch: {matches}")
            treatment = run(handle["model"], data, task, 64, True, "spread", True)
            score_delivery(handle, task, treatment)
            fork = first_fork(control, treatment)
            validate_fork(fork)
            row = dict(index=index, task_id=task["task_id"], dataset=task["dataset"],
                       seed=task["seed"], reference_sha256=sha(reference_path),
                       native_replay=matches, first_fork={
                           key: fork[key] for key in ("kind", "tick", "control_prefix",
                                                      "treatment_prefix",
                                                      "control_prefix_pool",
                                                      "treatment_prefix_pool")},
                       control=control, treatment=treatment)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            error = repr(exc)
            row = dict(index=index, task_id=task.get("task_id"), dataset=task.get("dataset"),
                       seed=task.get("seed"), error=error, control=locals().get("control"),
                       treatment=locals().get("treatment"))
        row["error"] = error
        rows.append(row)
        (args.root / f"case-{index:02}.json").write_text(json.dumps(row, separators=(",", ":"),
                                                                   allow_nan=False) + "\n")
        compact = {k: row.get(k) for k in ("index", "error", "native_replay", "first_fork")}
        for arm in ("control", "treatment"):
            v = row.get(arm) or {}
            compact[arm] = {k: v.get(k) for k in ("N", "F", "decoder_F", "decoder_Q",
                                                  "encoder_F", "encoder_Q", "route_F",
                                                  "route_Q", "prefix_order")}
        print("CASE " + json.dumps(compact, separators=(",", ":"), allow_nan=False), flush=True)
        if error:
            print("STOP " + json.dumps(dict(reason="first_invalid_case", index=index,
                                            error=error), separators=(",", ":")), flush=True)
            failures.append(index)
            break
    gate = "passed" if not failures and len(rows) == len(SIX) else "failed"
    summary = dict(schema="gemma-verified-eos-q64-prefix-identity-a0-six-v1", gpu=A0_UUID,
                   q=64, cases=list(SIX), arms=("right", "spread"), gate=gate,
                   failed=failures, per_case=[dict(index=r["index"], error=r.get("error"))
                                               for r in rows],
                   transformers=handle["transformers"])
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY " + json.dumps(dict(gate=gate, failed=failures),
                                  separators=(",", ":")), flush=True)
    return 0 if gate == "passed" else 5


if __name__ == "__main__":
    raise SystemExit(main())

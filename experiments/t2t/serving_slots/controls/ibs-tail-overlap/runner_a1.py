"""A1-only runner for the registered fused strict-IBS Q8 tail-overlap pilot.

Emits one canonical JSON receipt per arm on stdout.  The external relay owns
exclusive n164 storage; this runner creates no persistent A800 output.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from unittest.mock import patch

A1_UUID = "GPU-861c9b5d-1e73-2550-f832-b73988bc64ff"
Q = 8
B = 32
CAP = 2048
INDICES = (7, 13, 25, 33, 55, 71)
ARMS = ("stock", "noop", "overlap")
ROTATION = (
    ("stock", "noop", "overlap"),
    ("noop", "overlap", "stock"),
    ("overlap", "stock", "noop"),
    ("stock", "noop", "overlap"),
    ("noop", "overlap", "stock"),
    ("overlap", "stock", "noop"),
)
SAMPLES_SHA256 = "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed"
FIXED_SHA = "b95003b5f8867bd6ef7d5126c9814e08fb6d03bab57f67642618028d8a1960d4"
UPSTREAM_SHA = "f21a4704697cace53b19dfbbbf94bbabc877241b9c979e15b0d40e9a2ea0d223"
SUCCESSOR_KEYS = {
    "head_rows", "tail_rows", "tail_proposed", "tail_writes", "discarded_writes",
    "promoted", "head_changed", "head_key_end", "tail_key_end", "layer_widths",
}
SOURCE_FILES = (
    "bdlm_serving/methods/optimus/fixed.py",
    "bdlm_serving/methods/optimus/upstream.py",
    "bdlm_serving/runtime/batch.py",
    "bdlm_serving/runtime/serial.py",
    "bdlm_serving/evaluation/outcomes.py",
    "bdlm_serving/evaluation/scoring.py",
    "bdlm_serving/evaluation/frozen90.py",
    "experiments/t2t/serving_slots/ibs_prefix_pair.py",
    "experiments/t2t/serving_slots/llada_all_slots90.py",
    "experiments/t2t/serving_slots/shared_llada_worker.py",
)
WORK_FILES = ("close_overlap.py", "fused.py", "runner_a1.py", "static_cases.json",
              "controls/Optimus/sequence.py", "controls/Optimus/unmasking.py")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    if isinstance(path, (bytes, bytearray)):
        return hashlib.sha256(path).hexdigest()
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def blob(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity(value):
    return hashlib.sha256(blob(value)).hexdigest()


def ids_sha(values):
    return hashlib.sha256(blob(list(values))).hexdigest()


def local_prompt(tok, task, static):
    text = tok.apply_chat_template(
        [{"role": "user", "content": task["prompt"]}], tokenize=False,
        add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
    actual = ids_sha(ids)
    require(actual == static["prompt_ids_sha256"] and len(ids) == static["prompt_len"],
            f"prompt drift on {task['index']:02d}")
    require((task["dataset"], task["task_id"], task["seed"]) ==
            (static["dataset"], static["task_id"], static["seed"]), f"identity drift {task['index']:02d}")
    return list(ids), "chat_template"


def check_registration(args, registration, static):
    require(registration["study"] == "llada21-ibs-q8-tail-overlap-a1-v1", "wrong study")
    require(tuple(registration["indices"]) == INDICES and tuple(registration["arms"]) == ARMS and
            all(tuple(row) == row_expected for row, row_expected in
                zip(registration["rotation"], ROTATION)), "wrong registration case order")
    require(registration["q"] == Q and registration["block"] == B and registration["cap"] == CAP,
            "wrong registered geometry")
    require(registration["gpu_uuid"] == A1_UUID and registration["deadline"] == args.deadline,
            "wrong physical A1 or deadline")
    require(registration["main_pass_authorized"] is True, "main-thread pass is not registered")
    require(tuple(static["indices"]) == INDICES and static["q"] == Q and
            static["block"] == B and static["generation_cap"] == CAP, "static case drift")
    source_root = Path(args.src)
    for relative in SOURCE_FILES:
        require(sha(source_root/relative) == registration["sources"][relative],
                f"registered source drift: {relative}")
    work_root = Path(args.work)
    for relative in WORK_FILES:
        require(sha(work_root/relative) == registration["sources"][relative],
                f"registered work drift: {relative}")
    require(sha(Path(__file__).read_bytes()) == registration["sources"]["runner_a1.py"],
            "runner changed after registration")
    require(sha(Path(args.samples)) == SAMPLES_SHA256, "frozen90 sample drift")
    require(identity(static) == registration["static_cases_sha256"], "static case digest drift")


def preflight(args):
    registration = json.loads(Path(args.registration).read_bytes())
    static = json.loads((Path(args.work)/"static_cases.json").read_bytes())
    check_registration(args, registration, static)
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from transformers import AutoTokenizer
    tasks = load_tasks(Path(args.samples))
    require([task["index"] for task in tasks] == list(range(90)), "wrong frozen90 ordering")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True,
                                              local_files_only=True)
    for index in INDICES:
        local_prompt(tokenizer, tasks[index], static["cases"][str(index)])
    print(json.dumps({"preflight": "ok", "registration": identity(registration),
                      "indices": list(INDICES), "arms": list(ARMS)}), flush=True)


def check_idle(deadline):
    require(time.time() < deadline - 120, "deadline is too close")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == A1_UUID, "runner is not pinned to A1 UUID")
    info = subprocess.check_output(
        ["nvidia-smi", "-i", A1_UUID, "--query-gpu=uuid,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    require(len(info) == 1, "A1 query failed")
    uuid, memory, utilization = [part.strip() for part in info[0].split(",")]
    require(uuid == A1_UUID and int(memory) == 0 and int(utilization) == 0,
            f"A1 is not empty: {info[0]}")
    apps = subprocess.check_output(
        ["nvidia-smi", "-i", A1_UUID, "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"], text=True).split()
    require(not apps, f"A1 compute process exists: {apps}")


def pristine(row):
    omit = {"arm", "index_ordinal", "noop_vote_receipt", "overlap_vote_receipt",
            "overlap_audit", "stock_noop_actions_equal", "historical_stock_match",
            "registration_sha256", "runtime", "host", "gpu", "gpu_uuid", "worker_pid"}
    return {key: value for key, value in row.items() if key not in omit}


def first_armed(row):
    for tick, event in enumerate(row["events"]):
        successor = event.get("successor")
        if successor is not None:
            return tick, event, successor
    raise RuntimeError("overlap arm had no fused joint forward")


def audit_pair(stock, noop, overlap, index, mask_id):
    require(pristine(stock) == pristine(noop), f"#{index:02d} no-op differs from stock")
    require(stock["events"] == noop["events"] and stock["calls"] == noop["calls"] and
            stock["forward_inputs"] == noop["forward_inputs"], f"#{index:02d} no-op physical drift")
    armed = [(tick, event, event["successor"]) for tick, event in enumerate(overlap["events"])
             if event.get("successor") is not None]
    if not armed:
        require(pristine(stock) == pristine(overlap), "untriggered overlap differs from stock")
        require(stock["events"] == overlap["events"] and stock["calls"] == overlap["calls"] and
                stock["forward_inputs"] == overlap["forward_inputs"],
                "untriggered overlap physical drift")
        return {"triggered": False, "first_joint_tick": None, "tail_rows": 0,
                "tail_writes": 0, "promoted": False, "head_changed": False,
                "head_write_set_drift": False}
    tick, event, successor = armed[0]
    require(tick < len(stock["events"]) and tick < len(noop["events"]), "first fork lacks stock context")
    require(stock["events"][:tick] == noop["events"][:tick] == overlap["events"][:tick] and
            stock["calls"][:tick] == noop["calls"][:tick] == overlap["calls"][:tick] and
            stock["forward_inputs"][:tick] == noop["forward_inputs"][:tick] ==
            overlap["forward_inputs"][:tick], "history before first joint forward changed")
    stock_event = stock["events"][tick]
    head = successor["head_rows"]
    tail = successor["tail_rows"]
    actual = head + tail
    tail_start = (stock_event["block"]+1)*B
    require(event["phase"] == "optimus" and event["physical_cap"] == Q and
            event["physical_forwards"] == 1, "wrong fused event")
    require(event["absolute_positions"] == actual and
            event["q_tokens"] == len(actual) <= Q, "actual combined geometry mismatch")
    require(head == stock_event["absolute_positions"], "head seats were preempted")
    require(tail and tail == sorted(set(tail)) and
            tail == list(range(tail_start, tail_start+len(tail))), "tail is not left contiguous")
    require(set(head).isdisjoint(tail), "head/tail overlap")
    require(successor["head_key_end"] == max(head)+1 and
            successor["tail_key_end"] == max(tail)+1, "key extents mismatch")
    require(successor["layer_widths"] == [len(actual)]*20, "per-layer joint width mismatch")
    require(set(successor) == SUCCESSOR_KEYS, "successor schema drift")
    for key in ("offer_tick", "physical_cap", "candidate_local", "dirty_before",
                "cached_prefix_before", "active_width", "block", "step"):
        require(stock_event.get(key) == event.get(key), f"original head pre-state changed: {key}")
    require(stock_event["visible_kv"] == successor["head_key_end"],
            "stock visible extent is not the head-only key end")
    live = dict(zip(event["absolute_positions"], overlap["forward_inputs"][tick], strict=True))
    require([live[pos] for pos in stock_event["absolute_positions"]] ==
            stock["forward_inputs"][tick], "original head input tokens changed")
    proposed = successor["tail_proposed"]
    require(len(proposed) == len(tail), "tail proposal width mismatch")
    selected = {pos for pos, token in zip(tail, proposed) if token != mask_id}
    writes = set(successor["tail_writes"])
    discarded = set(successor["discarded_writes"])
    require(writes <= selected and discarded <= selected and writes.isdisjoint(discarded) and
            writes | discarded <= set(tail) and
            (writes == set() if not successor["promoted"] else discarded == set()),
            "tail write ledger mismatch: "+json.dumps({
                "promoted": successor["promoted"], "tail": tail,
                "proposed": proposed, "mask_id": mask_id,
                "tail_writes": successor["tail_writes"],
                "discarded_writes": successor["discarded_writes"],
            }, sort_keys=True))
    require(successor["promoted"] is False or successor["head_changed"] is False,
            "promoted after a changed head")
    stock_written = set(stock_event["written_absolute"])
    overlap_written = set(event["written_absolute"])
    head_write_set_drift = (overlap_written-set(tail)) != stock_written
    if successor["promoted"]:
        require(set(successor["tail_writes"]) <= set(overlap_written) and
                tick+1 < len(overlap["events"]), "promoted tail debt was not carried")
        following = overlap["events"][tick+1]
        require(set(successor["tail_writes"]) <= set(following["dirty_before"]) and
                set(successor["tail_writes"]) <= set(following["absolute_positions"]),
                "promoted tail debt was not selected for payment")
        debt_inputs = dict(zip(following["absolute_positions"], overlap["forward_inputs"][tick+1],
                               strict=True))
        require(all(debt_inputs[pos] == proposed[tail.index(pos)] for pos in successor["tail_writes"]),
                "inherited tail debt token changed before payment")
    else:
        require(not (set(tail) & overlap_written) and
                not (tick+1 < len(overlap["events"]) and
                     set(tail) & set(overlap["events"][tick+1]["dirty_before"])),
                "discarded speculative state became debt")
    return {"triggered": True, "first_joint_tick": tick, "tail_rows": len(tail),
            "tail_writes": len(successor["tail_writes"]),
            "promoted": successor["promoted"], "head_changed": successor["head_changed"],
            "head_write_set_drift": head_write_set_drift}


def audit_receipts(noop_receipt, overlap_receipt):
    if not overlap_receipt:
        require(not noop_receipt, "no-op observed an opportunity that overlap did not")
        return {"first_opportunity": None, "overlap_matched_noop": True}
    require(noop_receipt, "overlap triggered without a no-op opportunity")
    require(noop_receipt[0] == overlap_receipt[0], "first opportunity differs from no-op")
    return {"first_opportunity": overlap_receipt[0], "overlap_matched_noop": True}


def run(args):
    registration = json.loads(Path(args.registration).read_bytes())
    static = json.loads((Path(args.work)/"static_cases.json").read_bytes())
    check_registration(args, registration, static)
    check_idle(args.deadline)
    require(args.deadline - time.time() < registration["maximum_wall_seconds"],
            "deadline is farther than the registered wall")
    work_root = str(Path(args.work).resolve())
    source_root = str(Path(args.src).resolve())
    sys.path[:] = [work_root, source_root, "/gs/env/fixed-baseline-deps"] + [
        path for path in sys.path[1:]
        if path not in (work_root, source_root, "/gs/env/fixed-baseline-deps") and
           "/gs/history-repo" not in path
    ]
    from bdlm_serving.methods.optimus import fixed, upstream
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from experiments.t2t.serving_slots import ibs_prefix_pair as pair
    from experiments.t2t.serving_slots import shared_llada_worker as worker
    import torch
    import transformers
    require(str(torch.__version__) == "2.6.0+cu124" and transformers.__version__ == "4.57.1",
            "unexpected pinned runtime")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "expected one GPU")
    tasks = load_tasks(Path(args.samples))
    require([task["index"] for task in tasks] == list(range(90)), "wrong frozen ordering")
    work_controls = Path(args.work)/"controls"/"Optimus"
    upstream.control_directory = lambda name: work_controls
    close_overlap_spec = importlib.util.spec_from_file_location(
        "tail_overlap_close_overlap", Path(args.work)/"close_overlap.py")
    close_overlap = importlib.util.module_from_spec(close_overlap_spec)
    close_overlap_spec.loader.exec_module(close_overlap)
    import bdlm_serving.methods.t2t as _tail_t2t
    sys.modules["bdlm_serving.methods.t2t.close_overlap"] = close_overlap
    _tail_t2t.close_overlap = close_overlap
    import fused
    require(sha(Path(fused.__file__).read_bytes()) == registration["sources"]["fused.py"],
            "fused hook drift")
    handle = worker.load(Path(args.checkpoint))
    print("model loaded on A1", file=sys.stderr, flush=True)
    registration_sha = identity(registration)
    runtime = {"torch": torch.__version__, "transformers": transformers.__version__,
               "gpu": torch.cuda.get_device_name(), "host": socket.gethostname()}
    total = 0
    for ordinal, index in enumerate(INDICES):
        if time.time() >= args.deadline - 120:
            print(json.dumps({"stopped": "deadline", "index": index, "completed": total,
                              "at": time.time()}), flush=True)
            return
        task = tasks[index]
        prompt, _how = local_prompt(handle["tok"], task, static["cases"][str(index)])
        rows = {}
        receipts = {"noop": [], "overlap": []}
        for arm in ROTATION[ordinal]:
            if time.time() >= args.deadline - 120:
                print(json.dumps({"stopped": "deadline", "index": index, "arm": arm,
                                  "completed": total, "at": time.time()}), flush=True)
                return
            check_idle_requested = False
            if arm == "stock":
                context = nullcontext()
            else:
                patched = fused.install_run(fixed, arm=arm, receipt=receipts[arm])
                context = patch.object(fixed.FixedOptimus, "run", patched)
            with context:
                row = pair.generate(handle["model"], handle["tok"], task, prompt, CAP, 0)
            row.update(arm=arm, q=Q, block=B, cap=CAP, index_ordinal=ordinal,
                       host=socket.gethostname(), gpu_uuid=A1_UUID,
                       gpu=torch.cuda.get_device_name(), worker_pid=os.getpid(),
                       samples_sha256=SAMPLES_SHA256,
                       prompt_ids_sha256=static["cases"][str(index)]["prompt_ids_sha256"],
                       registration_sha256=registration_sha, runtime=runtime)
            rows[arm] = row
            total += 1
            print(f"SEAL {index:02d} {arm:7s} status={row['status']} "
                  f"N/F/Q={row['output_tokens']}/{row['generation_forwards']}/"
                  f"{row['generation_query_rows']} "
                  f"max_q={row['max_q']}", file=sys.stderr, flush=True)
        overlap_audit = audit_pair(rows["stock"], rows["noop"], rows["overlap"], index,
                                   handle["mask_id"])
        audit_receipts(receipts["noop"], receipts["overlap"])
        rows["stock"]["historical_stock_match"] = None
        rows["noop"]["stock_noop_actions_equal"] = True
        rows["noop"]["noop_vote_receipt"] = receipts["noop"]
        rows["overlap"]["overlap_vote_receipt"] = receipts["overlap"]
        rows["overlap"]["overlap_audit"] = overlap_audit
        for arm in ARMS:
            row = dict(rows[arm])
            row["registration_sha256"] = registration_sha
            print(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                             allow_nan=False), flush=True)
    require(total == 18 and time.time() < args.deadline, "incomplete or late run")
    print(json.dumps({"complete": True, "rows": total, "at": time.time()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    for name in ("samples", "checkpoint", "work", "registration"):
        pre.add_argument("--" + name, required=True, type=Path)
    pre.add_argument("--src", default="/gs/code/formal90-7a38ffc/src", type=Path)
    pre.add_argument("--deadline", required=True, type=int)
    pre.set_defaults(func=preflight)
    runner = sub.add_parser("run")
    for name in ("samples", "checkpoint", "work", "registration"):
        runner.add_argument("--" + name, required=True, type=Path)
    runner.add_argument("--src", default="/gs/code/formal90-7a38ffc/src", type=Path)
    runner.add_argument("--deadline", required=True, type=int)
    runner.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

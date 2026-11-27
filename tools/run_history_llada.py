"""Fixed-C batched LLaDA2.1 block-history driver (no tuning matrix).

Examples (GPU host only; this file itself never runs a model on import):

  HISTORY_PREPARE=/path/_stage.py DG_SIGNAL=/path/signal \
    python tools/run_history_llada.py --checkpoint /models/llada21 \
      --samples /data/frozen30/samples.json --reference /data/refs30 \
      --indices 0-29 --gate history --C 8 --batch 4 --bucket exact \
      --output /data/run/history-c8-b4

  # Native control (SerialKernel, batch one) and the full-read endpoint:
  python tools/run_history_llada.py ... --gate native --output .../native
  python tools/run_history_llada.py ... --gate full --C 32 --batch 4 \
      --bucket exact --output .../full-c32
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import runpy
from statistics import mean
import sys

import torch
import transformers.utils as utils

if not hasattr(utils, "LossKwargs"):
    from typing import TypedDict

    class LossKwargs(TypedDict, total=False):
        num_items_in_batch: int

    utils.LossKwargs = LossKwargs

from transformers import AutoModelForCausalLM, AutoTokenizer

from bdlm_serving.evaluation.frozen import load_tasks
from bdlm_serving.methods.t2t.history_llada_batch import (
    BLOCK, NATIVE_CONFIG, BatchedHistoryRunner, HistoryStore, make_request,
    package_request, summarize)
from bdlm_serving.runtime.serial import (
    SerialKernel, install_llada_rectangular_mask)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_MANIFEST = ROOT / "benchmarks/checkpoints/llada21.json"


def parse_indices(values):
    result = []
    for token in values:
        if "-" in token:
            a, b = token.split("-", 1)
            if not a or not b:
                raise argparse.ArgumentTypeError(f"bad index range {token}")
            result.extend(range(int(a), int(b) + 1))
        else:
            result.append(int(token))
    if len(set(result)) != len(result) or any(i < 0 for i in result):
        raise argparse.ArgumentTypeError("indices must be distinct and non-negative")
    return sorted(result)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--samples", required=True, type=Path)
    p.add_argument("--reference", required=True, type=Path,
                   help="directory containing {index:02d}-native32.json")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--indices", nargs="+", help="sample indices; default is the entire selected suite")
    p.add_argument("--suite", choices=("frozen30", "frozen90"), default="frozen90")
    p.add_argument("--gate", choices=("native", "full", "history"), required=True,
                   help="native: SerialKernel batch one; full: C>=32 endpoint; "
                        "history: fixed sparse C")
    p.add_argument("--C", type=int, help="fixed nominal history Q; required for "
                                         "full/history; 32 or 64 are the same full endpoint")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--bucket", choices=("exact", "pooled"), default="exact",
                   help="exact: equal committed length per group, no KV padding; "
                        "pooled: mixed lengths with a rectangular mask")
    p.add_argument("--prepare-script", type=Path,
                   help="optional module with prepare(model); also read from HISTORY_PREPARE")
    p.add_argument("--max-rounds", type=int, default=20000)
    p.add_argument("--skip-checkpoint-audit", action="store_true")
    return p.parse_args()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def durable(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("x") as stream:
        json.dump(data, stream, ensure_ascii=False)
    temp.replace(path)


def audit_checkpoint(path):
    entries = json.loads(CHECKPOINT_MANIFEST.read_text())
    for entry in entries:
        target = Path(path) / entry["name"]
        if not target.is_file() or target.stat().st_size != entry["bytes"]:
            raise ValueError(f"checkpoint size/file mismatch: {target}")
        if "sha256" in entry and sha(target) != entry["sha256"]:
            raise ValueError(f"checkpoint sha mismatch: {target}")


def load_model(args):
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    opts = dict(trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, **opts)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, **opts, dtype=torch.bfloat16,
        attn_implementation="sdpa").eval()
    install_llada_rectangular_mask(model, allow_selected_cache=True)
    prepare = args.prepare_script or (Path(os.environ["HISTORY_PREPARE"])
                                     if "HISTORY_PREPARE" in os.environ else None)
    if "DG_SIGNAL" in os.environ or prepare is not None:
        if prepare is None:
            raise ValueError("DG_SIGNAL is set but HISTORY_PREPARE/--prepare-script is missing")
        # Per-request RNG and KV are private; never share parameter storage
        # with the staging helper's temporary views.
        for parameter in model.parameters():
            parameter.data = parameter.data.clone()
        runpy.run_path(str(prepare), run_name="bdlm_prepare")["prepare"](model)
    else:
        model.to("cuda")
    if model.device.type != "cuda":
        raise RuntimeError("refusing CPU inference; model must be on CUDA")
    return tokenizer, model


def load_reference(args, tokenizer, task):
    index = task["index"]
    path = args.reference / f"{index:02d}-native32.json"
    if not path.is_file():
        raise ValueError(f"missing native reference: {path}")
    ref = json.loads(path.read_text())
    prompt = tokenizer.apply_chat_template(
        [dict(role="user", content=task["prompt"])],
        tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, return_tensors="pt",
                    add_special_tokens=False).input_ids[0].tolist()
    if ids != ref["prompt_ids"]:
        raise ValueError(f"tokenizer/prompt disagrees with frozen refs on {index}")
    if "seed" in ref and int(ref["seed"]) != int(task["seed"]):
        raise ValueError(f"seed disagrees with frozen refs on {index}")
    cap = int(ref.get("generation_cap", 2048))
    if cap != 2048:
        raise ValueError(f"unexpected frozen generation cap {cap} on {index}")
    return ref, ids


def tasks_by_index(args):
    if args.suite == "frozen30":
        tasks = load_tasks(args.samples)
    else:
        from bdlm_serving.evaluation.frozen90 import load_tasks as load_tasks90
        tasks = load_tasks90(args.samples)
    by_index = {int(task["index"]): task for task in tasks}
    indices = sorted(by_index) if args.indices is None else parse_indices(args.indices)
    missing = [i for i in indices if i not in by_index]
    if missing:
        raise ValueError(f"indices not in {args.suite}: {missing}")
    return [by_index[i] for i in indices]


def package_native(tokenizer, state, special_ids):
    ids = SerialKernel.output(state)[0].tolist()
    eos_found = any(token in state.eos_ids for token in ids)
    n = sum(token not in special_ids for token in ids)
    events = state.events
    def width(event):
        return int(event.get("physical_q", event["q_tokens"]))
    work = [(event["phase"], width(event))
            for event in events if event["phase"] != "prefill"]
    if not work:
        raise RuntimeError("native request produced no generation forward")
    f = len(work)
    q = sum(rows for _, rows in work)
    phases = Counter(event["phase"] for event in events)
    phase_q = Counter()
    for event in events:
        phase_q[event["phase"]] += width(event)
    return dict(
        status="success", index=state.index if hasattr(state, "index") else None,
        output_ids=ids, output_text=tokenizer.decode(ids, skip_special_tokens=True),
        output_tokens=n, generation_forwards=f, generation_query_rows=q,
        generation_physical_padded_rows=q, max_q=max(rows for _, rows in work),
        tpf=n / f, token_efficiency=n / q, eos=eos_found, truncated=not eos_found,
        prompt_tokens=state.prompt_length,
        prefill_rows=sum(width(event) for event in events
                         if event["phase"] == "prefill"),
        forwards_by_phase=dict(phases), q_by_phase=dict(phase_q),
        q_histogram=dict(Counter(str(rows) for _, rows in work)),
        history_rows=0, denoise_history_rows=0, commit_history_rows=0,
        history_kv_edges=0, block_attention_edges=0, stale_reused_rows=0,
        history_q_per_forward=[], decoder_q_per_forward=[rows for _, rows in work],
        dirty_wait=[], cold_wait=[], stale_age=[],
        batch_sizes=[1], events=events,
    )


def run_native(args, tokenizer, model, tasks, tag):
    kernel = SerialKernel(model, "llada21")
    mask = tokenizer.convert_tokens_to_ids("<|mask|>")
    eos = [tokenizer.eos_token_id]
    special = set(tokenizer.all_special_ids) | set(eos) | {mask}
    rows = []
    for task, ref, prompt_ids in tasks:
        state = kernel.create(
            torch.tensor([prompt_ids], device=model.device), NATIVE_CONFIG,
            ref.get("generation_cap", 2048), task["seed"], mask, eos)
        kernel.run(state)
        row = package_native(tokenizer, state, special)
        row.update(index=task["index"], task_id=task["task_id"],
                   dataset=task["dataset"], seed=task["seed"],
                   original30=task["index"] < 30)
        durable(args.output / f"{task['index']:02d}-{tag}.json", row)
        rows.append(row)
        print(f"{tag} {task['index']:02d} N={row['output_tokens']} "
              f"F={row['generation_forwards']} Q={row['generation_query_rows']}",
              flush=True)
    return rows


@torch.inference_mode()
def run_history(args, tokenizer, model, tasks, tag, full_endpoint):
    from bdlm_serving.methods.t2t.history_llada import LLadaHistory
    mask = tokenizer.convert_tokens_to_ids("<|mask|>")
    eos = [tokenizer.eos_token_id]
    special = set(tokenizer.all_special_ids) | set(eos) | {mask}
    adapter = LLadaHistory(model)
    rows = []
    try:
        # Same single request, same backend, same committed prompt. This gate
        # runs before the scored work and must not change any measured request.
        kernel = SerialKernel(model,"llada21")
        for task,ref,ids in tasks:
            if len(ids) < BLOCK:
                continue
            plain = kernel.create(torch.tensor([ids],device=model.device),NATIVE_CONFIG,
                                  ref["generation_cap"],task["seed"],mask,eos)
            req = make_request(model,task["index"],dict(task,_mask_id=mask,_eos_ids=eos),
                               ids,generation_cap=ref["generation_cap"])
            kernel.step(plain)
            warm = BatchedHistoryRunner(model,adapter,[req],budget=args.C,batch_size=1)
            warm._prefill([req])
            if len(plain.cache.layers) != len(req.store.keys):
                raise RuntimeError("prefill layer count differs from native")
            for i,layer in enumerate(plain.cache.layers):
                if not (torch.equal(layer.keys,req.store.keys[i])
                        and torch.equal(layer.values,req.store.values[i])):
                    raise RuntimeError(f"prefill differs from native on layer {i}")
            print(f"prefill parity passed: request {task['index']}, {req.store.length} tokens, "
                  f"{len(plain.cache.layers)} layers, exact native KV",flush=True)
            del warm,req,plain,layer
            break
        with (args.output / "trace.jsonl").open("x") as stream:
            offset = 0
            for start in range(0, len(tasks), args.batch):
                requests = [make_request(model, task["index"], dict(task, _mask_id=mask, _eos_ids=eos),
                                         ids, generation_cap=ref["generation_cap"])
                            for task, ref, ids in tasks[start:start + args.batch]]
                runner = BatchedHistoryRunner(model, adapter, requests, budget=args.C,
                                              batch_size=args.batch, bucket=args.bucket,
                                              full_endpoint=full_endpoint, max_rounds=args.max_rounds)
                runner.run()
                for event in runner.events:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                stream.flush()
                for request in requests:
                    row = package_request(request, tokenizer, special)
                    for event in row["events"]:
                        if "trace" in event:
                            event["trace"] += offset
                    durable(args.output / f"{request.index:02d}-{tag}.json", row)
                    print(f"{tag} {request.index:02d} N={row['output_tokens']} "
                          f"F={row['generation_forwards']} Q={row['generation_query_rows']} "
                          f"H={row['history_rows']}", flush=True)
                    rows.append({k:v for k,v in row.items() if k not in ("events","output_text","output_ids")})
                offset += len(runner.events)
                del runner, requests, request, row
    finally:
        adapter.close()
    return rows


def main():
    args = parse_args()
    if args.gate == "history":
        if args.C is None or not 1 <= args.C <= 2 * BLOCK:
            raise ValueError("--gate history requires 1 <= C <= 64")
        # C >= B is the full-read endpoint by definition; C == 2B is the same
        # single-block endpoint, not an independent second-block result.
        full_endpoint = args.C >= BLOCK
        effective = min(args.C, BLOCK)
    elif args.gate == "full":
        if args.C is None:
            args.C = BLOCK
        if not BLOCK <= args.C <= 2 * BLOCK:
            raise ValueError("--gate full requires 32 <= C <= 64 (same B-full endpoint)")
        full_endpoint = True
        effective = BLOCK
    else:
        full_endpoint = False
        effective = None
    if args.batch < 1:
        raise ValueError("--batch must be >= 1")
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    if not args.skip_checkpoint_audit:
        audit_checkpoint(args.checkpoint)
    tasks = tasks_by_index(args)
    args.output.mkdir(parents=True)
    tag = ("native" if args.gate == "native" else
           f"{args.gate}-c{args.C}-b{args.batch}-{args.bucket}")
    tokenizer, model = load_model(args)
    references = [load_reference(args, tokenizer, task) for task in tasks]
    paired = [(task, ref, ids) for task, (ref, ids) in zip(tasks, references)]
    metadata = dict(
        schema="llada-block-history-v1", model="llada21",
        checkpoint=str(args.checkpoint), gate=args.gate,
        nominal_C=args.C, effective_C=effective,
        batch=(1 if args.gate == "native" else args.batch),
        bucket=args.bucket, suite=args.suite, indices=[t["index"] for t in tasks],
        samples=str(args.samples), samples_sha256=sha(args.samples),
        reference=str(args.reference), protocol=dict(NATIVE_CONFIG),
        generation_cap=2048, device=torch.cuda.get_device_name(model.device),
        torch=str(torch.__version__),
        visible_device=os.environ.get("CUDA_VISIBLE_DEVICES"),
        prepare=("history_prepare" if ("DG_SIGNAL" in os.environ or
                                       args.prepare_script) else "model.to(cuda)"),
        full_endpoint=full_endpoint,
        endpoint_same_as_b=bool(full_endpoint and args.C is not None and args.C > BLOCK),
        note=("native is SerialKernel batch one; full/history rows count every "
              "projected block row (32) and separately every history re-read row"),
    )
    durable(args.output / "run.json", metadata)
    if args.gate == "native":
        rows = run_native(args, tokenizer, model, paired, tag)
    else:
        rows = run_history(args, tokenizer, model, paired, tag, full_endpoint)
    durable(args.output / "summary.json", summarize(rows))
    durable(args.output / "complete.json",
            dict(tag=tag, requests=len(rows), gate=args.gate,
                 nominal_C=args.C, batch=args.batch, bucket=args.bucket))
    print(json.dumps(summarize(rows)["all"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

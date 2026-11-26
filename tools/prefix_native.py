"""Build and verify the frozen90 native-reference bundle for the prefix study.

CPU-only commands:
  prepare  verify the original 30 native references, preserve them byte-for-byte,
           and record the added 60 as pending.
  verify   verify all 90 references after capture.

GPU-only command (call only through the protected n164 launcher):
  capture  run the actual native backend (native32 / native256) on the added 60,
           verify each reference, and merge it into the bundle.  No sparse policy
           is imported or executed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time

SCHEMA = 1
STUDY = "frozen90-prefix-native-v1"
OLD = 30
TOTAL = 90
MODELS = {"llada21": ("native32", 32), "gemma": ("native256", 256)}
DEFAULT_LLADA_REFERENCE = Path(
    "/data1/project_data/gx-erqw-20261123/results/"
    "fixed-baselines-llada-gpu4-v4/requests")
DEFAULT_GEMMA_REFERENCE = Path(
    "/data1/project_data/gx-erqw-20261123/runtime/"
    "diffusiongemma-native-20261125/run-v2")
DEFAULT_LLADA_CHECKPOINT = Path("/data0/ysy/models/LLaDA2.1-mini")
DEFAULT_GEMMA_CHECKPOINT = Path(
    "/data1/project_data/gx-erqw-20261123/models/diffusiongemma-cd40195")

LLADA_REQUIRED = frozenset({
    "status", "model", "policy", "task_id", "index", "dataset", "seed",
    "phase", "generation_cap", "prompt_ids", "output_ids", "output_text",
    "events", "calls", "extra", "output_tokens", "generation_forwards",
    "generation_query_rows", "generation_physical_padded_rows", "prefill_rows",
    "max_q", "q_histogram", "tpf", "token_efficiency",
    "physical_token_efficiency", "eos", "truncated", "remaining_masks",
    "e2e_shared_gpu_wall_ms", "t2t_edits", "request_failure",
})
GEMMA_REQUIRED = frozenset({
    "index", "dataset", "task_id", "seed", "output_tokens",
    "generation_forwards", "generation_query_rows",
    "generation_physical_padded_rows", "tpf", "token_efficiency", "max_q",
    "prefill_rows", "decoder_forwards", "decoder_query_rows",
    "reencode_forwards", "reencode_query_rows", "decoder_steps_per_canvas",
    "native_nonpad", "native_tpf", "eos", "truncated", "model", "policy",
    "status", "output_text", "output_ids", "events",
})


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def check_cohort(tasks):
    """Require indices 0..89 and byte-identical identity for the original 30."""
    if len(tasks) != TOTAL or [t["index"] for t in tasks] != list(range(TOTAL)):
        raise ValueError("frozen90 must contain indices 0..89 in order")
    from bdlm_serving.evaluation import frozen

    original = json.loads(frozen.MANIFEST.read_text())["samples"]

    def entry(task):
        return dict(index=task["index"], source_index=task["source_index"],
                    dataset=task["dataset"], task_id=task["task_id"],
                    seed=task["seed"],
                    prompt_sha256=hashlib.sha256(task["prompt"].encode()).hexdigest())

    if [entry(t) for t in tasks[:OLD]] != original:
        raise ValueError("original 0..29 identity changed")
    return tasks


def load_frozen90(path):
    from bdlm_serving.evaluation.frozen90 import load_tasks

    tasks = load_tasks(Path(path))
    return check_cohort(tasks)


def split_added(tasks):
    added = list(tasks[OLD:])
    if [t["index"] for t in added] != list(range(OLD, TOTAL)):
        raise ValueError("added60 indices are not 30..89")
    return added


def _metadata(task, ref):
    for key in ("index", "task_id", "dataset", "seed"):
        if ref[key] != task[key]:
            raise ValueError(f"metadata mismatch on {task['index']}: {key}")


def _counters(ref):
    return tuple(ref[k] for k in (
        "output_tokens", "generation_forwards", "generation_query_rows",
        "generation_physical_padded_rows"))


def _eos_check(ref, eos):
    stopped = any(token in eos for token in ref["output_ids"])
    if bool(ref["eos"]) != stopped or bool(ref["truncated"]) == stopped:
        raise ValueError(f"EOS/truncated disagreement on {ref['index']}")


def verify_llada(task, ref, *, prompt_ids=None, special=None, eos=None,
                 mask=None, decode=None):
    """Verify one native32 reference without running a model."""
    missing = LLADA_REQUIRED - ref.keys()
    if missing:
        raise ValueError(f"missing LLaDA fields on {task['index']}: {sorted(missing)}")
    _metadata(task, ref)
    if ref["model"] != "llada21" or ref["policy"] != "native32":
        raise ValueError(f"not a native32 LLaDA reference on {task['index']}")
    if ref.get("extra") != {"native_serial": True}:
        raise ValueError(f"not the serial native backend on {task['index']}")
    if any(key in ref for key in ("forwards_by_phase", "q_by_phase",
                                  "physical_events", "selection", "arm")):
        raise ValueError(f"sparse policy fields on {task['index']}")
    if ref["status"] != "success" or ref.get("request_failure") is not None:
        raise ValueError(f"failed request on {task['index']}")
    if prompt_ids is not None and ref["prompt_ids"] != prompt_ids:
        raise ValueError(f"prompt ids changed on {task['index']}")
    n, f, q, physical = _counters(ref)
    if f <= 0 or q != physical or q < f or n < 0:
        raise ValueError(f"bad N/F/Q on {task['index']}: {(n, f, q, physical)}")
    calls, events = ref["calls"], ref["events"]
    if len(calls) != len(events) or len(calls) < 2:
        raise ValueError(f"call/event mismatch on {task['index']}")
    if events[0].get("phase") != "prefill" or calls[0]["rows"] != ref["prefill_rows"]:
        raise ValueError(f"prefill mismatch on {task['index']}")
    if any(e.get("q_tokens") != c["rows"] for e, c in zip(events, calls)):
        raise ValueError(f"event/call rows disagree on {task['index']}")
    if f != len(calls) - 1:
        raise ValueError(f"forward count disagrees on {task['index']}")
    if q != sum(c["rows"] for c in calls[1:]):
        raise ValueError(f"query rows disagree on {task['index']}")
    if physical != sum(c["physical_rows"] for c in calls[1:]):
        raise ValueError(f"physical rows disagree on {task['index']}")
    if ref["max_q"] != max(c["rows"] for c in calls[1:]):
        raise ValueError(f"max_q disagrees on {task['index']}")
    histogram = {str(k): v for k, v in
                 Counter(str(c["rows"]) for c in calls[1:]).items()}
    if ref["q_histogram"] != histogram:
        raise ValueError(f"q_histogram disagrees on {task['index']}")
    if abs(ref["tpf"] - n / f) > 1e-10 or abs(ref["token_efficiency"] - n / q) > 1e-10:
        raise ValueError(f"TPF/token efficiency disagree on {task['index']}")
    if abs(ref["physical_token_efficiency"] - n / physical) > 1e-10:
        raise ValueError(f"physical token efficiency disagrees on {task['index']}")
    if special is not None and sum(t not in special for t in ref["output_ids"]) != n:
        raise ValueError(f"output_tokens disagrees on {task['index']}")
    if eos is not None:
        _eos_check(ref, eos)
    if mask is not None and ref["remaining_masks"] != ref["output_ids"].count(mask):
        raise ValueError(f"remaining_masks disagrees on {task['index']}")
    if decode is not None and ref["output_text"] != decode(ref["output_ids"]):
        raise ValueError(f"output_text disagrees on {task['index']}")
    return ref


def verify_gemma(task, ref, *, prompt_length=None, special=None, eos=None,
                 pad=0, decode=None):
    """Verify one native256 reference without running a model."""
    missing = GEMMA_REQUIRED - ref.keys()
    if missing:
        raise ValueError(f"missing Gemma fields on {task['index']}: {sorted(missing)}")
    _metadata(task, ref)
    if ref["model"] != "diffusiongemma" or ref["policy"] != "native256":
        raise ValueError(f"not a native256 DiffusionGemma reference on {task['index']}")
    if any(key in ref for key in ("forwards_by_phase", "q_by_phase",
                                  "physical_events", "selection", "arm", "calls")):
        raise ValueError(f"sparse policy fields on {task['index']}")
    if ref["status"] != "success":
        raise ValueError(f"failed request on {task['index']}")
    n, f, q, physical = _counters(ref)
    decoder, reencode = ref["decoder_forwards"], ref["reencode_forwards"]
    if f <= 0 or f != decoder + reencode or q != 256 * f or physical != q:
        raise ValueError(f"bad N/F/Q on {task['index']}: {(n, f, q, physical)}")
    if ref["decoder_query_rows"] != 256 * decoder:
        raise ValueError(f"decoder_query_rows disagree on {task['index']}")
    if ref["reencode_query_rows"] != 256 * reencode:
        raise ValueError(f"reencode_query_rows disagree on {task['index']}")
    if ref["max_q"] != 256:
        raise ValueError(f"max_q disagrees on {task['index']}")
    if prompt_length is not None and ref["prefill_rows"] != prompt_length:
        raise ValueError(f"prefill_rows disagree on {task['index']}")
    events = ref["events"]
    if not events or events[0] != ["encoder", ref["prefill_rows"]]:
        raise ValueError(f"prefill event disagrees on {task['index']}")
    if len(events) != 1 + f:
        raise ValueError(f"event count disagrees on {task['index']}")
    if any(len(e) != 2 or e[0] not in ("encoder", "decoder") or e[1] != 256
           for e in events[1:]):
        raise ValueError(f"physical event disagrees on {task['index']}")
    if sum(e[0] == "decoder" for e in events[1:]) != decoder:
        raise ValueError(f"decoder events disagree on {task['index']}")
    if sum(e[0] == "encoder" for e in events[1:]) != reencode:
        raise ValueError(f"reencode events disagree on {task['index']}")
    steps = ref["decoder_steps_per_canvas"]
    if (len(steps) != reencode + 1 or sum(steps) != decoder
            or any(not isinstance(s, int) or s <= 0 for s in steps)):
        raise ValueError(f"decoder_steps_per_canvas disagrees on {task['index']}")
    if ref["native_nonpad"] != len(ref["output_ids"]):
        raise ValueError(f"native_nonpad disagrees on {task['index']}")
    ratio = ref["native_nonpad"] / sum(steps)
    if abs(ref["native_tpf"] - ratio) > 1e-5 * max(1.0, abs(ratio)):
        raise ValueError(f"native_tpf disagrees on {task['index']}")
    if abs(ref["tpf"] - n / f) > 1e-10 or abs(ref["token_efficiency"] - n / q) > 1e-10:
        raise ValueError(f"TPF/token efficiency disagree on {task['index']}")
    if special is not None and sum(t not in special for t in ref["output_ids"]) != n:
        raise ValueError(f"output_tokens disagrees on {task['index']}")
    if eos is not None:
        _eos_check(ref, eos)
    if decode is not None and ref["output_text"] != decode(ref["output_ids"]):
        raise ValueError(f"output_text disagrees on {task['index']}")
    return ref


def llada_prompt_ids(tokenizer, task):
    text = tokenizer.apply_chat_template(
        [dict(role="user", content=task["prompt"])],
        tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt",
                     add_special_tokens=False).input_ids[0].tolist()


def gemma_prompt_length(processor, task):
    data = processor.apply_chat_template(
        [dict(role="user", content=task["prompt"])],
        tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt")
    return int(data["input_ids"].shape[1])


def load_llada_tools(checkpoint):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=True, local_files_only=True)
    mask = tokenizer.convert_tokens_to_ids("<|mask|>")
    eos = [tokenizer.eos_token_id]
    special = set(tokenizer.all_special_ids) | {mask} | set(eos)
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=True)
    return tokenizer, mask, eos, special, decode


def load_gemma_tools(checkpoint):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    tokenizer = processor.tokenizer
    config = read_json(Path(checkpoint) / "generation_config.json")
    raw = config["eos_token_id"]
    eos = set(raw if isinstance(raw, list) else [raw])
    pad = config.get("pad_token_id", 0)
    special = set(tokenizer.all_special_ids) | eos | {pad}
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=True)
    return processor, eos, pad, special, decode


def copy_verified(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copy2(source, destination)
    before, after = sha256(source), sha256(destination)
    if before != after:
        raise ValueError(f"copy changed bytes: {source}")
    return after


def _record(task, source, destination, ref):
    return dict(index=task["index"], task_id=task["task_id"],
                dataset=task["dataset"], seed=task["seed"],
                source=str(source), source_sha256=sha256(source),
                copy=str(destination), copy_sha256=sha256(destination),
                policy=ref["policy"], output_tokens=ref["output_tokens"],
                generation_forwards=ref["generation_forwards"],
                generation_query_rows=ref["generation_query_rows"])


def prepare(args):
    """CPU-only: verify and preserve the old30; leave added60 pending."""
    tasks = load_frozen90(args.samples)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    (output / "llada21").mkdir(parents=True)
    (output / "gemma").mkdir(parents=True)

    tokenizer, mask, eos, special, decode = load_llada_tools(args.llada_checkpoint)
    processor, geos, pad, gspecial, gdecode = load_gemma_tools(args.gemma_checkpoint)

    records = {"llada21": [], "gemma": []}
    for task in tasks[:OLD]:
        index = task["index"]
        source = Path(args.llada_reference) / f"{index:02d}-native32.json"
        ref = verify_llada(
            task, read_json(source), prompt_ids=llada_prompt_ids(tokenizer, task),
            special=special, eos=eos, mask=mask, decode=decode)
        destination = output / "llada21" / source.name
        copy_verified(source, destination)
        records["llada21"].append(_record(task, source, destination, ref))
    for task in tasks[:OLD]:
        index = task["index"]
        source = Path(args.gemma_reference) / f"{index:02d}-native256.json"
        ref = verify_gemma(
            task, read_json(source),
            prompt_length=gemma_prompt_length(processor, task),
            special=gspecial, eos=geos, pad=pad, decode=gdecode)
        destination = output / "gemma" / source.name
        copy_verified(source, destination)
        records["gemma"].append(_record(task, source, destination, ref))

    llada_policy, llada_width = MODELS["llada21"]
    gemma_policy, gemma_width = MODELS["gemma"]
    manifest = dict(
        schema=SCHEMA, study=STUDY, samples=str(Path(args.samples).resolve()),
        samples_sha256=sha256(args.samples), tasks=TOTAL,
        original30=dict(count=OLD, verified=True,
                        llada_reference=str(Path(args.llada_reference).resolve()),
                        gemma_reference=str(Path(args.gemma_reference).resolve())),
        added60=dict(count=TOTAL - OLD, status="pending",
                     indices=[OLD, TOTAL - 1]),
        models={
            "llada21": dict(policy=llada_policy, width=llada_width,
                            checkpoint=str(Path(args.llada_checkpoint).resolve())),
            "gemma": dict(policy=gemma_policy, width=gemma_width,
                          checkpoint=str(Path(args.gemma_checkpoint).resolve())),
        },
        created=time.time(),
    )
    write_json(output / "manifest.json", manifest)
    write_json(output / "prepared.json", dict(
        schema=SCHEMA, study=STUDY, status="prepared",
        original30=records, added60=dict(status="pending", count=TOTAL - OLD),
        created=time.time()))
    return records


def _staging(args, model):
    if getattr(args, "staging", None):
        return Path(args.staging)
    return Path(args.output) / f"capture-{model}"


def capture_llada(args, tasks, staging, prepare):
    from bdlm_serving import generation

    plan = dict(policies=["native32"], mix_policies=False, batch_size=1)
    namespace = argparse.Namespace(
        model="llada21", checkpoint=args.checkpoint, output=staging,
        experiment="baselines", batch_size=1, capture_traces=True)

    def staged(model):
        import torch

        for parameter in model.parameters():
            parameter.data = parameter.data.clone()
        prepare(model)

    generation.run(namespace, tasks, plan,
                   before_cuda=staged if prepare is not None else None)


def capture_gemma(args, tasks, staging, prepare):
    import bdlm_serving.runtime.diffusiongemma as native

    original = native.load_tasks
    native.load_tasks = lambda path, seed_offset=0: list(tasks)
    try:
        native.run_native(
            argparse.Namespace(checkpoint=args.checkpoint, samples=args.samples,
                               output=staging),
            before_cuda=prepare)
    finally:
        native.load_tasks = original


def capture(args, prepare=None):
    """GPU-only: native capture of added60, verify, and merge into the bundle."""
    tasks = load_frozen90(args.samples)
    added = split_added(tasks)
    model = args.model
    policy, _ = MODELS[model]
    output = Path(args.output)
    if not (output / "prepared.json").is_file():
        raise FileNotFoundError("run prepare before capture")
    for task in added:
        if (output / model / f"{task['index']:02d}-{policy}.json").exists():
            raise FileExistsError(f"added reference already exists: {task['index']}")
    staging = _staging(args, model)
    if staging.exists():
        raise FileExistsError(staging)
    if model == "llada21":
        staging.mkdir(parents=True, exist_ok=False)
        capture_llada(args, added, staging, prepare)
    else:
        capture_gemma(args, added, staging, prepare)

    if model == "llada21":
        tokenizer, mask, eos, special, decode = load_llada_tools(args.checkpoint)
    else:
        processor, geos, pad, gspecial, gdecode = load_gemma_tools(args.checkpoint)
    records = []
    for task in added:
        index = task["index"]
        source = staging / f"{index:02d}-{policy}.json"
        ref = read_json(source)
        if model == "llada21":
            verify_llada(
                task, ref, prompt_ids=llada_prompt_ids(tokenizer, task),
                special=special, eos=eos, mask=mask, decode=decode)
        else:
            verify_gemma(
                task, ref, prompt_length=gemma_prompt_length(processor, task),
                special=gspecial, eos=geos, pad=pad, decode=gdecode)
        destination = output / model / source.name
        copy_verified(source, destination)
        records.append(_record(task, source, destination, ref))
    write_json(output / f"capture-{model}.json", dict(
        schema=SCHEMA, study=STUDY, model=model, policy=policy,
        status="captured", count=len(records), records=records,
        staging=str(staging.resolve()), created=time.time()))
    return records


def verify(args):
    """CPU-only: verify all 90 references and the byte-preserved old30."""
    tasks = load_frozen90(args.samples)
    output = Path(args.output)
    if not (output / "prepared.json").is_file():
        raise FileNotFoundError("run prepare before verify")
    tokenizer, mask, eos, special, decode = load_llada_tools(args.llada_checkpoint)
    processor, geos, pad, gspecial, gdecode = load_gemma_tools(args.gemma_checkpoint)
    report = dict(schema=SCHEMA, study=STUDY, models={})
    for model, verify_one in (("llada21", verify_llada),
                              ("gemma", verify_gemma)):
        policy, _ = MODELS[model]
        rows = []
        for position, task in enumerate(tasks):
            index = task["index"]
            path = output / model / f"{index:02d}-{policy}.json"
            ref = read_json(path)
            if model == "llada21":
                verify_one(
                    task, ref, prompt_ids=llada_prompt_ids(tokenizer, task),
                    special=special, eos=eos, mask=mask, decode=decode)
            else:
                verify_one(
                    task, ref, prompt_length=gemma_prompt_length(processor, task),
                    special=gspecial, eos=geos, pad=pad, decode=gdecode)
            if position < OLD:
                source = (Path(args.llada_reference) if model == "llada21"
                          else Path(args.gemma_reference)) / f"{index:02d}-{policy}.json"
                if sha256(path) != sha256(source):
                    raise ValueError(f"old30 bytes changed: {path}")
            rows.append(dict(index=index, task_id=task["task_id"],
                             seed=task["seed"], policy=ref["policy"],
                             output_tokens=ref["output_tokens"],
                             generation_forwards=ref["generation_forwards"],
                             generation_query_rows=ref["generation_query_rows"]))
        report["models"][model] = dict(policy=policy, requests=len(rows),
                                       original30_preserved=True, rows=rows)
    report["status"] = "verified"
    report["created"] = time.time()
    write_json(output / "verified.json", report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "capture", "verify"))
    parser.add_argument("--model", choices=tuple(MODELS))
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--staging", type=Path)
    parser.add_argument("--llada-reference", type=Path,
                        default=DEFAULT_LLADA_REFERENCE)
    parser.add_argument("--gemma-reference", type=Path, default=DEFAULT_GEMMA_REFERENCE)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--llada-checkpoint", type=Path, default=DEFAULT_LLADA_CHECKPOINT)
    parser.add_argument("--gemma-checkpoint", type=Path, default=DEFAULT_GEMMA_CHECKPOINT)
    args = parser.parse_args(argv)
    if args.command == "capture" and args.model is None:
        parser.error("capture requires --model")
    if args.checkpoint is None:
        args.checkpoint = (args.llada_checkpoint if args.model == "llada21"
                           else args.gemma_checkpoint)
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "capture":
        capture(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()

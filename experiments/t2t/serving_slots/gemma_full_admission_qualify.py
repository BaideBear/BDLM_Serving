"""One-shot A3 physical qualification for fixed Q64/Q128 full admission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

GPU = "GPU-94c61e79-d80b-cae2-d85f-6bd2a2c99a1c"
CORE = Path("experiments/t2t/serving_slots/gemma_full_admission.py")
PARENT_PROFILE = "dense_stop_joint_verified_eos"
RUNTIME = {"torch": "2.6.0+cu124", "transformers": "5.12.0", "dtype": "bf16",
           "attention": "sdpa", "experts": "grouped_mm"}
CASES = (0, 7, 13)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("x") as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.write("\n")
    tmp.rename(path)


def read(path):
    return json.loads(Path(path).read_bytes())


def guard(stage):
    apps = subprocess.check_output(
        ["nvidia-smi", "-i", GPU, "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"], text=True).strip()
    observed = subprocess.check_output(
        ["nvidia-smi", "-i", GPU, "--query-gpu=uuid,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
    uuid, memory, util = [x.strip() for x in observed.split(",")]
    if uuid != GPU or os.environ.get("CUDA_VISIBLE_DEVICES") != GPU:
        raise RuntimeError(f"A3 UUID mismatch: {uuid!r}")
    if stage == "before" and (memory != "0" or util != "0" or apps):
        raise RuntimeError(f"A3 not idle before launch: {memory} MiB / {util}%, apps={apps!r}")
    if stage != "before":
        expected = f"{os.getpid()}, 0"
        if apps != expected:
            raise RuntimeError(f"A3 occupancy drift at {stage}: {apps!r}")


def source_manifest(manifest, core, script):
    """Verify the immutable tracked-source closure staged beside this harness."""
    files = {}
    for line in Path(manifest).read_text().splitlines():
        item_sha, name = line.split(maxsplit=1)
        p = Path(name)
        if not p.is_file() or sha_file(p) != item_sha:
            raise RuntimeError(f"Qualification source drift: {name}")
        files[name] = item_sha
    # A module entered by -m supplies absolute __file__; bind it to the staged
    # relative source identity that is recorded in the manifest.
    pinned = tuple(Path(p).resolve().relative_to(Path.cwd()) for p in (core, script))
    for p in pinned:
        key = p.as_posix()
        if key not in files or sha_file(p) != files[key]:
            raise RuntimeError(f"Qualification harness missing from manifest: {key}")
    Path("source.sha256").write_text(
        "".join(f"{sha}  {name}\n" for name, sha in sorted(files.items())))
    return sha_file("source.sha256"), files

def origin(seal):
    device = seal["device"]
    if (device["lane"] != "A3" or device["gpu_uuid"] != GPU or
            device["physical_index"] != 3 or device["visible_index"] != 0 or
            not device["host"].strip() or
            len(seal["qualification_commit"]) != 40):
        raise RuntimeError("Qualification does not identify one A3 physical device")
    files = seal["source_files"]
    if files[CORE.as_posix()] != seal["executor_sha256"] or digest(files) != seal["qualification_source_sha256"]:
        raise RuntimeError("Qualification source binding drift")
    return {key: seal[key] for key in ("device", "qualification_commit",
                                       "qualification_source_sha256", "executor_sha256")}


def run_generic(model, proc, task, prior_prompt, opener, offer):
    import torch
    from bdlm_serving.evaluation.scoring import score
    from . import gemma_response_forks as live
    from .gemma_unified_audit import projectors
    data = proc.apply_chat_template([{"role": "user", "content": task["prompt"]}],
                                    tokenize=True, add_generation_prompt=True,
                                    return_dict=True, return_tensors="pt").to(model.device)
    prompt = live.digest(data["input_ids"][0].tolist())
    if prompt != prior_prompt:
        raise RuntimeError("Frozen input IDs drift")
    torch.manual_seed(task["seed"])
    torch.cuda.manual_seed(task["seed"])
    eos = set(model.generation_config.eos_token_id)
    special = set(proc.tokenizer.all_special_ids) | eos | {model.generation_config.pad_token_id}
    ids = error = handle = close_error = result = None
    events, argmax = [], []
    from contextlib import ExitStack
    with ExitStack() as stack:
        tape = live.Tape(model, stack)
        widths, positions = stack.enter_context(projectors(model))
        stack.enter_context(model.register_forward_hook(
            lambda _, args, output: argmax.append(output.logits[0].argmax(-1).tolist())))
        try:
            handle = opener(model, data, seed=task["seed"], initial_offer=offer,
                            max_new_tokens=live.CAP)
            if handle.opened_prefill_q != data["input_ids"].shape[1]:
                raise RuntimeError("Handle prefill width drift")
            while handle.status == "ready":
                if len(events) >= 12000:
                    raise RuntimeError("Fixed request did not make bounded progress")
                events.append(handle.step(offer))
            if handle.status != "done":
                raise RuntimeError("Request did not complete")
            ids = list(handle.result())
            result = score(proc.tokenizer.decode(ids, skip_special_tokens=True), task)
        except Exception as exc:
            import traceback
            error = {"type": type(exc).__name__, "message": str(exc),
                     "traceback": traceback.format_exc()}
        finally:
            try:
                if handle is not None and handle.status == "ready":
                    handle.close()
            except Exception as exc:
                close_error = {"type": type(exc).__name__, "message": str(exc),
                               "traceback": traceback.format_exc()}
    hook_issue = None
    if tape.projected != positions:
        hook_issue = "Top-level and decoder position hook disagree"
    if not all((i, role) in widths for i in range(len(model.model.decoder.layers))
               for role in ("q_proj", "k_proj")):
        hook_issue = "Missing physical query/key hook"
    return {"identity": {k: task[k] for k in ("index", "seed", "task_id", "dataset")},
            "prompt_sha256": prompt, "prompt_q": data["input_ids"].shape[1],
            "decoder_positions": positions, "top_level_positions": tape.projected,
            "hook_issue": hook_issue, "close_error": close_error,
            "projection_widths": {f"{i}:{name}": w for (i, name), w in widths.items()},
            "encoder": tape.encoders, "events": events, "argmax": argmax,
            "eos_ids": sorted(eos), "special_ids": sorted(special), "output_ids": ids,
            "score": result, "error": error}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("q64", "q128"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--deadline", type=int, required=True)
    ap.add_argument("--samples", type=Path, default=Path("/gs/code/samples.json"))
    ap.add_argument("--weights", type=Path, default=Path("/gs/models/diffusiongemma-cd40195"))
    ap.add_argument("--checkpoint-manifest", type=Path,
                    default=Path("benchmarks/checkpoints/diffusiongemma.json"))
    ap.add_argument("--source-manifest", type=Path, default=Path("source.input.sha256"))
    ap.add_argument("--sealed-parent", type=Path)
    ap.add_argument("--commit", required=True)
    args = ap.parse_args()
    offer = 128 if args.mode == "q128" else 64
    guard("before")
    import torch
    import transformers
    from bdlm_serving.evaluation import scoring
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from . import gemma_response_forks as live
    from . import gemma_live_q128_frozen_h0 as parent
    from . import gemma_full_admission_fixed90_h0 as h0
    from .gemma_full_admission import open_full_admission
    if str(torch.__version__) != RUNTIME["torch"] or transformers.__version__ != RUNTIME["transformers"]:
        raise RuntimeError(f"Wrong A3 runtime: {torch.__version__}, {transformers.__version__}")
    expected_samples = "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed"
    if sha_file(args.samples) != expected_samples:
        raise RuntimeError("Frozen samples drift")
    manifest_sha, files = source_manifest(args.source_manifest, CORE, Path(__file__))
    tasks = load_tasks(args.samples)
    if len(tasks) <= max(CASES):
        raise RuntimeError("Missing registered qualification cases")
    args.out.mkdir(parents=False, exist_ok=False)
    proc = AutoProcessor.from_pretrained(args.weights, local_files_only=True)
    from bdlm_serving.runtime.diffusiongemma import check_snapshot
    check_snapshot(args.weights)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.weights, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", experts_implementation="grouped_mm",
        device_map={"": "cuda:0"}).eval()
    guard("loaded")
    device = {"lane": "A3", "gpu_uuid": GPU, "physical_index": 3, "visible_index": 0,
              "host": subprocess.check_output(["hostname"], text=True).strip()}
    seal = {"schema": "gemma-full-admission-fixed-q64-a3-torch26-v1" if offer == 64 else
            "gemma-full-admission-fixed-q128-a3-torch26-v1",
            "status": "qualified", "reviewed": True, "mode": "full_admission",
            "offer": offer, "admission_boundary": "first_projection_of_later_head",
            "parent_profile": PARENT_PROFILE, "cap": live.CAP, "runtime": RUNTIME,
            "samples_sha256": sha_file(args.samples),
            "checkpoint_manifest_sha256": sha_file(args.checkpoint_manifest),
            "scorer_sha256": sha_file(scoring.__file__),
            "parent_ticket_sha256": None, "executor_sha256": sha_file(CORE),
            "qualification_commit": args.commit,
            "qualification_source_sha256": digest(files), "source_files": files,
            "device": device, "cases": []}
    seal["qualification_origin"] = origin(seal)
    exposed, rows = [], []
    for index in CASES:
        if time.time() >= args.deadline:
            raise RuntimeError("Deadline reached before all registered cases")
        guard("request")
        task = next(t for t in tasks if t["index"] == index)
        if args.mode == "q128":
            sealed_path = args.sealed_parent / f"case-{index:02}.json"
            strict_ref = read(sealed_path)
            strict = parent.run_case(model, proc, task, strict_ref["summary"])
            if live.canonical(strict) != live.canonical(strict_ref):
                raise RuntimeError(f"Strict Q128 parent replay mismatch at {index}")
            raw = h0.run_candidate(model, proc, task, strict_ref["summary"]["prompt_sha256"],
                                   open_full_admission)
            row = h0.compare_receipt(raw, strict_ref)
            parent_sha = sha_file(sealed_path)
        else:
            from bdlm_serving.methods.t2t.serving_profiles import open_strict_gemma
            def parent_opener(model, data, *, seed, initial_offer, max_new_tokens):
                return open_strict_gemma(model, data, seed=seed, initial_offer=initial_offer,
                                         max_new_tokens=max_new_tokens, profile=PARENT_PROFILE)
            data_ids = proc.apply_chat_template(
                [{"role": "user", "content": task["prompt"]}], tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt")
            prompt = live.digest(data_ids["input_ids"][0].tolist())
            strict = run_generic(model, proc, task, prompt, parent_opener, offer)
            write(args.out / f"strict-{index:02}.json", strict)
            raw = run_generic(model, proc, task, prompt, open_full_admission, offer)
            saved_q = h0.Q
            h0.Q = offer
            try:
                row = h0.compare_receipt(raw, strict)
            finally:
                h0.Q = saved_q
            parent_sha = sha_file(args.out / f"strict-{index:02}.json")
        raw["qualification_origin"] = seal["qualification_origin"]
        path = args.out / f"case-{index:02}.json"
        write(path, raw)
        raw_sha = sha_file(path)
        if row["failure"] is not None or row["correct"] is None:
            raise RuntimeError(f"Candidate did not complete case {index}")
        if row["first_fork"] is None or row["over_offer_Q"] <= 0:
            raise RuntimeError(f"Candidate case {index} is unexposed")
        seal["cases"].append({"index": index, "sha256": raw_sha})
        rows.append(row)
        exposed.append(index)
        write(args.out / f"receipt-{index:02}.json", dict(row, raw_sha256=raw_sha,
                                                         strict_sha256=parent_sha))
        print("QUALIFIED " + json.dumps(dict(index=index, row=row), sort_keys=True,
                                         separators=(",", ":"), allow_nan=False), flush=True)
    if len(exposed) < 2:
        raise RuntimeError("Need at least two qualified exposed cases")
    write(args.out / "qualification.json", seal)
    write(args.out / "summary.json", {"schema": seal["schema"], "status": "qualified",
                                      "offer": offer, "cases": rows,
                                      "source_sha256": manifest_sha,
                                      "qualification_sha256": sha_file(args.out / "qualification.json"),
                                      "gpu": GPU, "pid": os.getpid(),
                                      "hostname": device["host"]})
    guard("release")
    print("SUMMARY " + json.dumps({"qualified": exposed, "qualification": str(args.out / "qualification.json"),
                                   "source_sha256": manifest_sha}, sort_keys=True,
                                  separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()

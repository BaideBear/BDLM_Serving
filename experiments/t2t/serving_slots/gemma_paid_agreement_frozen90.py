"""A800 physical A3: formal frozen90 Q128 certified paid-EOS paid-agreement.

Same three arms, model, seeds, samples, ledger floor 0.5 and special-ID rule as
the six-case pairing. All 90 cases are recorded, including original30, new60
and the 84 outside the fixed six. No parameter is changed for the formal run.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from experiments.t2t.serving_slots.gemma_paid_agreement_eval import (
    ARMS, GPU, MODEL, MODEL_MANIFEST, OPTIMUS, OPTIMUS_FILES, Q, SAMPLES,
    guard, match, run_arm, sha, stable)
from experiments.t2t.serving_slots.gemma_ibs_certified_eos import BASELINE_SHA, B, require

COHORTS = {"original30": range(30), "new60": range(30, 90), "frozen90": range(90)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--deadline", type=int, required=True)
    ap.add_argument("--verify-errors-from", type=Path)
    a = ap.parse_args()
    require(a.root.is_dir() and str(a.root).startswith("/dev/shm/"), "Use a /dev/shm root")
    guard("before")
    import torch
    import transformers
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from bdlm_serving.evaluation import frozen, frozen90, scoring
    from bdlm_serving.methods.optimus import upstream
    from bdlm_serving.runtime import diffusiongemma_optimus as rt, optimus_limits
    from bdlm_serving.runtime.diffusiongemma import check_snapshot
    require(transformers.__version__ == "5.12.0", "Wrong transformers version")
    controls, strategy, hashes = upstream.load_controls()
    require(hashes == OPTIMUS_FILES and upstream.COMMIT == OPTIMUS, "Wrong Optimus source")
    require(sha(rt.__file__) == BASELINE_SHA, "Wrong IBS runtime source")
    require(sha(scoring.__file__) == "a72cec90206f69616b9de436d052f6df54cf075ba9abeac4a75a11e46587d692"
            and sha(upstream.__file__) == "f21a4704697cace53b19dfbbbf94bbabc877241b9c979e15b0d40e9a2ea0d223"
            and sha(optimus_limits.__file__) == "842f078e06b5f79b336165b060d1212e947e01182e167379a1d1dbb9f62b5376",
            "Wrong scorer/upstream/limits source")
    require(sha(SAMPLES) == "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed",
            "Wrong frozen samples")
    check_snapshot(MODEL)
    require(sha(frozen.ROOT / "benchmarks/checkpoints/diffusiongemma.json") == MODEL_MANIFEST,
            "Wrong model manifest")
    proc = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa",
        experts_implementation="grouped_mm", device_map={"": "cuda:0"}).eval()
    require(model.config.canvas_length == B, "Wrong canvas length")
    unmask = controls.UnmaskingProcessor(SimpleNamespace(
        confidence_threshold=.9, block_length=B, denoising_steps=B,
        unmasking_strategy=strategy.LOW_CONFIDENCE_DYNAMIC))
    tasks = frozen90.load_tasks(SAMPLES)
    eos = set(model.generation_config.eos_token_id)
    special = set(proc.tokenizer.all_special_ids) | eos | {model.generation_config.pad_token_id}
    original = {}
    indices = range(90)
    if a.verify_errors_from:
        for i in range(90):
            old = json.loads((a.verify_errors_from / f"case-{i:02}.json").read_text())
            require(old["index"] == i and old["task_id"] == tasks[i]["task_id"]
                    and old["seed"] == tasks[i]["seed"], "Original case identity")
            if old["error"] is not None:
                require(set(old["arms"]) == set(ARMS), "Original inference did not complete")
                original[i] = old
        require(original, "No failed checks to replay")
        indices = sorted(original)
    per_case = []
    for index in indices:
        guard("request")
        require(time.time() < a.deadline, "Deadline")
        task = next(t for t in tasks if t["index"] == index)
        data = proc.apply_chat_template(
            [{"role": "user", "content": task["prompt"]}], tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda:0")
        order = ARMS if index % 2 == 0 else ARMS[::-1]
        error = None
        arms = {}
        first = None
        try:
            arms = {m: run_arm(rt, torch, model, data, proc, controls, unmask,
                               task, m, scoring, special) for m in order}
            first = match(arms)
        except Exception as exc:
            error = repr(exc)
        row = dict(
            index=index, task_id=task["task_id"], dataset=task["dataset"],
            seed=task["seed"], order=order, error=error, first_extra=first,
            prompt_sha256=sha256(json.dumps(data["input_ids"][0].tolist(),
                                            separators=(",", ":")).encode()).hexdigest(),
            stock_observer_identical=(not error and stable(arms["stock"]) == stable(arms["observer"])),
            arms={m: dict(
                bill=v["bill"], trigger=v["trigger"], payment=v["payment"],
                correct=v["correct"], N=v["N"], scoring=v["scoring"],
                first_extra=(None if v["agreement"] is None else v["agreement"]["first_extra"]),
                applied=(None if v["agreement"] is None else v["agreement"]["applied"]),
                full_run_extra=(None if v["agreement"] is None else
                                sum(len(o["extra"]) for o in v["agreement"]["observations"] if o["extra"])))
                  for m, v in arms.items()})
        per_case.append(row)
        if original:
            old = original[index]
            row = json.loads(json.dumps(row, allow_nan=False))
            if row["arms"] != old["arms"] or row["order"] != old["order"] or row["error"]:
                with (a.root / f"unverified-{index:02}.json").open("x") as f:
                    json.dump(row, f)
            require(not row["error"] and row["stock_observer_identical"]
                    and row["arms"] == old["arms"] and row["order"] == old["order"],
                    f"Changed original GPU arms or still failed verifier: {index} {row['error']}")
            row["original_sha256"] = sha(a.verify_errors_from / f"case-{index:02}.json")
        with (a.root / f"case-{index:02}.json").open("x") as stream:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        print(json.dumps(dict(index=index, first_extra=first, error=error,
                              identical=row["stock_observer_identical"]),
                         separators=(",", ":")), flush=True)
    if original:
        with (a.root/"replay.json").open("x") as f:
            json.dump(dict(verified=list(indices), gpu=GPU, n=len(per_case),
                           original_source=str(a.verify_errors_from)), f)
        return
    cohorts = {}
    for name, indices in COHORTS.items():
        rows = [per_case[i] for i in indices if per_case[i]["error"] is None]
        cohorts[name] = {arm: dict(
            requests=len(rows),
            correct=sum(r["arms"][arm]["correct"] for r in rows),
            N=sum(r["arms"][arm]["N"] for r in rows),
            F=sum(r["arms"][arm]["bill"]["F"] for r in rows),
            Q=sum(r["arms"][arm]["bill"]["Q"] for r in rows),
            macro_TPF=statistics.mean(r["arms"][arm]["N"] / r["arms"][arm]["bill"]["F"]
                                      for r in rows) if rows else None,
            pooled_TE=(sum(r["arms"][arm]["N"] for r in rows) /
                       sum(r["arms"][arm]["bill"]["Q"] for r in rows)) if rows else None,
            full_run_extra=sum(r["arms"][arm]["full_run_extra"] or 0 for r in rows),
        ) for arm in ARMS}
    good = [r for r in per_case if r["error"] is None]
    first_total = sum(0 if r["first_extra"] is None else len(r["first_extra"]["extra"]) for r in good)
    apply_extra = sum(r["arms"]["apply"]["full_run_extra"] or 0 for r in good)
    observer_extra = sum(r["arms"]["observer"]["full_run_extra"] or 0 for r in good)
    gate = ("passed" if not any(r["error"] for r in per_case)
            and all(r["stock_observer_identical"] for r in per_case) else "failed")
    summary = dict(
        schema="gemma-paid-agreement-q128-frozen90-v1", gpu=GPU, q=Q,
        cases=list(range(90)), cohorts=cohorts, gate=gate,
        first_fork_positions_total=first_total,
        apply_extra_admissions_total=apply_extra,
        observer_would_be_extra_total=observer_extra,
        failed_cases=[r["index"] for r in per_case if r["error"] is not None],
        per_case=per_case, torch=str(torch.__version__),
        transformers=transformers.__version__, model=str(MODEL), samples=str(SAMPLES),
        samples_sha256=sha(SAMPLES),
        baseline_sha=BASELINE_SHA, optimus=OPTIMUS)
    (a.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(gate=gate, first_fork_positions_total=first_total,
                          apply_extra_admissions_total=apply_extra,
                          observer_would_be_extra_total=observer_extra,
                          failed_cases=summary["failed_cases"]), separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()

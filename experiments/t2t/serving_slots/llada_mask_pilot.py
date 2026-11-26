"""Private CPU-only end-to-end pilot of the registered all-slots mask pair.

Run once from the n164 checkout: CUDA_VISIBLE_DEVICES='' taskset -c 32-39
../venv/bin/python -Bu -m experiments.t2t.serving_slots.llada_mask_pilot
(one shell command, PYTHONPATH=.:../fixed-baseline-deps). Append cpu-original30
for the first 30 frozen cases, or summary-cpu-original30 for sealed offline
readback. No GPU registration or return receipt.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from experiments.t2t.serving_slots import llada_all_slots90 as reg


OUT = reg.DATA / "llada21-all-slots90-mask-cpu-pilot-v1"
OUT30 = reg.DATA / "llada21-all-slots90-mask-cpu-original30-v1"
CKPT = Path("/data0/ysy/models/LLaDA2.1-mini")
INDICES = (0, 10, 20)
CFG = dict(confidence_threshold=.5, editing_threshold=.5, max_post_steps=16)
V1_RUN = "d780ceca4bb5a8f0b5b73c6cd3fbbb3716cf00e5a4e35296d29ef9aea98369f6"
V1_COMPLETE = "558efb13598132dcd7758d52ef2bd497300c203228d2cc5ec1b235638633e791"
V1_MODULE = "cd6991c3c100a517b45f7b09ddda3f62fe6b663e74119b5e75779c257b33d93b"
V1_GENERATE = "78f6a098f1b0e4c6efb58693fe92868ab51976deab286221f841a7b65a275fd3"


def generate(model, tok, task, prompt, arm):
    import torch
    from bdlm_serving.methods.t2t.online import package
    from experiments.t2t.serving_slots.llada_all_slots import AllSlotKernel
    from experiments.t2t.serving_slots.mask_pair_llada import head_to_tail_mask

    eos, mask = tok.eos_token_id, tok.convert_tokens_to_ids("<|mask|>")
    special = sorted(set(tok.all_special_ids) | {mask, eos})
    kernel = AllSlotKernel(model, prefix_mode="zero", fuse_growth=True)
    state = kernel.create(torch.tensor([prompt]), CFG, reg.CAP, task["seed"], mask, [eos])
    counts = []

    def witness(_module, _args, kw):
        # Downstream of the formal hook AND binary->additive mask conversion.
        # Check the entire effective mask at attention, not just the nominal offer.
        bias = kw["attention_mask"][0, 0]
        pos = kw["position_ids"][0]
        cols = torch.arange(bias.shape[1])
        native = (pos[:, None] // 32 >= cols[None, :] // 32)
        lo = (state.block + 1) * 32
        window = ((pos >= max(state.prompt_length, lo-32)) & (pos < lo))[:, None] & (
            (cols >= lo) & (cols < lo+32))[None, :]
        opened = arm == reg.PAIR[1]
        expected = native | window if opened else native
        reg.require(torch.equal(bias, torch.zeros_like(bias).masked_fill(
            ~expected, torch.finfo(bias.dtype).min)), "Layer-0 attention received wrong mask")
        edges = int((~native & (bias == 0)).sum())
        if edges and not any(counts):
            print(f"index={task['index']} {arm} first_open_tick={state.offer_tick} edges={edges}", flush=True)
        counts.append(edges)
        if len(counts) == 1 or len(counts) % 100 == 0:
            print(f"index={task['index']} {arm} decoder_call={len(counts)}", flush=True)

    with model.model.layers[0].attention.register_forward_pre_hook(witness, with_kwargs=True):
        stamp = reg.observe(kernel, state, arm, head_to_tail_mask)
    reg.require(counts == [e["opened_edges"] for e in state.events], "Hook/attention mismatch")
    work = [(e["stage"], e["physical_q"]) for e in state.events if e["phase"] != "prefill"]
    row = package(tok, kernel.output(state)[0].tolist(), work, [eos], special,
                  sum(e["physical_q"] for e in state.events if e["phase"] == "prefill"),
                  events=state.events, prompt_ids=prompt, first_input_sha256=stamp,
                  special_ids=special, eos_id=eos, policy=arm, model="llada21",
                  **{k: task[k] for k in reg.KEYS})
    reg.audit(row, arm)
    return row


def score_pair(task, out):
    from bdlm_serving.evaluation.scoring import score
    paths = [out / reg.filename(task["index"], a) for a in reg.PAIR]
    rows = [reg.read(p) for p in paths]
    first = reg.paired(*rows)  # For no exposure this requires full identity.
    result = dict(index=task["index"], dataset=task["dataset"], seed=task["seed"],
                  same_ids=rows[0]["output_ids"] == rows[1]["output_ids"],
                  first_tick=None, first_geometry=None, first_input_sha256=rows[0]["first_input_sha256"])
    if first is not None:
        e = rows[0]["events"][first]
        result.update(first_tick=e["offer_tick"], first_geometry=e["mask_window"])
    for path, row in zip(paths, rows, strict=True):
        reg.require(all(row[k] == task[k] for k in reg.KEYS), "Wrong frozen task")
        verdict = score(row["output_text"], task)
        arm = row["policy"]
        result[arm] = dict(correct=bool(verdict["correct"]), metric=verdict["metric"],
                           N=row["output_tokens"], F=row["generation_forwards"],
                           Q=row["generation_query_rows"], eos=row["eos"],
                           triggered_calls=sum(e["opened_edges"] > 0 for e in row["events"]),
                           edges=sum(e["opened_edges"] for e in row["events"]), row_sha256=reg.sha(path))
    reg.save(out/f"{task['index']:02d}-pair.json", result)
    return result


def check_environment():
    from importlib import metadata, util
    expected = {"torch": "2.9.0", "transformers": "4.57.1", "math-verify": "0.8.0",
                "accelerate": "1.11.0", "safetensors": "0.8.0", "tokenizers": "0.22.2"}
    versions = {k: metadata.version(k) for k in expected}
    cutoff = (OUT/"started").stat().st_ctime_ns
    reg.require(versions == expected and sys.version_info[:3] == (3, 12, 13)
                and Path(sys.prefix) == reg.REPO.parent/"venv"
                and Path(util.find_spec("math_verify").origin).is_relative_to(
                    reg.REPO.parent/"fixed-baseline-deps"), "CPU runtime or judge environment changed")
    extras = ("bdlm_serving/runtime/serial.py", "bdlm_serving/runtime/batch.py",
              "bdlm_serving/methods/t2t/frontier.py", "bdlm_serving/methods/t2t/planner.py",
              "bdlm_serving/methods/t2t/online.py", "bdlm_serving/evaluation/frozen.py",
              "bdlm_serving/evaluation/frozen90.py", "experiments/t2t/serving_slots/mask_probe_llada.py")
    files = [CKPT/e["name"] for e in reg.read(reg.REPO/"benchmarks/checkpoints/llada21.json")]
    files += [reg.REPO/p for p in extras]
    files += [Path(util.find_spec(k.replace("-", "_")).origin) for k in expected]
    reg.require(all(p.stat().st_ctime_ns <= cutoff for p in files), "Runtime inputs changed since v1")
    return dict(python=sys.version.split()[0], venv=sys.prefix, packages=versions, cuda_visible="",
                dtype="bfloat16", attention="sdpa", extra_source_sha256={p: reg.sha(reg.REPO/p) for p in extras})


def reuse_three(out, run, tasks, reference, tok):
    reg.require(reg.sha(OUT/"run.json") == V1_RUN and reg.sha(OUT/"complete.json") == V1_COMPLETE,
                "Unrecognized v1 archive")
    old = reg.read(OUT/"run.json")
    reg.require(old == {k: (list(INDICES) if k == "indices" else V1_MODULE if k == "pilot_sha256"
                            else run[k]) for k in old}, "CPU settings, checkpoint, sources or scorer changed")
    reg.require(hashlib.sha256(inspect.getsource(generate).rstrip("\n").encode()).hexdigest()
                == V1_GENERATE, "Per-arm generation/witness differs from v1")
    cases = {}
    prior = reg.read(OUT/"complete.json")["cases"]
    reg.require([c["index"] for c in prior] == list(INDICES), "Unexpected v1 indices")
    for expected in prior:
        i = expected["index"]
        task = tasks[i]
        reg.require(reg.read(OUT/f"{i:02d}-pair.json") == expected, "Unsealed pair result")
        prompt = reg.read(reference/f"{i:02d}-native32.json")["prompt_ids"]
        for a in reg.PAIR:
            name = reg.filename(i, a)
            payload = (OUT/name).read_bytes()
            sha = expected[a]["row_sha256"]
            reg.require(hashlib.sha256(payload).hexdigest() == sha, f"Wrong v1 row bytes: {name}")
            row = json.loads(payload)
            reg.require(all(row[k] == task[k] for k in reg.KEYS) and row["policy"] == a
                        and row["prompt_ids"] == prompt and tok.decode(row["output_ids"],
                        skip_special_tokens=True) == row["output_text"], f"Frozen identity differs: {name}")
            with (out/name).open("xb") as f:
                f.write(payload)
            reg.require(reg.sha(out/name) == sha, f"Copy verification failed: {name}")
        case = score_pair(task, out)
        reg.require(case == expected, f"Re-audit/re-score differs for {i}")
        cases[i] = case
    reg.save(out/"reuse.json", dict(source=OUT.name, run_sha256=V1_RUN, complete_sha256=V1_COMPLETE,
             pilot_sha256=V1_MODULE, generate_sha256=V1_GENERATE, indices=INDICES,
             files={reg.filename(i,a): c[a]["row_sha256"] for i,c in cases.items() for a in reg.PAIR}))
    return cases


def summary(cases):
    rows = [cases[i] for i in sorted(cases)]
    a, b = reg.PAIR
    def group(sub):
        stats = {}
        for arm in reg.PAIR:
            vals = [r[arm] for r in sub]
            total = {k: sum(v[k] for v in vals) for k in ("correct", "N", "F", "Q", "triggered_calls", "edges")}
            stats[arm] = dict(**total, accuracy=total["correct"]/len(vals),
                              macro_TPF=sum(v["N"]/v["F"] for v in vals)/len(vals),
                              token_efficiency=total["N"]/total["Q"])
        return dict(pairs=len(sub), arms=stats, exposed=sum(r["first_tick"] is not None for r in sub),
                    same_ids=sum(r["same_ids"] for r in sub),
                    rescued=[r["index"] for r in sub if not r[a]["correct"] and r[b]["correct"]],
                    lost=[r["index"] for r in sub if r[a]["correct"] and not r[b]["correct"]])
    return dict(cohort="original30", formal=False, expected_pairs=30,
                complete=set(cases) == set(range(30)), indices=sorted(cases),
                reused_indices=sorted(set(cases) & set(INDICES)),
                contrast="replay_open vs replay", **group(rows), by_dataset={
                    ds: group([r for r in rows if r["dataset"] == ds]) for ds in sorted({r["dataset"] for r in rows})})


def readback(out):
    cases = {}
    for i in range(30):
        p = out/f"{i:02d}-pair.json"
        case = reg.read(p)
        reg.require(case["index"] == i and all(reg.sha(out/reg.filename(i,a)) == case[a]["row_sha256"]
                    for a in reg.PAIR), f"Damaged pair: {i}")
        cases[i] = case
    return dict(**summary(cases), cases=list(cases.values()), run_sha256=reg.sha(out/"run.json"),
                reuse_sha256=reg.sha(out/"reuse.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu-three", "cpu-original30", "summary-cpu-original30"),
                        default="cpu-three", nargs="?")
    mode = parser.parse_args().mode
    if mode == "summary-cpu-original30":
        report = readback(OUT30)
        reg.require(report == reg.read(OUT30/"complete.json"), "Incomplete or changed CPU original30")
        print(json.dumps({k:v for k,v in report.items() if k != "cases"}))
        return
    expanded = mode == "cpu-original30"
    out = OUT30 if expanded else OUT
    indices = list(range(30) if expanded else INDICES)
    os.umask(0o077)
    cpus = sorted(os.sched_getaffinity(0))
    reg.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and 1 <= len(cpus) <= 8,
                "Require hidden CUDA and an affinity of at most eight CPUs")
    # The launcher may have already created the private directory for its log.
    reg.require(not expanded or cpus == list(range(32,40)), "original30 requires CPUs 32-39")
    out.mkdir(mode=0o700, exist_ok=True)
    reg.require(not out.is_symlink() and out.stat().st_mode & 0o777 == 0o700
                and not ({p.name for p in out.iterdir()} - {"run.log"}), "Existing or nonprivate pilot")
    (out/"started").touch(exist_ok=False)  # No partial restart or duplicate execution.
    (out/"tmp").mkdir()
    os.environ.update(TMPDIR=str(out/"tmp"), HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      HF_ENABLE_PARALLEL_LOADING="false", OMP_NUM_THREADS=str(len(cpus)),
                      MKL_NUM_THREADS=str(len(cpus)), OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    import tempfile
    tempfile.tempdir = str(out/"tmp")
    import torch
    torch.set_num_threads(len(cpus))
    torch.set_num_interop_threads(1)
    import transformers.utils as utils
    if not hasattr(utils, "LossKwargs"):
        from typing import TypedDict
        utils.LossKwargs = TypedDict("LossKwargs", {"num_items_in_batch": int}, total=False)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.runtime.serial import install_llada_rectangular_mask
    from experiments.t2t.serving_slots.mask_probe_llada import verify_checkpoint
    from experiments.t2t.serving_slots.top_transition90 import resolve_reference, validate_prompts

    samples = reg.DATA/"samples.json"
    spec = reg.spec(0)
    reg.require(reg.CAP == 2048 and reg.PAIR == ("replay", "replay_open") and
                reg.ARMS["replay"] == reg.ARMS["replay_open"] == (4, 8, 16, 32, 64)
                and reg.sha(samples) == reg.SAMPLES_SHA, "Registered protocol changed")
    frozen = load_tasks(samples)
    tasks = [frozen[i] for i in indices]
    reference = resolve_reference(SimpleNamespace(samples=samples))
    verify_checkpoint(CKPT)
    opts = dict(local_files_only=True, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(CKPT, **opts)
    validate_prompts(tok, tasks, reference)
    run = dict(formal=False, device="cpu", indices=indices, cpus=cpus,
             threads=torch.get_num_threads(), interop=torch.get_num_interop_threads(),
             config=CFG, cap=reg.CAP, arms=spec["arms"], samples_sha256=reg.SAMPLES_SHA,
             checkpoint=str(CKPT), sources=reg.source_hashes(), pilot_sha256=reg.sha(Path(__file__)),
             scorer_sha256=reg.sha(reg.REPO/"bdlm_serving/evaluation/scoring.py"),
             checkpoint_manifest_sha256=reg.sha(reg.REPO/"benchmarks/checkpoints/llada21.json"))
    cases = {}
    if expanded:
        run.update(mode=mode, cohort="original30", generated_indices=[i for i in indices if i not in INDICES],
                   reused_indices=INDICES, environment=check_environment())
    reg.save(out/"run.json", run)
    if expanded:
        cases = reuse_three(out, run, tasks, reference, tok)
        reg.save(out/"progress.json", summary(cases))
        print("Validated 3 archived pairs; 27 new pairs scheduled", flush=True)
    model = AutoModelForCausalLM.from_pretrained(CKPT, **opts, device_map={"": "cpu"},
                dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    reg.require(all(p.device.type == "cpu" for p in model.parameters()), "Non-CPU weights")
    install_llada_rectangular_mask(model, allow_selected_cache=True)
    print("CPU weights ready", flush=True)
    for task in tasks:
        idx = task["index"]
        if idx in cases:
            continue
        prompt = reg.read(reference/f"{idx:02d}-native32.json")["prompt_ids"]
        for arm in reg.PAIR:
            with torch.inference_mode():
                row = generate(model, tok, task, prompt, arm)
            reg.save(out/reg.filename(idx, arm), row)
        case = cases[idx] = score_pair(task, out)
        if expanded:
            reg.save(out/"progress.json", summary(cases))
        print(json.dumps(case), flush=True)
    reg.require(sorted(cases) == indices, "Incomplete CPU pilot")
    if expanded:
        reg.save(out/"complete.json", readback(out))
    else:
        rows = list(cases.values())
        reg.save(out/"complete.json", dict(pairs=len(rows), exposed=sum(
            r["first_tick"] is not None for r in rows), correct={
                a: sum(r[a]["correct"] for r in rows) for a in reg.PAIR}, cases=rows))


if __name__ == "__main__":
    main()

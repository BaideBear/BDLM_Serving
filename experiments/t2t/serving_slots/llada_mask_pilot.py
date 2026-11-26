"""Private CPU-only end-to-end pilot of the registered all-slots mask pair.

Run once from the n164 checkout: CUDA_VISIBLE_DEVICES='' taskset -c 32-39
../venv/bin/python -Bu -m experiments.t2t.serving_slots.llada_mask_pilot
(one shell command, PYTHONPATH=.:../fixed-baseline-deps). Not a GPU receipt.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

from experiments.t2t.serving_slots import llada_all_slots90 as reg


OUT = reg.DATA / "llada21-all-slots90-mask-cpu-pilot-v1"
CKPT = Path("/data0/ysy/models/LLaDA2.1-mini")
INDICES = (0, 10, 20)
CFG = dict(confidence_threshold=.5, editing_threshold=.5, max_post_steps=16)


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


def score_pair(task):
    from bdlm_serving.evaluation.scoring import score
    paths = [OUT / reg.filename(task["index"], a) for a in reg.PAIR]
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
    reg.save(OUT/f"{task['index']:02d}-pair.json", result)
    return result


def main():
    os.umask(0o077)
    cpus = sorted(os.sched_getaffinity(0))
    reg.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and 1 <= len(cpus) <= 8,
                "Require hidden CUDA and an affinity of at most eight CPUs")
    # The launcher may have already created the private directory for its log.
    OUT.mkdir(mode=0o700, exist_ok=True)
    reg.require(not OUT.is_symlink() and OUT.stat().st_mode & 0o777 == 0o700
                and not ({p.name for p in OUT.iterdir()} - {"run.log"}), "Existing or nonprivate pilot")
    (OUT/"started").touch(exist_ok=False)  # No partial restart or duplicate execution.
    (OUT/"tmp").mkdir()
    os.environ.update(TMPDIR=str(OUT/"tmp"), HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      HF_ENABLE_PARALLEL_LOADING="false", OMP_NUM_THREADS=str(len(cpus)),
                      MKL_NUM_THREADS=str(len(cpus)), OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    import tempfile
    tempfile.tempdir = str(OUT/"tmp")
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
    tasks = [frozen[i] for i in INDICES]
    reference = resolve_reference(SimpleNamespace(samples=samples))
    verify_checkpoint(CKPT)
    opts = dict(local_files_only=True, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(CKPT, **opts)
    validate_prompts(tok, tasks, reference)
    reg.save(OUT/"run.json", dict(formal=False, device="cpu", indices=INDICES, cpus=cpus,
             threads=torch.get_num_threads(), interop=torch.get_num_interop_threads(),
             config=CFG, cap=reg.CAP, arms=spec["arms"], samples_sha256=reg.SAMPLES_SHA,
             checkpoint=str(CKPT), sources=reg.source_hashes(), pilot_sha256=reg.sha(Path(__file__)),
             scorer_sha256=reg.sha(reg.REPO/"bdlm_serving/evaluation/scoring.py"),
             checkpoint_manifest_sha256=reg.sha(reg.REPO/"benchmarks/checkpoints/llada21.json")))
    model = AutoModelForCausalLM.from_pretrained(CKPT, **opts, device_map={"": "cpu"},
                dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    reg.require(all(p.device.type == "cpu" for p in model.parameters()), "Non-CPU weights")
    install_llada_rectangular_mask(model, allow_selected_cache=True)
    print("CPU weights ready", flush=True)
    cases = []
    for task in tasks:
        idx = task["index"]
        prompt = reg.read(reference/f"{idx:02d}-native32.json")["prompt_ids"]
        for arm in reg.PAIR:
            with torch.inference_mode():
                row = generate(model, tok, task, prompt, arm)
            reg.save(OUT/reg.filename(idx, arm), row)
        case = score_pair(task)
        cases.append(case)
        print(json.dumps(case), flush=True)
    reg.require([r["index"] for r in cases] == list(INDICES), "Incomplete CPU pilot")
    reg.save(OUT/"complete.json", dict(pairs=len(cases), exposed=sum(
        r["first_tick"] is not None for r in cases), correct={
            a: sum(r[a]["correct"] for r in cases) for a in reg.PAIR}, cases=cases))


if __name__ == "__main__":
    main()

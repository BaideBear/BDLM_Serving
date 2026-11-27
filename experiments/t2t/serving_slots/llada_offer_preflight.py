"""Short CPU checkpoint preflight; no GPU, no correctness scores."""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from experiments.t2t.serving_slots import llada_offer_masks as om
from experiments.t2t.serving_slots import llada_all_slots90 as reg


CAP = 96
OUT = reg.DATA / "llada21-offer-mask-checkpoint-preflight-v1"


def verify(path):
    data = reg.read(path)
    reg.require(data["samples_sha256"] == reg.SAMPLES_SHA
                and data["sources"] == om.provenance() and data["cap"] == CAP
                and data["index"] == 0 and data["mode"] == "cpu-bfloat16"
                and data["trace"] == list(om.TRACE) and len(data["arms"]) == 5
                and set(data["arms"]) == set(om.ARMS)
                and set(data["pairs"]) == {str(q) for q in om.TARGETS},
                "Mask preflight is incomplete or does not correspond to these sources")
    for a in data["arms"].values():
        reg.require(a["F"] > 0 and a["Q"] >= a["F"] and a["N"] >= 0,
                    "Invalid preflight accounting")


def run():
    from experiments.t2t.serving_slots.llada_mask_pilot import CKPT
    from experiments.t2t.serving_slots.mask_probe_llada import verify_checkpoint
    from experiments.t2t.serving_slots.top_transition90 import resolve_reference, validate_prompts
    from bdlm_serving.runtime.serial import install_llada_rectangular_mask
    from bdlm_serving.evaluation.frozen90 import load_tasks
    import torch
    import transformers.utils as utils
    from typing import TypedDict
    if not hasattr(utils, "LossKwargs"):
        utils.LossKwargs = TypedDict("LossKwargs", {"num_items_in_batch": int}, total=False)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    reg.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available()
                and set(os.sched_getaffinity(0)) == {32, 33, 34, 35}
                and not (OUT / "ready.json").exists() and om.TARGETS == (4, 8, 16, 32),
                "Not the registered CPU replay preflight")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    samples = reg.DATA / "samples.json"
    reg.require(reg.sha(samples) == reg.SAMPLES_SHA, "Frozen input changed")
    task = load_tasks(samples)[0]
    reference = resolve_reference(SimpleNamespace(samples=samples))
    verify_checkpoint(CKPT)
    opts = dict(trust_remote_code=True, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(CKPT, **opts)
    validate_prompts(tok, [task], reference)
    prompt = reg.read(reference / reg.filename(0, "native32"))["prompt_ids"]
    model = AutoModelForCausalLM.from_pretrained(
        CKPT, **opts, device_map={"": "cpu"}, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    install_llada_rectangular_mask(model, allow_selected_cache=True)
    sources = om.provenance()
    rows = {}
    with torch.inference_mode():
        for arm in om.ARMS:
            rows[arm] = om.generate(model, tok, task, prompt, arm, cap=CAP)
            print(f"preflight {arm} complete", flush=True)
    pairs = {str(q): om.pair(rows[om.CAUSAL], rows[f"open{q}"], q, cap=CAP)
             for q in om.TARGETS}
    reg.require(sources == om.provenance(), "Sources changed during preflight")
    OUT.mkdir(mode=0o700, exist_ok=True)
    temp = OUT / "pending.json"
    reg.save(temp, dict(index=0, cap=CAP, mode="cpu-bfloat16", trace=list(om.TRACE),
                        samples_sha256=reg.SAMPLES_SHA, sources=sources, pairs=pairs,
                        arms={k: dict(N=r["output_tokens"], F=r["generation_forwards"],
                                      Q=r["generation_query_rows"]) for k, r in rows.items()}))
    verify(temp)
    temp.rename(OUT / "ready.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    if parser.parse_args().verify:
        verify(OUT / "ready.json")
    else:
        run()

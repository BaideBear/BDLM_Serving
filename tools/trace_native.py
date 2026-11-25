"""Capture frozen30 native trajectories; fail closed if work or output diverges from the archive."""
import argparse
import json
from pathlib import Path

from bdlm_serving.evaluation.frozen import load_tasks
from bdlm_serving.methods.t2t.trace import gemma_trace, llada_request, writer


def llada(args, prepare):
    import torch
    import transformers.utils as utils
    if not hasattr(utils, "LossKwargs"):
        from typing import TypedDict
        class LossKwargs(TypedDict, total=False):
            num_items_in_batch: int
        utils.LossKwargs = LossKwargs
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from bdlm_serving.runtime.batch import ModelFacade, ForwardBatcher
    from bdlm_serving.runtime.serial import SerialKernel, install_llada_rectangular_mask

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    tasks = load_tasks(args.samples)
    tok = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    install_llada_rectangular_mask(model, allow_selected_cache=True)
    for p in model.parameters():
        p.data = p.data.clone()
    if prepare:
        prepare(model)
    model.to("cuda")
    batcher = ForwardBatcher(model, "llada21", 1)
    cfg = dict(confidence_threshold=.5, editing_threshold=.5, max_post_steps=16)
    def invoke(t):
        ref = json.loads((args.reference / f"{t['index']:02d}-native32.json").read_text())
        assert all(ref[k] == t[k] for k in ("task_id", "index", "seed", "dataset"))
        with torch.inference_mode(), writer(args.output, t) as emit:
            prompt = tok.apply_chat_template([dict(role="user", content=t["prompt"])], tokenize=False,
                                              add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            assert ids[0].tolist() == ref["prompt_ids"]
            facade = ModelFacade(batcher, f"formal/{t['index']:02d}/native32", t["seed"])
            engine = SerialKernel(facade, "llada21")
            s = engine.create(ids, cfg, 2048, t["seed"], tok.convert_tokens_to_ids("<|mask|>"),
                              [tok.eos_token_id], cached=True)
            emit(prompt_length=s.prompt_length, mask_id=s.mask_id)
            llada_request(engine, s, facade, emit)
            assert engine.output(s)[0].tolist() == ref["output_ids"]
            assert s.events == ref["events"], "per-step work or native decisions diverged"
            assert len(facade.calls) == len(ref["calls"])
            assert all(all(c[k] == b[k] for k in ("rows", "physical_rows", "batch"))
                       for c,b in zip(facade.calls, ref["calls"]))
        print("verified native32", t["index"], flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    for t in tasks:
        batcher.run([lambda: invoke(t)])


def main(args, prepare=None):
    if args.model == "llada21":
        llada(args, prepare)
    else:
        from bdlm_serving.runtime.diffusiongemma import run_native
        run_native(args, before_cuda=prepare, trace_request=gemma_trace(args.output))
        for t in load_tasks(args.samples):
            name = f"{t['index']:02d}-native256.json"
            new, old = [json.loads((d / name).read_text()) for d in (args.output, args.reference)]
            assert new == old, f"Native Gemma parity failed: {name}"
    assert len(list(args.output.glob("*.trace.jsonl.gz"))) == 30
    (args.output/"verified.json").write_text(json.dumps(dict(model=args.model, matched_requests=30)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=("llada21", "gemma"))
    for key in ("checkpoint", "samples", "output", "reference"):
        p.add_argument("--"+key, required=True, type=Path)
    main(p.parse_args())

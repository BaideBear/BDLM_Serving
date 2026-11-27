"""Real single-block Optimus IBS C512. PYTHONPATH must include the study checkout."""
import argparse
import json
import os
from pathlib import Path
import platform
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import transformers
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

from bdlm_serving.evaluation import frozen90
from bdlm_serving.evaluation.frozen import ROOT, sha
from bdlm_serving.evaluation.original30 import save_json
from bdlm_serving.runtime import diffusiongemma_optimus as ibs


B, C, POLICY = 256, 512, "ibs_512"


def planner_gate(controls):
    # Compare controller positions, not model results. Both are given the same
    # synthetic admission history; only the C512 controller is used in inference.
    checked = 0
    for seed in range(6):
        rng = np.random.default_rng(seed)
        plans = [ibs.controller(controls, "ibs", c, 0, None) for c in (B, C)]
        if any(w != B or ctl.extended_window_enabled or ctl.chunk_size != c
               for (ctl, w), c in zip(plans, (B, C), strict=True)):
            raise ValueError("IBS did not open exactly one B256 canvas")
        prior = np.zeros(B, dtype=np.uint8)
        for _ in range(B + 1):
            selections = []
            for (ctl, _), c in zip(plans, (B, C), strict=True):
                buf = np.empty(B, dtype=np.int64)
                n, _ = ctl.fill_processing_buffers(buf, 0, prior, chunk_size_override=c)
                ids = buf[:n]
                if not (0 < n <= B and np.all((0 <= ids) & (ids < B))):
                    raise ValueError("planner selected outside the block")
                selections.append(ids.copy())
            np.testing.assert_array_equal(*selections)
            checked += 1
            new = prior.copy()
            masked = selections[0][prior[selections[0]] == 0]
            if masked.size:
                new[rng.choice(masked, min(masked.size, 1 + seed * 7), replace=False)] = 1
            for ctl, _ in plans:
                ctl.update_from_mask(prior, mark_ready_if_all_left_non_mask=True, dllm_mask_cur=new)
            if plans[0][0].cached_prefix_len != plans[1][0].cached_prefix_len:
                raise ValueError("planner frontiers differ")
            prior = new
            if plans[0][0].cached_prefix_len == B:
                break
        else:
            raise ValueError("planner probe did not finish")
    return checked


def validate(events, trace, prompt_len):
    if (events[:1] != [("encoder", prompt_len)] or not trace or
            any(tag not in ("encoder", "decoder") or (tag == "encoder" and q != B)
                for tag, q in events[1:]) or
            [q for tag, q in events[1:] if tag == "decoder"] != [s["q"] for s in trace]):
        raise ValueError("model projection widths disagree with the generation ledger")
    if any(s["cross"] or not 0 < s["q"] <= B or not 0 <= s["accepted"] <= s["q"]
           or not s["block"] * B < s["visible_end"] <= (s["block"] + 1) * B
           for s in trace):
        raise ValueError("IBS left the physical canvas")


def main(args):
    settings = json.loads((ROOT / "archive/diffusiongemma-native/config.json").read_text())
    if (transformers.__version__ != settings["transformers"] or
            (ibs.B, ibs.CAP, settings["batch_size"]) != (B, 2048, 1)):
        raise ValueError("wrong baseline configuration")
    tasks = frozen90.load_tasks(args.samples)
    controls, strategy, hashes = ibs.load_controls()
    unmask = controls.UnmaskingProcessor(SimpleNamespace(
        confidence_threshold=.9, block_length=B, denoising_steps=B,
        unmasking_strategy=strategy.LOW_CONFIDENCE_DYNAMIC))
    # Narrow process-local extension of the entry whitelist, never change
    # controller(), the upstream plan, limited_rows(), or sampling.
    with patch.object(ibs, "CHUNKS", (*ibs.CHUNKS, C)):
        checked = planner_gate(controls)
        print(f"planner gate: C512 single B256, {checked} identical C256/C512 plans", flush=True)
        if args.planner_only:
            return
        device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not device.startswith("GPU-") or "," in device or torch.cuda.device_count() != 1:
            raise ValueError("bind one explicitly verified idle physical GPU UUID")
        args.output.mkdir(exist_ok=False, parents=True)
        proc = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
        ibs.check_snapshot(args.checkpoint)
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            args.checkpoint, dtype=torch.bfloat16, local_files_only=True,
            attn_implementation=settings["attention"], experts_implementation=settings["experts"],
            device_map={"": "cuda:0"}).eval()
        if model.config.canvas_length != B or model.generation_config.max_denoising_steps != 48:
            raise ValueError("wrong checkpoint generation configuration")
        save_json(args.output / "run.json", dict(settings, model="gemma", policies=[POLICY],
                  suite="frozen90", seed_offset=0, chunk=C, block=B, hard_q_cap=False,
                  extended_window_enabled=False, acceptance="optimus_dynamic_sparse_argmax_0.9",
                  upstream=ibs.COMMIT, hashes=hashes, samples_sha256=sha(args.samples),
                  planner_probe_steps=checked, source_sha256=sha(Path(ibs.__file__)),
                  script_sha256=sha(Path(__file__)), torch=str(torch.__version__),
                  gpu=torch.cuda.get_device_name(), gpu_uuid=device, host=platform.node(),
                  pid=os.getpid(), qualification_excluded=True))
        eos = set(model.generation_config.eos_token_id)
        special = set(proc.tokenizer.all_special_ids) | eos | {model.generation_config.pad_token_id}
        with torch.inference_mode():
            for t in tasks:
                data = proc.apply_chat_template([{"role": "user", "content": t["prompt"]}],
                    tokenize=True, add_generation_prompt=True, return_dict=True,
                    return_tensors="pt").to(model.device)
                if t["index"] == 0:
                    torch.manual_seed(t["seed"])
                    _, tr = ibs.generate(model, data, controls, unmask, "ibs", C, qualify=True)
                    save_json(args.output / "qualified-logits.json", dict(
                        index=0, chunk=C, first_step_logits="bit_identical", probe_trace=tr))
                torch.manual_seed(t["seed"])
                with ibs.observe(model) as events:
                    seq, tr = ibs.generate(model, data, controls, unmask, "ibs", C)
                validate(events, tr, data["input_ids"].shape[1])
                n = sum(i not in special for i in seq)
                f, q = len(events) - 1, sum(width for _, width in events[1:])
                if not seq or len(seq) > ibs.CAP or (seq[-1] not in eos and len(seq) != ibs.CAP):
                    raise ValueError("unexpected stop")
                row = {k: t[k] for k in ("index", "dataset", "task_id", "seed")}
                row.update(policy=POLICY, model="gemma", status="success", output_ids=seq,
                           output_text=proc.tokenizer.decode(seq, skip_special_tokens=True),
                           events=events, trace=tr, output_tokens=n, generation_forwards=f,
                           generation_query_rows=q, generation_physical_padded_rows=q,
                           max_q=max(w for _, w in events[1:]), max_decoder_q=max(s["q"] for s in tr),
                           tpf=n/f, token_efficiency=n/q, eos=seq[-1] in eos, truncated=seq[-1] not in eos)
                save_json(args.output / f"{t['index']:02d}-{POLICY}.json", row)
                print(f"{POLICY} i={t['index']:02d} N={n} F={f} Q={q}", flush=True)
        save_json(args.output / f"complete-{POLICY}.json", dict(policy=POLICY, requests=len(tasks)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--planner-only", action="store_true")
    args = p.parse_args()
    if not args.planner_only and not (args.checkpoint and args.output):
        p.error("--checkpoint and --output required for inference")
    main(args)

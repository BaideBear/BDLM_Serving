"""Replay the first real active_w2 dual decoder call, outside the scored A/B.

The CLI is CPU-only. probe() can reuse an already-loaded eval model, but must
not be called inside a scored request or its forward observer.
"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from bdlm_serving.runtime import diffusiongemma_wave as wave
from experiments.t2t.serving_slots.mask_pair_gemma import mask_for_arm


class _StopCapture(Exception):
    pass


def _copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(map(_copy, value))
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError(f"Unexpected decoder argument: {type(value).__name__}")


def _rng(device):
    return (torch.get_rng_state(),
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def _set_rng(device, state):
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1], device)


def _head(x, block):
    # Experts use flattened rows, attention/layer outputs use batch-one rows.
    if isinstance(x, tuple):
        x = x[0]
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2 or x.shape[0] != 2 * block:
        raise ValueError(f"Unexpected two-canvas output shape: {tuple(x.shape)}")
    return x[:block]


def _hash(x):
    return hashlib.sha256(x.contiguous().view(torch.uint8).cpu().numpy()).hexdigest()


def _compare(a, b, a_layers=None, b_layers=None):
    first = None
    if a_layers is not None:
        if tuple(a_layers) != tuple(b_layers):
            raise ValueError("Replay layer hook order differs")
        first = next((k for k in a_layers if a_layers[k] != b_layers[k]), None)
    return dict(exact=_hash(a) == _hash(b),
                max_abs=float((a.float() - b.float()).abs().max()),
                top1_changed=int((a.argmax(-1) != b.argmax(-1)).sum()),
                first_layer_diff=first)


def _capture(model, data, plan, limit):
    block = plan["block_size"]
    device = data["input_ids"].device
    decoder = model.model.decoder
    state = {}
    steps = 0

    with ExitStack() as hooks:
        def save_soft(_, args):
            # Installed from the decoder pre-hook, after wave's fresh-row hook.
            if len(args) != 2 or "soft" in state:
                raise ValueError("Unexpected self-conditioning call")
            state["soft"] = _copy(args[1])

        def before_decoder(_, args, kw):
            nonlocal steps
            ids = kw["decoder_input_ids"]
            if steps == limit:
                raise _StopCapture
            steps += 1
            if ids.shape == (1, block):
                return
            if args or ids.shape != (1, 2 * block):
                raise ValueError("Expected a batch-one active_w2 decoder call")
            state["step"] = steps - 1
            state["rng"] = _rng(device)
            cache = kw["past_key_values"]
            mask_for_arm(kw["decoder_attention_mask"], cache,
                         model.config.text_config.layer_types, block, 2 * block, opened=False)
            state["cache"] = deepcopy(cache)
            state["kwargs"] = {k: _copy(v) for k, v in kw.items() if k != "past_key_values"}
            hooks.callback(decoder.self_conditioning.register_forward_pre_hook(save_soft).remove)

        def after_model(_, args, output):
            if "step" in state:
                state["anchor"] = _copy(_head(output.logits, block)).cpu()
                # After the real decoder/lm_head, before wave samples this step.
                raise _StopCapture

        hooks.callback(decoder.register_forward_pre_hook(before_decoder, with_kwargs=True).remove)
        hooks.callback(model.register_forward_hook(after_model).remove)
        try:
            wave.generate(model, data, "active_w2", plan)
        except _StopCapture:
            pass
    if "step" not in state:
        return None, steps
    if "soft" not in state or "anchor" not in state:
        raise ValueError("Incomplete trigger capture")
    return state, steps


def _edited_mask(native, block, opened):
    # Sham and off take the same clone + quadrant-write path.
    result = {k: m.clone() for k, m in native.items()}
    for m in result.values():
        m[..., :block, -block:] = opened
    return result


def _replay(model, state, block, arm, replacement):
    kw = _copy(state["kwargs"])
    kw["past_key_values"] = deepcopy(state["cache"])
    ids = kw["decoder_input_ids"]
    if replacement is not None:
        ids[0, block] = replacement
    if arm != "native":
        kw["decoder_attention_mask"] = _edited_mask(kw["decoder_attention_mask"], block, arm == "off")

    decoder = model.model.decoder
    fingerprints = {}

    def restore_soft(_, args):
        if len(args) != 2 or args[1].shape != state["soft"].shape:
            raise ValueError("Self-conditioning replay shape differs")
        return (args[0], _copy(state["soft"]))

    def fingerprint(tag):
        def hook(_, args, output):
            fingerprints[tag] = _hash(_head(output, block))
        return hook

    with ExitStack() as hooks:
        hooks.callback(decoder.self_conditioning.register_forward_pre_hook(restore_soft).remove)
        for i, layer in enumerate(decoder.layers):
            for kind, module in (("attn", layer.self_attn), ("experts", layer.experts), ("output", layer)):
                if module is not None:
                    handle = module.register_forward_hook(fingerprint(f"{i}/{kind}"))
                    hooks.callback(handle.remove)
        _set_rng(ids.device, state["rng"])
        logits = _head(model(**kw).logits, block).detach().cpu().clone()
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Non-finite head logits in replay")
    return logits, fingerprints


@torch.inference_mode()
def probe(model, data, task, plan, *, max_decoder_steps=16):
    """Isolated active_w2 request + six same-state replays; return diagnostics only."""
    device = data["input_ids"].device
    block = plan["block_size"]
    if (device.type not in ("cpu", "cuda") or model.training
            or device != model.device or max_decoder_steps < 1):
        raise ValueError("Expected colocated eval model and a positive capture limit")
    report = dict(index=task["index"], seed=task["seed"], device=str(device), q=2 * block,
                  prompt_rows=data["input_ids"].shape[-1], policy="active_w2",
                  decoder_step_limit=max_decoder_steps,
                  attention=model.config.text_config._attn_implementation,
                  experts=model.config.text_config._experts_implementation,
                  dtype=str(model.dtype), torch=str(torch.__version__))
    with torch.random.fork_rng(devices=[] if device.type == "cpu" else [device]):
        torch.default_generator.manual_seed(task["seed"])
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(task["seed"])
        state, steps = _capture(model, data, plan, max_decoder_steps)
        if state is None:
            return dict(report, status="no_trigger", capture_decoder_forwards=steps)
        ids = state["kwargs"]["decoder_input_ids"]
        vocab = model.config.text_config.vocab_size
        replacement = (int(ids[0, block]) + 1) % vocab
        replays = {arm: [_replay(model, state, block, arm, change)
                         for change in (None, replacement)] for arm in ("native", "sham", "off")}

        def compare_pair(left, right):
            return _compare(left[0], right[0], left[1], right[1])

        anchor = _compare(state["anchor"], replays["native"][0][0])
        sham = [compare_pair(a, b) for a, b in zip(replays["native"], replays["sham"], strict=True)]
        tail_effect = {arm: compare_pair(*pair) for arm, pair in replays.items()}
        matched = anchor["exact"] and all(x["exact"] and x["first_layer_diff"] is None for x in sham)
        positive = tail_effect["off"]["max_abs"] > 0
        return dict(report, status="measured" if matched and positive else "inconclusive",
                    first_dual_step=state["step"], capture_decoder_forwards=steps, replay_forwards=6,
                    cache_rows_by_layer=[layer.keys.shape[-2] for layer in state["cache"].layers],
                    position_start=int(state["kwargs"]["decoder_position_ids"][0, 0]),
                    has_sc_logits=state["kwargs"]["self_conditioning_logits"] is not None,
                    effective_soft_nonzero_rows=state["soft"].ne(0).any(-1).sum(-1).tolist(),
                    effective_tail_soft_is_zero=not bool(state["soft"][:, block:].any()),
                    perturbation=dict(tail_offset=0, old_id=int(ids[0, block]), new_id=replacement),
                    anchor=anchor, sham=dict(original=sham[0], perturbed=sham[1]),
                    opened_tail_positive=positive, tail_effect=tail_effect)


def main():
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

    from bdlm_serving.evaluation.frozen90 import load_tasks
    from bdlm_serving.runtime.diffusiongemma import STUDY

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--index", type=int, choices=range(90), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, choices=range(4, 9), default=4)
    p.add_argument("--max-decoder-steps", type=int, default=16)
    a = p.parse_args()
    if a.output.exists() or a.max_decoder_steps < 1:
        p.error("Choose an unused report path and a positive decoder-step limit")
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    task = load_tasks(a.samples)[a.index]
    plan = wave.resolve_plan(wave.ROOT / "archive/diffusiongemma-wave/config.json")
    native = json.loads((STUDY / "config.json").read_text())
    tokenizer = AutoProcessor.from_pretrained(a.checkpoint, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        a.checkpoint, local_files_only=True, dtype=torch.bfloat16, device_map={"": "cpu"},
        attn_implementation=native["attention"], experts_implementation=native["experts"]).eval()
    data = tokenizer.apply_chat_template(
        [{"role": "user", "content": task["prompt"]}], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt")
    report = probe(model, data, task, plan, max_decoder_steps=a.max_decoder_steps)
    with a.output.open("x") as f:
        json.dump(dict(report, checkpoint=str(a.checkpoint), threads=a.threads), f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()

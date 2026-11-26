"""CPU directional probe for the real Q64 two-B32 LLaDA2.1 mask.

The dual runner uses a rectangular block-causal mask: a left B32 query sees
only the left block, while a right B32 query sees both blocks.  This probe
loads the real checkpoint on CPU and calls the real
``DualBlockKernel._execute`` path.  It checks:

* changing a right-block token leaves every left-block logit bit-exact;
* changing a left-block token changes right-block logits;
* a single B32 block is bidirectional (positive control);
* opening the forbidden left<-right quadrant makes left logits respond to
  the right block (positive control);
* the block-causal split (left commit + right reencode) reproduces the
  full-pair logits, while the opened-mask split does not.

This is a mechanism and effect-size probe, not a downstream quality score.
End-to-end quality still requires the frozen90 runner on the same seeds.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from bdlm_serving.evaluation.frozen import ROOT, sha
from bdlm_serving.methods.t2t.llada_dual import DualBlockKernel
from bdlm_serving.methods.t2t.llada_sparse import _AuditedDenseCache
from bdlm_serving.runtime.serial import install_llada_rectangular_mask


PROMPT = ("Compute 17 * 23 and explain the steps. Then write a short "
          "paragraph about why the sky is blue. The quick brown fox jumps "
          "over the lazy dog. In 1969, humans first landed on the Moon.")
LEFT_POS = 21
RIGHT_POS = 54
SINGLE_POS = 22
PAIR = 64
BLOCK = 32


def compare(left, right):
    if left.shape != right.shape:
        raise ValueError("Different readout shapes")
    diff = (left.float() - right.float()).abs()
    return {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
            "argmax_changed": int((left.argmax(-1) != right.argmax(-1)).sum()),
            "rows": len(left)}


def kl(left, right):
    """Mean KL(left || right) over rows, chunked to bound CPU memory."""
    total = 0.0
    for a, b in zip(left.split(8), right.split(8), strict=True):
        a = a.float().log_softmax(-1)
        b = b.float().log_softmax(-1)
        total += float((a.exp() * (a - b)).sum())
    return total / len(left)


def verify_checkpoint(checkpoint):
    manifest = json.loads((ROOT / "benchmarks/checkpoints/llada21.json").read_text())
    for entry in manifest:
        path = checkpoint / entry["name"]
        if not path.exists() or path.stat().st_size != entry["bytes"]:
            raise ValueError("Checkpoint file mismatch: %s" % entry["name"])
        if "sha256" in entry and sha(path) != entry["sha256"]:
            raise ValueError("Checkpoint hash mismatch: %s" % entry["name"])


def make_state(x, mask_id, eos_id):
    return SimpleNamespace(
        x=x[None].clone(), events=[], block=0, committed_end=0,
        context_version=0, step_in_block=0, prompt_length=len(x),
        mask_id=mask_id, eos_ids=[eos_id],
        config={"confidence_threshold": .5, "editing_threshold": .5,
                "max_post_steps": 16},
        history=SimpleNamespace(fresh=set()), last_query=None,
        left_history=None, right_history=None, left_post_steps=0,
        right_post_steps=0, left_done=False, right_done=False)


def execute(kernel, x, positions, end, mask_id, eos_id, cache=None):
    """One real dual-path forward, with a fresh request state."""
    state = make_state(x, mask_id, eos_id)
    cache = DynamicCache() if cache is None else cache
    audit = _AuditedDenseCache(cache, positions.numel(), end)
    with torch.inference_mode():
        logits = kernel._execute(state, positions, end, audit, "denoise",
                                 "probe", False)
    return logits, cache


def forward(model, x, positions, mask, cache=None):
    """Direct model call used only for the opened-mask intervention."""
    cache = DynamicCache() if cache is None else cache
    embeds = model.get_input_embeddings()(x[None].index_select(1, positions))
    with torch.inference_mode():
        out = model(inputs_embeds=embeds, position_ids=positions[None],
                    attention_mask=mask, past_key_values=cache,
                    use_cache=True, output_attentions=True)
    return out.logits, cache


def content_position(ids, preferred, tokenizer, mask_id):
    special = set(tokenizer.all_special_ids) | {mask_id, tokenizer.eos_token_id}

    def ok(index):
        token = int(ids[index])
        text = tokenizer.decode([token]).strip()
        return token not in special and text and any(c.isalpha() for c in text)

    if ok(preferred):
        return preferred
    for delta in range(1, len(ids)):
        for index in (preferred - delta, preferred + delta):
            if 0 <= index < len(ids) and ok(index):
                return index
    raise ValueError("No content token near %d" % preferred)


def perturb(token, tokenizer, mask_id, vocab):
    special = set(tokenizer.all_special_ids) | {mask_id, tokenizer.eos_token_id}
    out = (token + 1) % vocab
    while out in special:
        out = (out + 1) % vocab
    return out


def run(args):
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    verify_checkpoint(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": "cpu"}).eval()
    install_llada_rectangular_mask(model, allow_selected_cache=True)
    mask_id = tokenizer.convert_tokens_to_ids("<|mask|>")
    eos_id = tokenizer.eos_token_id
    vocab = model.config.vocab_size
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False,
        add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt",
                    add_special_tokens=False).input_ids[0]
    if len(ids) < PAIR:
        raise ValueError("Prompt is shorter than %d tokens" % PAIR)
    ids = ids[:PAIR].clone()

    left_pos = content_position(ids, LEFT_POS, tokenizer, mask_id)
    right_pos = content_position(ids, RIGHT_POS, tokenizer, mask_id)
    single_pos = content_position(ids, SINGLE_POS, tokenizer, mask_id)
    if not (0 <= left_pos < BLOCK and BLOCK <= right_pos < PAIR
            and 0 <= single_pos < BLOCK):
        raise ValueError("Probe positions must fall in the expected blocks")
    left_pert = ids.clone()
    left_pert[left_pos] = perturb(int(ids[left_pos]), tokenizer, mask_id, vocab)
    right_pert = ids.clone()
    right_pert[right_pos] = perturb(int(ids[right_pos]), tokenizer, mask_id, vocab)
    single_pert = ids.clone()
    single_pert[single_pos] = perturb(int(ids[single_pos]), tokenizer, mask_id, vocab)

    kernel = DualBlockKernel(model, PAIR, 0, PAIR, policy="hard_noedit",
                             selection_seed=0)
    captured = []

    def hook(_module, hook_args, hook_kwargs):
        mask = hook_kwargs.get("attention_mask")
        if mask is None and len(hook_args) > 1:
            mask = hook_args[1]
        if mask is not None:
            captured.append(mask.detach().clone())

    handle = model.model.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        positions = torch.arange(PAIR)
        full, _ = execute(kernel, ids, positions, PAIR, mask_id, eos_id)
        if not captured:
            raise RuntimeError("No attention mask captured from the real path")
        blocked = captured.pop()
        geometry = dict(
            shape=list(blocked.shape),
            forbidden=int((blocked[0, 0, :BLOCK, BLOCK:] == 0).sum()),
            allowed=int(blocked[0, 0, BLOCK:, :BLOCK].sum()),
        )
        if (tuple(blocked.shape) != (1, 1, PAIR, PAIR)
                or geometry["forbidden"] != BLOCK * BLOCK
                or geometry["allowed"] != BLOCK * BLOCK
                or not bool(blocked[0, 0, :BLOCK, :BLOCK].all())
                or not bool(blocked[0, 0, BLOCK:, BLOCK:].all())):
            raise RuntimeError("Unexpected real-path mask geometry: %s" % geometry)

        right_changed, _ = execute(kernel, right_pert, positions, PAIR,
                                   mask_id, eos_id)
        captured.clear()
        right_to_left = compare(full[0, :BLOCK], right_changed[0, :BLOCK])
        right_self = compare(full[0, BLOCK:], right_changed[0, BLOCK:])

        left_changed, _ = execute(kernel, left_pert, positions, PAIR,
                                  mask_id, eos_id)
        captured.clear()
        left_to_right = compare(full[0, BLOCK:], left_changed[0, BLOCK:])
        left_self = compare(full[0, :BLOCK], left_changed[0, :BLOCK])

        single_positions = torch.arange(BLOCK)
        single_full, _ = execute(kernel, ids, single_positions, BLOCK,
                                 mask_id, eos_id)
        single_changed, _ = execute(kernel, single_pert, single_positions,
                                    BLOCK, mask_id, eos_id)
        captured.clear()
        keep = [i for i in range(BLOCK) if i != single_pos]
        single = compare(single_full[0, keep], single_changed[0, keep])

        opened = blocked.clone()
        opened[..., :BLOCK, BLOCK:] = True
        if (not bool(opened.all())
                or int((opened != blocked).sum()) != BLOCK * BLOCK):
            raise RuntimeError("The intervention must open exactly the forbidden quadrant")
        opened_full, _ = forward(model, ids, positions, opened)
        opened_right, _ = forward(model, right_pert, positions, opened)
        captured.clear()
        opened_front_to_back = compare(opened_full[0, :BLOCK],
                                       opened_right[0, :BLOCK])
        opened_self = compare(opened_full[0, BLOCK:], opened_right[0, BLOCK:])
        mask_effect = compare(full[0, :BLOCK], opened_full[0, :BLOCK])

        left_cache = DynamicCache()
        left_only, left_cache = execute(kernel, ids, single_positions, BLOCK,
                                        mask_id, eos_id, cache=left_cache)
        right_only, _ = execute(kernel, ids, torch.arange(BLOCK, PAIR), PAIR,
                                mask_id, eos_id, cache=left_cache)
        captured.clear()
        split_blocked = dict(
            left_max_abs=float((full[0, :BLOCK] - left_only[0]).abs().max()),
            right_max_abs=float((full[0, BLOCK:] - right_only[0]).abs().max()),
            left_argmax_changed=int(
                (full[0, :BLOCK].argmax(-1) != left_only[0].argmax(-1)).sum()),
            right_argmax_changed=int(
                (full[0, BLOCK:].argmax(-1) != right_only[0].argmax(-1)).sum()),
        )

        open_left, open_left_cache = forward(
            model, ids, single_positions, opened[..., :BLOCK, :BLOCK])
        open_right, _ = forward(
            model, ids, torch.arange(BLOCK, PAIR),
            torch.ones(1, 1, BLOCK, PAIR, dtype=torch.bool),
            cache=open_left_cache)
        split_opened = dict(
            left_max_abs=float(
                (opened_full[0, :BLOCK] - open_left[0]).abs().max()),
            right_max_abs=float(
                (opened_full[0, BLOCK:] - open_right[0]).abs().max()),
            left_argmax_changed=int(
                (opened_full[0, :BLOCK].argmax(-1)
                 != open_left[0].argmax(-1)).sum()),
            right_argmax_changed=int(
                (opened_full[0, BLOCK:].argmax(-1)
                 != open_right[0].argmax(-1)).sum()),
        )
    finally:
        handle.remove()

    report = dict(
        checkpoint=str(args.checkpoint), model="llada21", device="cpu",
        dtype="bfloat16", attention="sdpa", torch=str(torch.__version__),
        threads=torch.get_num_threads(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        prompt_tokens=len(ids), pair=PAIR, block=BLOCK,
        positions=dict(left=left_pos, right=right_pos, single=single_pos),
        perturbations=dict(
            left=dict(original=int(ids[left_pos]),
                      changed=int(left_pert[left_pos])),
            right=dict(original=int(ids[right_pos]),
                       changed=int(right_pert[right_pos])),
            single=dict(original=int(ids[single_pos]),
                        changed=int(single_pert[single_pos])),
        ),
        mask=geometry,
        right_to_left=right_to_left,
        right_self_change=right_self,
        left_to_right=left_to_right,
        left_self_change=left_self,
        single_block=single,
        opened_front_to_back=opened_front_to_back,
        opened_self_change=opened_self,
        mask_effect_left=mask_effect,
        commit_reencode=dict(blocked=split_blocked, opened=split_opened),
        quality_proxy=dict(
            right_left_kl=kl(full[0, BLOCK:], left_changed[0, BLOCK:]),
            mask_effect_left_kl=kl(full[0, :BLOCK], opened_full[0, :BLOCK]),
        ),
    )
    return report


def controls_pass(report):
    return (
        report["right_to_left"]["max_abs"] == 0.0
        and report["right_to_left"]["argmax_changed"] == 0
        and report["right_self_change"]["max_abs"] > 0.0
        and report["left_to_right"]["max_abs"] > 0.0
        and report["single_block"]["max_abs"] > 0.0
        and report["single_block"]["argmax_changed"] > 0
        and report["opened_front_to_back"]["max_abs"] > 0.0
        and report["opened_front_to_back"]["argmax_changed"] > 0
        and report["commit_reencode"]["blocked"]["left_max_abs"] == 0.0
        and report["commit_reencode"]["blocked"]["right_max_abs"] == 0.0
        and report["commit_reencode"]["opened"]["left_max_abs"] > 0.0
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2), flush=True)
    if not controls_pass(result):
        raise SystemExit("Directional controls failed; inspect raw values before interpreting")

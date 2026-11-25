"""Observe real QKV projection rows at a B32 boundary, using full 2.0 weights on CPU."""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM

from bdlm_serving.runtime.serial import install_llada_rectangular_mask


@torch.inference_mode()
def probe(checkpoint):
    torch.manual_seed(174)
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cpu().eval()
    layers = model.model.layers
    widths = [[] for _ in layers]
    for i, layer in enumerate(layers):
        qkv, = [m for name, m in layer.named_modules() if name.endswith(".query_key_value")]
        qkv.register_forward_hook(lambda mod, args, out, i=i:
                                  widths[i].append((args[0].shape[1], out.shape[1])))
    inputs, reference = [], []

    class TwoForwardsObserved(Exception):
        pass

    def observe_input(mod, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        assert kwargs.get("past_key_values") is None
        inputs.append(ids.clone())

    def observe_logits(mod, args, out):
        if len(inputs) == 2:
            reference.append(out[:, -32:].clone())
            raise TwoForwardsObserved

    h = model.model.register_forward_pre_hook(observe_input, with_kwargs=True)
    h2 = model.lm_head.register_forward_hook(observe_logits)
    try:
        # 63 prompt tokens leave one mask in block 2. Disable EOS only for
        # this boundary probe; no frozen30 answer is generated or rescored.
        model.generate(inputs=torch.arange(10, 73)[None], gen_length=64, steps=32,
                       block_length=32, threshold=.95, temperature=0., eos_id=-1)
    except TwoForwardsObserved:
        pass
    finally:
        h.remove()
        h2.remove()
    assert all(w == [(64, 64), (96, 96)] for w in widths)

    # Replay the exact 96 input IDs observed above. Prefill two final blocks
    # once, then expose their K/V to the 32 live queries via a rectangular mask.
    install_llada_rectangular_mask(model)
    ids = inputs[-1]
    pos = torch.arange(96)
    mask = (pos[:, None] // 32 >= pos[None, :] // 32)[None, None]
    prefix = model(input_ids=ids[:, :64], position_ids=pos[None, :64],
                   attention_mask=mask[:, :, :64, :64], use_cache=True).past_key_values
    assert prefix.get_seq_length() == 64
    out = model(input_ids=ids[:, 64:], position_ids=pos[None, 64:],
                attention_mask=mask[:, :, 64:, :], past_key_values=prefix, use_cache=True)
    assert out.past_key_values.get_seq_length() == 96
    assert all(w[-2:] == [(64, 64), (32, 32)] for w in widths)
    logits, ref = out.logits.float(), reference[0].float()
    return dict(layers_observed=len(layers), checkpoint_generate_q=[64, 96],
                cached_prefix_build_q=64, cached_active_q=32, visible_kv=96,
                active_top1_matches=int((logits.argmax(-1) == ref.argmax(-1)).sum()),
                positions_compared=32, mean_abs_logit_diff=(logits-ref).abs().mean().item())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    print(json.dumps(probe(p.parse_args().checkpoint), indent=2))

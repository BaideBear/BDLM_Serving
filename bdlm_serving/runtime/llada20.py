"""Measure the checkpoint's own serial generate, including every replayed prefix row."""
import torch


def checkpoint_generate(model, original_forward, facade, prompt, cap, seed, mask_id, eos):
    events = []

    def observe(_module, _args, kwargs):
        ids = kwargs.get("input_ids")
        hidden = ids if ids is not None else kwargs["inputs_embeds"]
        q = hidden.shape[1]
        assert hidden.shape[0] == 1
        events.append({"phase": "native_full_prefix", "q_tokens": q})
        facade.calls.append({"rows": q, "physical_rows": q})

    adapted = model.model.forward
    model.model.forward = original_forward
    handle = model.model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        with torch.random.fork_rng(devices=[prompt.device.index]):
            torch.manual_seed(seed)
            new_ids = model.generate(inputs=prompt, gen_length=cap, block_length=32, steps=32,
                                     temperature=0., threshold=.95, mask_id=mask_id,
                                     eos_id=eos, eos_early_stop=False)
    finally:
        handle.remove()
        model.model.forward = adapted
    # With eos_early_stop=False the checkpoint finishes the block, stops at EOS
    # and returns completion IDs only. There is no separate prompt-only prefill.
    assert events and new_ids.shape[1] <= cap
    return torch.cat([prompt, new_ids], dim=1), events, {
        "backend": "original checkpoint generate; full prefix recomputed each denoising step",
        "threshold": .95, "t2t": False, "sample": "checkpoint multinomial", "eos_early_stop": False}

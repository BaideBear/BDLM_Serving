"""Run the checkpoint's own sampler with one live block and a committed KV prefix."""
from collections import Counter
from contextlib import ExitStack
from unittest.mock import patch

import torch
from transformers.cache_utils import DynamicCache


def cached_generate(model, facade, prompt, cap, seed, mask_id, eos):
    b = 32
    cache = DynamicCache()
    events = []
    original = model.forward
    last = None
    widths = []

    def observe(_module, args, out):
        assert args[0].shape[:2] == out.shape[:2] and out.shape[0] == 1
        widths.append(int(out.shape[1]))

    def run_slice(ids, pos, bias, start, end, phase):
        assert cache.get_seq_length() == start
        assert phase == "prefill" or end - start == b
        widths.clear()
        # The checkpoint supplies additive 0 / -inf; the rectangular adapter
        # accepts binary visibility. Slice BOTH axes before calling the model.
        visible = bias[:, :, start:end, :end].eq(0)
        out = original(input_ids=ids[:, start:end], position_ids=pos[:, start:end],
                       attention_mask=visible, past_key_values=cache, use_cache=True)
        assert out.past_key_values is cache and cache.get_seq_length() == end
        assert widths == [end - start] * len(model.model.layers)
        q = widths[0]
        events.append(dict(phase=phase, q_tokens=q, kv=end))
        facade.calls.append(dict(rows=q, physical_rows=q))
        if phase == "denoise":
            cache.crop(start)
            assert cache.get_seq_length() == start
        return out

    def commit():
        ids, pos, bias, start, end = last
        # ids aliases HF's live canvas and now holds the final sampled values.
        run_slice(ids, pos, bias, start, end, "commit")

    def live_forward(ids, *, attention_mask, position_ids):
        nonlocal last
        end = ids.shape[1]
        start = end - b
        if last is None:
            assert start == prompt.shape[1] // b * b
            if start:
                run_slice(ids, position_ids, attention_mask, 0, start, "prefill")
        elif end != last[-1]:
            assert start == last[-1]
            commit()
        last = (ids, position_ids, attention_mask, start, end)
        return run_slice(ids, position_ids, attention_mask, start, end, "denoise")

    with ExitStack() as stack:
        for layer in model.model.layers:
            qkv, = [m for n, m in layer.named_modules() if n.endswith(".query_key_value")]
            stack.callback(qkv.register_forward_hook(observe).remove)
        stack.enter_context(patch.object(model, "forward", live_forward))
        devices = [prompt.device.index] if prompt.is_cuda else []
        stack.enter_context(torch.random.fork_rng(devices=devices))
        torch.manual_seed(seed)
        new = model.generate(inputs=prompt, gen_length=cap, block_length=b, steps=b,
                             temperature=0., threshold=.95, mask_id=mask_id,
                             eos_id=eos, eos_early_stop=False)
        assert last is not None
        commit()  # Same terminal-commit convention as the 2.1 reference.
    counts = Counter(e["phase"] for e in events)
    return torch.cat([prompt, new], dim=1), events, dict(
        backend="checkpoint generate with committed-prefix KV", sample="checkpoint multinomial",
        threshold=.95, t2t=False, eos_early_stop=False, terminal_commit=True,
        phase_forwards=dict(counts),
        phase_q={p: sum(e["q_tokens"] for e in events if e["phase"] == p) for p in counts},
        layers_observed=len(model.model.layers), max_visible_kv=last[-1])

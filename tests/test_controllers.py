import os
import types

import pytest
import torch

from bdlm_serving.methods.b3.gate import B3Gate
from bdlm_serving.methods.flowblock.prefix_cache import PrefixBackend
from bdlm_serving.methods.optimus.boundary import BoundaryModel, native_visible
from bdlm_serving.methods.optimus.fixed import FixedOptimus
from bdlm_serving.runtime.sampling import top_k_logits, top_p_logits
from bdlm_serving.upstream import control_directory


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    torch.set_num_threads(1)


def test_source_identity():
    for name in ("Optimus", "FlowBlock"):
        assert control_directory(name).is_dir()


def test_absolute_block_visibility():
    positions = torch.tensor([[30, 31, 32, 35, 64]])
    original = torch.ones(1, 1, 5, 66, dtype=torch.bool)
    original[..., 0] = False
    result = native_visible(original, positions)[0, 0]
    for i, q in enumerate(positions[0].tolist()):
        assert result[i].tolist() == [k != 0 and k//32 <= q//32 for k in range(66)]


@pytest.mark.parametrize("mode", ["ibs", "obs", "masked_obs"])
@pytest.mark.parametrize("chunk", [4, 8, 16, 32])
@pytest.mark.parametrize("kind", ["llada21", "sdar"])
def test_fixed_streaming(kind, mode, chunk):
    class Toy:
        def __call__(self, tokens, **kw):
            proposed = torch.where(tokens == 0, 2, 3)
            logits = torch.full((*tokens.shape, 8), -30.)
            return types.SimpleNamespace(logits=logits.scatter_(-1, proposed[..., None], 30.))
    prompt = torch.ones(1, 35, dtype=torch.long)
    model = BoundaryModel(Toy()) if mode == "masked_obs" else Toy()
    actual_mode = "obs" if mode == "masked_obs" else mode
    x, events, info = FixedOptimus(model, kind, chunk, actual_mode).run(prompt, 96, 17, 0, [7])
    assert torch.equal(x[:, :35], prompt) and info["window_switches"] == 0
    generation = [e for e in events if e["phase"] != "prefill"]
    pending = set()
    for e in generation:
        assert e["fixed_chunk"] == chunk
        eligible = {p for p, masked in zip(e["absolute_positions"], e["selected_masked"])
                    if not masked and p >= 35}
        assert set(e["edited_positions"]) <= eligible
        pending -= set(e["absolute_positions"])
        pending.update(e["edited_positions"])
        assert all(p >= e["block"]*32 + e["cached_prefix_after"] for p in pending)
        if mode == "ibs":
            assert e["cross_boundary_rows"] == 0
    assert not pending
    assert any(e["edited"] for e in generation) == (kind == "llada21")


@pytest.mark.parametrize("size", [32, 256])
def test_retired_queries_never_reach_transformer(size):
    class Backend:
        config = types.SimpleNamespace(vocab_size=128)
        model = types.SimpleNamespace(layers=[1])
        device = torch.device("cpu")
        calls = []

        def __call__(self, **kw):
            q = kw["inputs_embeds"].shape[1]
            self.calls.append(q)
            kw["past_key_values"].update(torch.zeros(1, 1, q, 2), torch.zeros(1, 1, q, 2), 0)
            return types.SimpleNamespace(logits=torch.zeros(1, q, 128))
    native = Backend()
    raw = torch.full((1, 3*size), 127)
    owner = types.SimpleNamespace(block_size=size, raw_tokens=types.SimpleNamespace(data=raw),
            owner=types.SimpleNamespace(decoder=types.SimpleNamespace(mask_id=127)))
    b = PrefixBackend(native, owner, True)
    def call():
        return b(None, inputs_embeds=torch.zeros(1, 2*size, 4), position_ids=torch.arange(size, 3*size)[None],
                 use_cache=True, past_key_values=types.SimpleNamespace(_data=torch.zeros(1, 2, 1, 1, 4*size, 2)),
                 attention_mask=torch.ones(1, 2*size, 4*size, dtype=torch.bool), replace_position=(0, 0))
    call()
    raw[0, size:size+4] = 5
    call()
    raw[0, size+1] = 6
    call()
    call()
    assert native.calls == [2*size, 2*size, 2*size-1, 2*size-4]
    assert b.prefixes[1] == 4
    assert not set(b.events[-1]["absolute_positions"]) & set(b.events[-1]["removed_positions"])


def test_b3_gate_and_sampling():
    gate = B3Gate(block_size=4, reset_previous_on_promotion=True)
    e = gate.observe(0, torch.tensor([9., 8., 100., 6.]), [0], False, False)
    assert e["tau"] is None and not e["open_on_next_pass"]
    e = gate.observe(0, torch.ones(4), [2], False, False)
    assert e["candidate"] == 9. and e["open_on_next_pass"]
    logits = torch.tensor([[2., 2., 1., 0.]])
    assert torch.isfinite(top_k_logits(logits, 1)).sum() == 2
    assert torch.isfinite(top_p_logits(logits, 0.01)).sum() == 1


@pytest.mark.skipif(not os.getenv("LLADA_TEST_CHECKPOINT"), reason="requires trusted local checkpoint config")
def test_flowblock_tiny_model():
    from transformers import AutoConfig, AutoModelForCausalLM
    from bdlm_serving.methods.flowblock.control import FlowBlockControl
    from bdlm_serving.methods.flowblock.prefix_cache import FlowBlockPrefix
    from bdlm_serving.runtime.serial import install_llada_rectangular_mask
    cfg = AutoConfig.from_pretrained(os.environ["LLADA_TEST_CHECKPOINT"], trust_remote_code=True,
                                     local_files_only=True)
    for key, val in dict(hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            rotary_dim=8, num_experts=4, num_experts_per_tok=2, n_group=1,
            topk_group=1, vocab_size=128, pad_token_id=0).items():
        setattr(cfg, key, val)
    torch.manual_seed(1729)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, attn_implementation="sdpa").eval()
    install_llada_rectangular_mask(model, allow_selected_cache=True)
    config = dict(confidence_threshold=1., editing_threshold=2., max_post_steps=0)
    for length in (32, 43):
        prompt = torch.randint(1, 120, (1, length))
        for w in (2, 3, 4):
            args = model, w, 127, -1, config
            x, events, _ = FlowBlockControl(*args).run(prompt, 192)
            y, off, _ = FlowBlockPrefix(*args, enabled=False).run(prompt, 192)
            assert torch.equal(x, y) and events[1:] == off[1:]
            observed = []
            handle = model.model.layers[0].register_forward_pre_hook(lambda mod, a, kw:
                    observed.append((a[0] if a else kw["hidden_states"]).shape[-2]), with_kwargs=True)
            try:
                y, on, _ = FlowBlockPrefix(*args).run(prompt, 192)
            finally:
                handle.remove()
            assert torch.equal(y[:, :length], prompt)
            assert observed == [e["q_tokens"] for e in on if e["physical_forwards"]]
            assert sum(e.get("skipped_q", 0) for e in on) > 0

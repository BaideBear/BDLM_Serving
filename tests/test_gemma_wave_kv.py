"""CPU toy tests for WaveKV and wave_attention.

No checkpoint or GPU is used.  The toy decoder mimics the native contract:
each layer receives ``[prefix K/V, current Q K/V]`` and the attention function
sees a boolean mask (or ``None``).  A fake registry lets the same wrapper run
without importing Transformers; one integration test exercises the real
DiffusionGemma AttentionInterface when Transformers 5.12 is available.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch
from torch import nn

try:
    from experiments.t2t.serving_slots import gemma_wave_kv as gkv
except ImportError:
    # Explicit local staging: this test lives beside gemma_wave_kv.py in a
    # scratch directory.  Register a synthetic canonical package so the same
    # import path resolves to the staged file.
    import importlib.util

    _here = Path(__file__).resolve().parent
    for _name in ("experiments", "experiments.t2t", "experiments.t2t.serving_slots"):
        if _name not in sys.modules:
            _module = ModuleType(_name)
            _module.__path__ = []
            sys.modules[_name] = _module
    _spec = importlib.util.spec_from_file_location(
        "experiments.t2t.serving_slots.gemma_wave_kv", _here / "gemma_wave_kv.py"
    )
    gkv = importlib.util.module_from_spec(_spec)
    sys.modules["experiments.t2t.serving_slots.gemma_wave_kv"] = gkv
    _spec.loader.exec_module(gkv)


CPU = torch.device("cpu")
ALIAS = "bdlm_gemma_wave_kv_sdpa"


def ids(values):
    return torch.tensor([values], dtype=torch.long)


def pos(values):
    return torch.tensor([values], dtype=torch.long)


def toy_sdpa(q, k, v, mask):
    if k.shape[1] != q.shape[1]:
        repeat = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def make_registry():
    calls = []

    def sdpa(module, q, k, v, mask, **kwargs):
        calls.append(
            dict(
                tag=getattr(module, "tag", None),
                impl=module.config._attn_implementation,
                q=q.detach().clone(),
                k=k.detach().clone(),
                v=v.detach().clone(),
                mask=None if mask is None else mask.detach().clone(),
                kwargs=dict(kwargs),
            )
        )
        return toy_sdpa(q, k, v, mask), None

    return {"sdpa": sdpa}, calls


class ToyAttention(nn.Module):
    def __init__(self, layer_idx, tag, sliding_window=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.tag = tag
        self.config = NS(_attn_implementation="sdpa", sliding_window=sliding_window)
        self.is_causal = False
        self.num_key_value_groups = 1


class ToyLayer(nn.Module):
    def __init__(self, layer_idx, tag, dim, heads, kv_heads, prefix_len, sliding_window=None, seed=0):
        super().__init__()
        self.self_attn = ToyAttention(layer_idx, tag, sliding_window)
        self.dim = dim
        self.heads = heads
        self.kv_heads = kv_heads
        self.q_proj = nn.Linear(dim, heads * dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * dim, bias=False)
        self.o_proj = nn.Linear(heads * dim, dim, bias=False)
        generator = torch.Generator().manual_seed(seed + 100 * layer_idx + (0 if tag == "decoder" else 10000))
        self.register_buffer("prefix_k", torch.randn(1, kv_heads, prefix_len, dim, generator=generator))
        self.register_buffer("prefix_v", torch.randn(1, kv_heads, prefix_len, dim, generator=generator))

    def forward(self, x, q_len, mask, registry):
        batch = x.shape[0]
        q = self.q_proj(x).view(batch, q_len, self.heads, self.dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, q_len, self.kv_heads, self.dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, q_len, self.kv_heads, self.dim).transpose(1, 2)
        k = torch.cat((self.prefix_k, k), dim=2)
        v = torch.cat((self.prefix_v, v), dim=2)
        fn = registry[self.self_attn.config._attn_implementation]
        out, _ = fn(
            self.self_attn,
            q,
            k,
            v,
            mask,
            dropout=0.0,
            scaling=1.0,
            sliding_window=self.self_attn.config.sliding_window,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(batch, q_len, self.heads * self.dim)
        return self.o_proj(out)


class ToyDecoder(nn.Module):
    def __init__(self, prefix_lens=(5, 3), dim=4, heads=2, kv_heads=1, vocab=16, max_pos=64, seed=0):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_pos, dim)
        self.out = nn.Linear(dim, vocab, bias=False)
        self.layers = nn.ModuleList(
            [
                ToyLayer(
                    i,
                    "decoder",
                    dim,
                    heads,
                    kv_heads,
                    prefix_len,
                    sliding_window=None if i == 0 else 3,
                    seed=seed + i,
                )
                for i, prefix_len in enumerate(prefix_lens)
            ]
        )
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.5)

    def forward(self, token_ids, positions, masks=None, registry=None):
        q_len = token_ids.shape[1]
        x = self.embed(token_ids) + self.pos(positions)
        for layer in self.layers:
            mask = masks.get(layer.self_attn.layer_idx) if isinstance(masks, dict) else masks
            x = x + layer(x, q_len, mask, registry)
        return NS(logits=self.out(x))


class ToyEncoder(nn.Module):
    def __init__(self, dim=4, heads=2, kv_heads=1, prefix_len=5):
        super().__init__()
        self.layers = nn.ModuleList([ToyLayer(0, "encoder", dim, heads, kv_heads, prefix_len, seed=999)])

    def forward(self, token_ids, positions, registry):
        q_len = token_ids.shape[1]
        x = torch.zeros(1, q_len, self.layers[0].dim)
        for layer in self.layers:
            x = layer(x, q_len, None, registry)
        return x


class ToyModel:
    def __init__(self, prefix_lens=(5, 3)):
        self.model = NS(decoder=ToyDecoder(prefix_lens), encoder=ToyEncoder())

    def decoder(self, token_ids, positions, masks=None, registry=None):
        return self.model.decoder(token_ids, positions, masks=masks, registry=registry)


def make_world(monkeypatch, prefix_lens=(5, 3)):
    model = ToyModel(prefix_lens)
    registry, calls = make_registry()
    monkeypatch.setattr(gkv, "_attention_functions", lambda: registry)
    return model, registry, calls


def forward(model, registry, bank, positions, token_ids, canvas_origin=10, causal=True, masks=None):
    bank.begin(positions, canvas_origin, CPU, causal=causal)
    return model.decoder(token_ids, pos(positions), masks=masks, registry=registry)


def test_selected_q_projection_and_three_step_kv_cache(monkeypatch):
    model, registry, calls = make_world(monkeypatch)
    bank = gkv.WaveKV(4)
    with gkv.wave_attention(model, bank):
        out0 = forward(model, registry, bank, [10, 11, 12, 13], ids([1, 2, 3, 4]))
        assert out0.logits.shape == (1, 4, 16)
        assert bank.live == [10, 11, 12, 13]
        assert bank.visited == {0, 1}
        assert bank.prefix_lengths == {0: 5, 1: 3}
        kv0 = {layer: tuple(x.clone() for x in pair) for layer, pair in bank.buffers.items()}

        out1 = forward(model, registry, bank, [11, 13], ids([5, 6]))
        assert out1.logits.shape == (1, 2, 16)
        assert bank.live == [10, 11, 12, 13]
        assert bank.positions == [11, 13]
        for layer in (0, 1):
            k_store, _ = bank.buffers[layer]
            assert torch.equal(k_store[..., 0, :], kv0[layer][0][..., 0, :])
            assert torch.equal(k_store[..., 2, :], kv0[layer][0][..., 2, :])
            assert not torch.equal(k_store[..., 1, :], kv0[layer][0][..., 1, :])
            assert not torch.equal(k_store[..., 3, :], kv0[layer][0][..., 3, :])
            call = calls[-2 + layer]
            prefix_len = (5, 3)[layer]
            assert call["k"].shape[-2] == prefix_len + 4
            assert call["mask"].shape[-1] == prefix_len + 4
            assert torch.equal(
                call["k"][..., :prefix_len, :], model.model.decoder.layers[layer].prefix_k
            )
            assert call["mask"][..., prefix_len:].all()
            assert torch.equal(call["k"][..., prefix_len:, :], k_store[..., [0, 1, 2, 3], :])

        kv1 = {layer: tuple(x.clone() for x in pair) for layer, pair in bank.buffers.items()}
        out2 = forward(model, registry, bank, [10, 12], ids([7, 8]))
        assert out2.logits.shape == (1, 2, 16)
        assert bank.live == [10, 11, 12, 13]
        for layer in (0, 1):
            k_store, _ = bank.buffers[layer]
            assert torch.equal(k_store[..., 1, :], kv1[layer][0][..., 1, :])
            assert torch.equal(k_store[..., 3, :], kv1[layer][0][..., 3, :])
            assert not torch.equal(k_store[..., 0, :], kv1[layer][0][..., 0, :])
            assert not torch.equal(k_store[..., 2, :], kv1[layer][0][..., 2, :])


def test_block_causal_hides_tail_from_head_and_keeps_prefix(monkeypatch):
    model, registry, calls = make_world(monkeypatch)
    bank = gkv.WaveKV(4)
    a, b = [10, 11, 12, 13], [14, 15, 16, 17]
    with gkv.wave_attention(model, bank):
        forward(model, registry, bank, a, ids([1, 2, 3, 4]))
        forward(model, registry, bank, b, ids([5, 6, 7, 8]))
        out_a = forward(model, registry, bank, a, ids([1, 2, 3, 4]))
        m0, m1 = calls[-2]["mask"], calls[-1]["mask"]
        assert m0.shape[-1] == 5 + 8 and m1.shape[-1] == 3 + 8
        assert m0[..., 5:9].all() and not m0[..., 9:].any()
        assert m1[..., 3:7].all() and not m1[..., 7:].any()
        assert torch.equal(calls[-2]["k"][..., :5, :], model.model.decoder.layers[0].prefix_k)
        assert torch.equal(calls[-2]["v"][..., :5, :], model.model.decoder.layers[0].prefix_v)
        assert torch.equal(calls[-1]["k"][..., :3, :], model.model.decoder.layers[1].prefix_k)

        # Perturb the tail.  The head must stay bitwise unchanged while causal.
        forward(model, registry, bank, b, ids([9, 10, 11, 12]))
        out_a2 = forward(model, registry, bank, a, ids([1, 2, 3, 4]))
        assert torch.equal(out_a.logits, out_a2.logits)


def test_tail_sees_head_under_causal(monkeypatch):
    model, registry, calls = make_world(monkeypatch)
    bank = gkv.WaveKV(4)
    a, b = [10, 11, 12, 13], [14, 15, 16, 17]
    with gkv.wave_attention(model, bank):
        forward(model, registry, bank, a, ids([1, 2, 3, 4]))
        forward(model, registry, bank, b, ids([5, 6, 7, 8]))
        out_b1 = forward(model, registry, bank, b, ids([5, 6, 7, 8]))
        forward(model, registry, bank, a, ids([13, 14, 15, 0]))
        out_b2 = forward(model, registry, bank, b, ids([5, 6, 7, 8]))
        assert not torch.allclose(out_b1.logits, out_b2.logits)
        m0 = calls[-2]["mask"]
        assert m0[..., 5:9].all() and m0[..., 9:].all()


def test_causal_false_opens_ab_bidirectional(monkeypatch):
    model, registry, calls = make_world(monkeypatch)
    bank = gkv.WaveKV(4)
    a, b = [10, 11, 12, 13], [14, 15, 16, 17]
    with gkv.wave_attention(model, bank):
        forward(model, registry, bank, a, ids([1, 2, 3, 4]))
        forward(model, registry, bank, b, ids([5, 6, 7, 8]))
        out_a_causal = forward(model, registry, bank, a, ids([1, 2, 3, 4]))
        forward(model, registry, bank, b, ids([9, 10, 11, 12]))
        out_a_open = forward(model, registry, bank, a, ids([1, 2, 3, 4]), causal=False)
        assert not torch.allclose(out_a_causal.logits, out_a_open.logits)
        assert calls[-2]["mask"][..., 5:].all()


def test_parked_tail_kv_survives_three_head_steps_and_promotes(monkeypatch):
    model, registry, _ = make_world(monkeypatch)
    bank = gkv.WaveKV(4)
    a, b = [10, 11, 12, 13], [14, 15, 16, 17]
    with gkv.wave_attention(model, bank):
        forward(model, registry, bank, a, ids([1, 2, 3, 4]))
        out_b_initial = forward(model, registry, bank, b, ids([5, 6, 7, 8]))
        b_kv = {
            layer: (k[..., 4:8, :].clone(), v[..., 4:8, :].clone())
            for layer, (k, v) in bank.buffers.items()
        }
        for token in ([9, 10, 11, 12], [13, 14, 15, 0], [3, 4, 5, 6]):
            forward(model, registry, bank, a, ids(token))
            for layer, (k, v) in bank.buffers.items():
                assert torch.equal(k[..., 4:8, :], b_kv[layer][0])
                assert torch.equal(v[..., 4:8, :], b_kv[layer][1])

        a_kv = {
            layer: (k[..., 0:4, :].clone(), v[..., 0:4, :].clone())
            for layer, (k, v) in bank.buffers.items()
        }
        out_b_promoted = forward(model, registry, bank, b, ids([5, 6, 7, 8]))
        assert bank.live == [10, 11, 12, 13, 14, 15, 16, 17]
        assert bank.positions == [14, 15, 16, 17]
        for layer, (k, v) in bank.buffers.items():
            assert torch.equal(k[..., 0:4, :], a_kv[layer][0])
            assert torch.equal(v[..., 0:4, :], a_kv[layer][1])
        # Layer 0 K/V depends only on the token/position embeddings, so B's
        # first-layer rows are identical after re-projection.  Later layers may
        # legitimately change because layer 0 now reads A's latest K/V.
        k0, v0 = bank.buffers[0]
        assert torch.equal(k0[..., 4:8, :], b_kv[0][0])
        assert torch.equal(v0[..., 4:8, :], b_kv[0][1])
        # B now reads the head's latest K/V, so its logits differ from the
        # initial pass even though its own token IDs and positions are fixed.
        assert not torch.allclose(out_b_initial.logits, out_b_promoted.logits)


def test_drop_before_compacts_and_allows_new_request_origin(monkeypatch):
    model, registry, calls = make_world(monkeypatch)
    bank = gkv.WaveKV(4)
    a, b = [10, 11, 12, 13], [14, 15, 16, 17]
    with gkv.wave_attention(model, bank):
        forward(model, registry, bank, a, ids([1, 2, 3, 4]))
        forward(model, registry, bank, b, ids([5, 6, 7, 8]))
        b_kv = {
            layer: (k[..., 4:8, :].clone(), v[..., 4:8, :].clone())
            for layer, (k, v) in bank.buffers.items()
        }
        bank.drop_before(14)
        assert bank.live == [14, 15, 16, 17]
        assert bank.base == 14
        for layer, (k, v) in bank.buffers.items():
            assert torch.equal(k[..., :4, :], b_kv[layer][0])
            assert torch.equal(v[..., :4, :], b_kv[layer][1])
        with pytest.raises(ValueError):
            bank.begin([13], 10, CPU)

        # Simulate re-encoding A into the full-attention prefix.  The sliding
        # layer keeps its clipped native prefix length.
        layer0 = model.model.decoder.layers[0]
        layer0.prefix_k = torch.randn(1, 1, 9, 4, generator=torch.Generator().manual_seed(77))
        layer0.prefix_v = torch.randn(1, 1, 9, 4, generator=torch.Generator().manual_seed(78))
        out = forward(model, registry, bank, b, ids([5, 6, 7, 8]))
        assert out.logits.shape == (1, 4, 16)
        assert bank.live == [14, 15, 16, 17]
        assert bank.positions == [14, 15, 16, 17]
        assert calls[-2]["mask"].shape[-1] == 9 + 4
        assert calls[-1]["mask"].shape[-1] == 3 + 4
        assert calls[-2]["mask"][..., 9:].all()
        for layer, (k, v) in bank.buffers.items():
            if layer == 0:
                # Layer 0 K/V depends only on the token/position embeddings.
                assert torch.equal(k[..., :4, :], b_kv[layer][0])
                assert torch.equal(v[..., :4, :], b_kv[layer][1])
            # Later layers may change because the full-attention prefix grew.
            assert k.shape[-2] >= 4

        bank.drop_before(18)
        assert bank.live == [] and bank.base == 18
        bank.begin([30, 31, 32, 33], 30, CPU)
        assert bank.live == [30, 31, 32, 33]


def test_native_prefix_mask_and_per_layer_lengths_are_preserved(monkeypatch):
    model, registry, calls = make_world(monkeypatch)
    bank = gkv.WaveKV(4)
    a = [10, 11, 12, 13]
    masks = {}
    for layer in model.model.decoder.layers:
        prefix_len = layer.prefix_k.shape[-2]
        mask = torch.ones(1, 1, 4, prefix_len + 4, dtype=torch.bool)
        mask[..., 0] = False
        mask[..., prefix_len:] = True
        masks[layer.self_attn.layer_idx] = mask
    with gkv.wave_attention(model, bank):
        forward(model, registry, bank, a, ids([1, 2, 3, 4]), masks=masks)
        c0, c1 = calls[-2], calls[-1]
        assert c0["mask"].shape == (1, 1, 4, 9)
        assert c1["mask"].shape == (1, 1, 4, 7)
        assert torch.equal(c0["mask"][..., :5], masks[0][..., :5])
        assert torch.equal(c1["mask"][..., :3], masks[1][..., :3])
        assert c0["mask"][..., 5:].all()
        assert c1["mask"][..., 3:].all()
        assert torch.equal(c0["k"][..., :5, :], model.model.decoder.layers[0].prefix_k)
        assert torch.equal(c0["v"][..., :5, :], model.model.decoder.layers[0].prefix_v)
        assert torch.equal(c1["k"][..., :3, :], model.model.decoder.layers[1].prefix_k)
        assert torch.equal(c1["v"][..., :3, :], model.model.decoder.layers[1].prefix_v)


def test_none_native_mask_makes_prefix_visible(monkeypatch):
    model, registry, calls = make_world(monkeypatch)
    bank = gkv.WaveKV(4)
    with gkv.wave_attention(model, bank):
        forward(model, registry, bank, [10, 11, 12, 13], ids([1, 2, 3, 4]))
        assert calls[-2]["mask"][..., :5].all()
        assert calls[-1]["mask"][..., :3].all()


def test_block_size_one_is_token_causal(monkeypatch):
    model, registry, calls = make_world(monkeypatch)
    bank = gkv.WaveKV(1)
    with gkv.wave_attention(model, bank):
        forward(model, registry, bank, [10, 11, 12], ids([1, 2, 3]))
        live = calls[-2]["mask"][..., 5:]
        expected = torch.tril(torch.ones(3, 3, dtype=torch.bool))
        assert torch.equal(live[0, 0], expected)
        forward(model, registry, bank, [10, 11, 12], ids([1, 2, 3]), causal=False)
        assert calls[-2]["mask"][..., 5:].all()


def test_begin_validation():
    bank = gkv.WaveKV(4)
    with pytest.raises(ValueError):
        bank.begin([11, 10], 10, CPU)
    with pytest.raises(ValueError):
        bank.begin([10, 10], 10, CPU)
    with pytest.raises(ValueError):
        bank.begin([9], 10, CPU)
    with pytest.raises(ValueError):
        bank.begin([], 10, CPU)
    with pytest.raises(TypeError):
        bank.begin([10.0], 10, CPU)
    with pytest.raises(TypeError):
        bank.begin([10], 10, CPU, causal=1)
    bank.begin([10], 10, CPU)
    with pytest.raises(ValueError, match="stable"):
        bank.begin([10], 11, CPU)
    bank.begin([10, 11], 10, CPU)
    bank.drop_before(11)
    assert bank.live == [11]
    with pytest.raises(ValueError):
        bank.begin([10], 10, CPU)


def test_begin_rejects_uninitialized_resident_rows(monkeypatch):
    model, registry, _ = make_world(monkeypatch)
    bank = gkv.WaveKV(4)
    with gkv.wave_attention(model, bank):
        forward(model, registry, bank, [10, 11, 12, 13], ids([1, 2, 3, 4]))
        del bank.buffers[1]
        bank._written.pop(1, None)
        with pytest.raises(RuntimeError, match="missing resident"):
            bank.begin([11, 12, 13], 10, CPU)


def test_begin_rebases_for_a_gap_before_the_first_live_row():
    bank = gkv.WaveKV(4)
    bank._layer_count = 2
    bank.device = CPU
    bank.canvas_origin = 10
    bank.live = [10, 12]
    bank._base = 10
    bank._capacity = 4
    bank._written = {0: {10, 12}, 1: {10, 12}}
    for layer in (0, 1):
        bank.buffers[layer] = (
            torch.arange(4, dtype=torch.float32).view(1, 1, 4, 1) + layer,
            torch.arange(4, dtype=torch.float32).view(1, 1, 4, 1) + layer + 10,
        )
    bank.drop_before(11)
    assert bank.live == [12] and bank.base == 12 and bank._floor == 11

    # Row 11 was never resident, but it is at/after the drop floor and before
    # the first live row.  The bank must rebase storage instead of failing.
    bank.begin([11], 10, CPU)
    assert bank.live == [11, 12]
    assert bank.base == 11
    for layer in (0, 1):
        k_store, _ = bank.buffers[layer]
        assert k_store[..., 1, 0].item() == 2 + layer

    q = torch.zeros(1, 1, 1, 1)
    k = torch.cat((torch.zeros(1, 1, 5, 1), torch.full((1, 1, 1, 1), 99.0)), dim=2)
    v = torch.cat((torch.zeros(1, 1, 5, 1), torch.full((1, 1, 1, 1), 199.0)), dim=2)
    k_new, v_new, mask = bank.extend(0, q, k, v, None)
    assert k_new.shape[-2] == 5 + 2
    assert k_new[0, 0, 5, 0].item() == 99.0
    assert k_new[0, 0, 6, 0].item() == 2.0
    assert v_new[0, 0, 6, 0].item() == 12.0
    assert mask[0, 0, 0, 5:].all()


def test_bypass_passthrough_and_begin_required(monkeypatch):
    model, registry, _ = make_world(monkeypatch)
    bank = gkv.WaveKV(4)
    bank.bypass = True
    with gkv.wave_attention(model, bank):
        out = model.decoder(ids([1, 2]), pos([10, 11]), registry=registry)
        assert out.logits.shape == (1, 2, 16)
        assert bank.live == [] and bank.buffers == {}
    bank.bypass = False
    with gkv.wave_attention(model, bank):
        with pytest.raises(RuntimeError, match="begin"):
            model.decoder(ids([1, 2]), pos([10, 11]), registry=registry)


def test_wave_attention_only_patches_decoder_and_restores_on_exception(monkeypatch):
    model, registry, calls = make_world(monkeypatch)
    dec = [layer.self_attn.config for layer in model.model.decoder.layers]
    enc = [layer.self_attn.config for layer in model.model.encoder.layers]
    dec_impl = [config._attn_implementation for config in dec]
    enc_impl = [config._attn_implementation for config in enc]
    bank = gkv.WaveKV(4)
    with pytest.raises(RuntimeError, match="boom"):
        with gkv.wave_attention(model, bank) as yielded:
            assert yielded is bank
            for i, layer in enumerate(model.model.decoder.layers):
                assert layer.self_attn.config._attn_implementation == ALIAS
                assert layer.self_attn.config is not dec[i]
            for i, layer in enumerate(model.model.encoder.layers):
                assert layer.self_attn.config is enc[i]
                assert layer.self_attn.config._attn_implementation == "sdpa"
            model.model.encoder(ids([1, 2]), pos([0, 1]), registry=registry)
            assert calls[-1]["tag"] == "encoder"
            assert calls[-1]["impl"] == "sdpa"
            raise RuntimeError("boom")
    assert ALIAS not in registry
    for i, layer in enumerate(model.model.decoder.layers):
        assert layer.self_attn.config is dec[i]
        assert layer.self_attn.config._attn_implementation == dec_impl[i]
    for i, layer in enumerate(model.model.encoder.layers):
        assert layer.self_attn.config is enc[i]
        assert layer.self_attn.config._attn_implementation == enc_impl[i]


def test_wave_attention_normal_exit_restores(monkeypatch):
    model, registry, _ = make_world(monkeypatch)
    dec = [layer.self_attn.config for layer in model.model.decoder.layers]
    dec_impl = [config._attn_implementation for config in dec]
    bank = gkv.WaveKV(4)
    with gkv.wave_attention(model, bank):
        assert ALIAS in registry
    assert ALIAS not in registry
    for i, layer in enumerate(model.model.decoder.layers):
        assert layer.self_attn.config is dec[i]
        assert layer.self_attn.config._attn_implementation == dec_impl[i]


def test_wave_attention_requires_sdpa_and_restores_partial_patch(monkeypatch):
    model, registry, _ = make_world(monkeypatch)
    model.model.decoder.layers[1].self_attn.config._attn_implementation = "eager"
    dec = [layer.self_attn.config for layer in model.model.decoder.layers]
    dec_impl = [config._attn_implementation for config in dec]
    bank = gkv.WaveKV(4)
    with pytest.raises(ValueError):
        with gkv.wave_attention(model, bank):
            pass
    assert ALIAS not in registry
    for i, layer in enumerate(model.model.decoder.layers):
        assert layer.self_attn.config is dec[i]
        assert layer.self_attn.config._attn_implementation == dec_impl[i]


def test_wave_attention_rejects_nested_alias(monkeypatch):
    model, registry, _ = make_world(monkeypatch)
    bank = gkv.WaveKV(4)
    with gkv.wave_attention(model, bank):
        with pytest.raises(RuntimeError, match="already active"):
            with gkv.wave_attention(model, gkv.WaveKV(4)):
                pass
    assert ALIAS not in registry


def test_real_diffusion_gemma_attention_interface_alias_roundtrip():
    pytest.importorskip("transformers.models.diffusion_gemma.modeling_diffusion_gemma")
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        ALL_ATTENTION_FUNCTIONS,
    )

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = NS(_attn_implementation="sdpa")
            self.layer_idx = 0

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attn()

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer()])

    model = NS(model=NS(decoder=Decoder()))
    bank = gkv.WaveKV(4)
    with gkv.wave_attention(model, bank):
        assert ALIAS in ALL_ATTENTION_FUNCTIONS
        assert model.model.decoder.layers[0].self_attn.config._attn_implementation == ALIAS
    assert ALIAS not in ALL_ATTENTION_FUNCTIONS
    assert model.model.decoder.layers[0].self_attn.config._attn_implementation == "sdpa"

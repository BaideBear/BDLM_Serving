"""CPU integration contract using real pinned Gemma projections, not checkpoint parity."""
from collections import defaultdict
from contextlib import ExitStack
from itertools import chain

import pytest
import torch

hf = pytest.importorskip("transformers")
if hf.__version__ != "5.12.0":
    pytest.skip("requires the pinned Transformers 5.12.0", allow_module_level=True)

from experiments.t2t.serving_slots.gemma_strict_request import generate


B, V = 256, 64
PROMPT = [2, 8, 9, 13, 21]


def inputs():
    ids = torch.tensor([PROMPT])
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


@pytest.fixture(scope="module")
def model():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = hf.DiffusionGemmaConfig(
            canvas_length=B,
            vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                               num_attention_heads=2, num_key_value_heads=2, head_dim=8),
            text_config=dict(vocab_size=V, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_experts=4, top_k_experts=2,
                             moe_intermediate_size=8, num_attention_heads=2,
                             num_key_value_heads=1, num_global_key_value_heads=1,
                             head_dim=8, global_head_dim=8, sliding_window=12,
                             layer_types=["sliding_attention", "full_attention"]))
        cfg._attn_implementation = "sdpa"
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.default_generator.manual_seed(109)
            net = hf.DiffusionGemmaForBlockDiffusion(cfg).eval()
        net.generation_config.max_denoising_steps = 48
        net.generation_config.eos_token_id = [V + 1]  # unreachable unless a test overrides it
        yield net
    finally:
        torch.set_num_threads(old_threads)


@torch.inference_mode()
def audited(model, offers, *, limit=B, blocks=1, quota=0, seed=19):
    events, positions, logits_top1, encoder_ids = [], [], [], []
    widths = {}

    def encoder(_m, _args, kw):
        assert kw["input_ids"].shape[0] == 1
        encoder_ids.append(kw["input_ids"][0].tolist())
        events.append("E")

    def decoder(_m, _args, kw):
        ids, pos = kw["decoder_input_ids"], kw["decoder_position_ids"]
        assert ids.shape == pos.shape and ids.shape[0] == 1
        positions.append(pos[0].tolist())
        events.append("D")

    def returned_logits(_m, _args, out):
        assert out.logits.shape == (1, len(positions[-1]), V)
        logits_top1.append(out.logits[0].argmax(-1).tolist())

    def width_hook(name):
        tape = widths[name] = []

        def hook(_m, args):
            x, = args
            assert x.ndim == 3 and x.shape[0] == 1 and x.device.type == "cpu"
            tape.append(x.shape[1])
        return hook

    with ExitStack() as stack:
        stack.callback(model.model.encoder.register_forward_pre_hook(encoder, with_kwargs=True).remove)
        stack.callback(model.model.decoder.register_forward_pre_hook(decoder, with_kwargs=True).remove)
        stack.callback(model.register_forward_hook(returned_logits).remove)
        for i, layer in enumerate(model.model.decoder.layers):
            for key in ("q_proj", "k_proj", "v_proj"):
                proj = getattr(layer.self_attn, key)
                if proj is None and key == "v_proj":  # Gemma may share KV
                    continue
                stack.callback(proj.register_forward_pre_hook(width_hook((i, key))).remove)
        stack.callback(model.lm_head.register_forward_pre_hook(width_hook("lm_head")).remove)
        tokens, trace = generate(model, inputs(), seed=seed, offers=offers, prefix_quota=quota,
                                 prefix_mode="right", max_new_tokens=limit)

    assert trace and events == ["E", *chain.from_iterable(
        ("D", "E") if e["encoded"] else ("D",) for e in trace)]
    assert len(positions) == len(logits_top1) == len(trace)
    assert all(tape == [e["q"] for e in trace] for tape in widths.values())
    sealed = [dict() for _ in range(blocks)]
    accepted = [set() for _ in range(blocks)]
    samples = defaultdict(int)
    ordinal = 0
    for tick, (e, pos, top1) in enumerate(zip(trace, positions, logits_top1, strict=True)):
        assert ordinal < blocks and e["ordinal"] == ordinal
        assert type(e["encoded"]) is bool and e["offer"] == offers[tick % len(offers)]
        rows = list(e["rows"])
        assert rows == sorted(set(rows)) and all(type(r) is int and 0 <= r < B for r in rows)
        assert 0 < len(rows) == e["q"] <= e["offer"]
        assert pos == [len(PROMPT) + B*ordinal + r for r in rows]
        before = accepted[ordinal]
        fields = [e[k] for k in ("prefix", "admitted", "edited", "revoked")]
        for group in fields:
            assert len(group) == len(set(group)) and set(group) <= set(rows)
        prefix, add, edit, revoke = map(set, fields)
        assert len(prefix) <= quota and prefix <= before
        assert not (add & before) and (edit | revoke) <= before and not (edit & revoke)
        before.difference_update(revoke)
        before.update(add)
        phase = e["phase"]
        assert phase in ("sample", "drain", "verify")
        if phase != "sample":
            assert not any(fields), "drain/verify cannot publish sampler actions"
        else:
            samples[ordinal] += 1
        if phase == "verify":
            assert not (sealed[ordinal].keys() & set(rows)), "duplicate terminal row"
            sealed[ordinal].update(zip(rows, top1, strict=True))
        else:
            assert not sealed[ordinal], "sample/drain after terminal sweep began"
        complete = len(sealed[ordinal]) == B
        assert e["encoded"] == (complete and ordinal < blocks-1)
        if complete:
            ordinal += 1
    assert ordinal == blocks and all(1 <= samples[i] <= 48 for i in range(blocks))
    drafts = [[block[r] for r in range(B)] for block in sealed]
    assert encoder_ids == [PROMPT, *drafts[:-1]]
    expected = list(chain.from_iterable(drafts))
    eos = set(model.generation_config.eos_token_id)
    end = next((i+1 for i, t in enumerate(expected) if t in eos), len(expected))
    assert list(tokens) == expected[:end]
    return list(tokens), trace


def test_fixed_offer_replays_tokens_trace_and_eos(model, monkeypatch):
    tokens, trace = audited(model, (32,))
    assert len(tokens) == B
    # Replay with different ambient CPU RNG. Only the explicit seed should matter.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(42)
        assert audited(model, (32,)) == (tokens, trace)
    # No fake logits: use a naturally occurring token as EOS, and check that
    # even with a further block allowed the terminal encoder is never invoked.
    stop = next((t for t in tokens[1:] if t != tokens[0]), tokens[0])
    monkeypatch.setattr(model.generation_config, "eos_token_id", [stop])
    short, _ = audited(model, (32,), limit=2*B)
    assert short == tokens[:tokens.index(stop)+1]


def test_cyclic_external_offers_two_blocks(model):
    offers = (128, 32, 64, 32, 64)
    tokens, trace = audited(model, offers, limit=2*B, blocks=2, quota=4)
    assert len(tokens) == 2*B
    assert {e["phase"] for e in trace} == {"sample", "drain", "verify"}


@pytest.mark.parametrize("bad", [dict(offers=()), dict(offers=(32, 512)), dict(offers=(True,)),
                                 dict(max_new_tokens=B+1), dict(prefix_quota=-1),
                                 dict(prefix_mode="unsupported"), dict(seed=True)])
def test_invalid_settings(model, bad):
    kwargs = dict(seed=19, offers=(32,), max_new_tokens=B)
    with pytest.raises((TypeError, ValueError)):
        generate(model, inputs(), **{**kwargs, **bad})


@pytest.mark.parametrize("kind", ["batch", "padded", "non_integer_ids"])
def test_invalid_prompt(model, kind):
    data = inputs()
    if kind == "batch":
        data = {k: v.repeat(2, 1) for k, v in data.items()}
    elif kind == "padded":
        data["attention_mask"][0, -1] = 0
    else:
        data["input_ids"] = data["input_ids"].float()
    with pytest.raises((TypeError, ValueError)):
        generate(model, data, seed=19, offers=(32,), max_new_tokens=B)

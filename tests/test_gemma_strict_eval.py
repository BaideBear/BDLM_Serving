from types import SimpleNamespace

import pytest
import torch

from experiments.t2t.serving_slots import gemma_strict_eval as ev


def fixture(blocks=2, k=0):
    """Two cold cohorts, one warm sample with real slack, refresh and close."""
    prompt = [8, 9, 10]
    tokens = [7] * (256*(blocks-1)) + [9, 0, 1]
    trace, widths, pos = [], [], []
    events, enc = [("encoder", 3)], [(prompt, list(range(3)))]
    for b in range(blocks):
        base = len(prompt) + b*256

        def add(rows, phase, *, sampled=(), admitted=(), revoked=(), dirty=(), prefix=(), encoded=False):
            rows = list(rows)
            w = len(rows)
            trace.append(dict(q=w, offer=128, ordinal=b, phase=phase, rows=rows, prefix=list(prefix),
                              sampled=list(sampled), admitted=list(admitted), revoked=list(revoked),
                              edited=[], dirty_before=list(dirty), encoded=encoded))
            widths.append(w)
            pos.append([base+r for r in rows])
            events.append(("decoder", w))

        a, z = range(128), range(128, 256)
        add(a, "sample", sampled=a, admitted=a)
        add(a, "drain", dirty=a)
        add(z, "sample", sampled=z, admitted=range(128, 250))
        # Previous sample changed the IDs of all 128 rows, including noisy rows.
        add(z, "drain", dirty=z)
        p = list(range(250-k, 250))
        w = p + list(range(250, 256))
        add(w, "sample", sampled=w, admitted=range(250, 256), prefix=p)
        add(range(250, 256), "drain", dirty=range(250, 256))
        add(a, "verify")
        add(z, "verify", encoded=b+1 < blocks)
        if b+1 < blocks:
            events.append(("encoder", 256))
            enc.append((tokens[256*b:256*(b+1)], list(range(base, base+256))))
    return dict(prompt=prompt, tokens=tokens, trace=trace, projections={(0, "q_proj"): widths},
                positions=pos, events=events, encoders=enc, q=128, k=k, eos={1}, special={0, 1}, pad=0)


@pytest.mark.parametrize("k", [0, 4])
def test_two_blocks_paid_and_opportunities(k):
    m = ev.audit(**fixture(k=k))
    assert (m["N"], m["F"], m["Q"], m["prefill_rows"]) == (257, 17, 1816 + 2*k, 3)
    assert m["N_per_Q"] == 257/m["Q"]
    assert (m["phase_Q"]["verify"], m["encoder_Q"]) == (512, 256)
    assert m["eligible_warm_calls"] == 2 and m["eligible_warm_seats"] == 244
    assert m["prefix_rows"] == 2*k and m["prefix_calls"] == (2 if k else 0)
    assert m["eos"] and not m["truncated"]


def test_cap_without_eos():
    a = fixture(blocks=8)
    a["tokens"] = [7]*2048
    assert ev.audit(**a)["truncated"]


@pytest.mark.parametrize("bad", ["layer", "decoder_position", "encoder_token", "encoder_position",
                                 "prefill_token", "unreported_call", "verify_order", "overspend",
                                 "encoder_flag", "ordinal", "drain_write", "missing_eos",
                                 "early_eos", "false_prefix", "terminal_coverage", "hidden_edit"])
def test_reject_corruption(bad):
    a = fixture()
    t = a["trace"]
    if bad == "layer":
        a["projections"][(1, "v_proj")] = [256]*len(t)
    elif bad == "decoder_position":
        a["positions"][-1][-1] -= 1
    elif bad == "encoder_token":
        a["encoders"][1][0][0] += 1
    elif bad == "encoder_position":
        a["encoders"][1][1][0] += 1
    elif bad == "prefill_token":
        a["encoders"][0] = ([9, 9, 10], list(range(3)))
    elif bad == "unreported_call":
        a["events"].append(("decoder", 128))
    elif bad == "verify_order":
        a["events"][8:10] = reversed(a["events"][8:10])
    elif bad == "overspend":
        t[0]["offer"] = 32
    elif bad == "encoder_flag":
        t[-1]["encoded"] = True
    elif bad == "ordinal":
        t[-1]["ordinal"] = 0
    elif bad == "drain_write":
        t[1]["edited"] = [0]
    elif bad == "missing_eos":
        a["tokens"][-1] = 99
    elif bad == "early_eos":
        a["tokens"][0] = 1
    elif bad == "false_prefix":
        t[4]["prefix"] = [250]
    elif bad == "terminal_coverage":
        t[-1]["rows"] = list(range(128))
        a["positions"][-1] = a["positions"][-2][:]
    elif bad == "hidden_edit":
        t[4]["edited"] = [2]
    with pytest.raises(ValueError):
        ev.audit(**a)


@pytest.mark.parametrize("args", [(512, 0, 0), (32, 33, 10), (32, -1, 20), (128, 0, 1)])
def test_arm_gate(args):
    with pytest.raises(ValueError):
        ev.check_arm(*args)


def test_tiny_real_request(monkeypatch):
    from transformers import BatchEncoding, DiffusionGemmaConfig, DiffusionGemmaForBlockDiffusion
    cfg = DiffusionGemmaConfig(canvas_length=256,
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                           num_attention_heads=4, num_key_value_heads=4, head_dim=8),
        text_config=dict(vocab_size=128, hidden_size=32, num_attention_heads=4, num_key_value_heads=2,
                         num_global_key_value_heads=2, head_dim=8, global_head_dim=8, num_experts=4,
                         top_k_experts=2, moe_intermediate_size=32, num_hidden_layers=2,
                         intermediate_size=64, layer_types=["sliding_attention", "full_attention"],
                         max_position_embeddings=4096))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        model = DiffusionGemmaForBlockDiffusion(cfg).to(dtype=torch.bfloat16).eval()
    # Force EOS on the first committed token, not before paying for terminal verify.
    model.generation_config.eos_token_id = list(range(128))
    model.generation_config.max_denoising_steps = 48
    ids = torch.tensor([[2, 3, 4]])
    processor = SimpleNamespace(
        apply_chat_template=lambda *a, **kw: BatchEncoding(dict(input_ids=ids, attention_mask=ids*0+1)),
        tokenizer=SimpleNamespace(all_special_ids=[0], decode=lambda ids, **kw: "42"))
    task = dict(index=0, task_id="test", seed=17000, dataset="gsm8k", prompt="test", reference="42")
    metrics, raw = ev.evaluate(model, processor, task)
    assert metrics["layers"] == 2 and metrics["correct"] and metrics["metric"] == "numeric_exact_match"
    assert metrics["phase_Q"]["verify"] == 256 and metrics["decoder_q_max"] == 128
    assert len(raw["output_ids"]) == 1 and metrics["N"] == 0
    assert raw["encoder_inputs"] == [([2, 3, 4], [0, 1, 2])]
    assert metrics["Q"] == sum(len(p) for p in raw["decoder_position_ids"])
    assert "trace" not in metrics and not model.model.encoder._forward_pre_hooks

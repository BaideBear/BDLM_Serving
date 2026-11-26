from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bdlm_serving.runtime import diffusiongemma_wave as wave
from experiments.t2t.serving_slots import mask_pair_gemma as pair


def cache_with_lengths(*lengths):
    return SimpleNamespace(layers=[
        SimpleNamespace(keys=torch.zeros(1, 1, n, 8), values=torch.zeros(1, 1, n, 8))
        for n in lengths])


def test_exact_nominal_q512_mask_delta_and_single_identity():
    kinds = ["sliding_attention", "full_attention", "sliding_attention"]
    cache = cache_with_lengths(13, 103, 13)
    b = 256
    one = wave.packing_mask(cache, kinds, b, 1, "cpu")
    for opened in (False, True):
        assert pair.mask_for_arm(one, cache, kinds, b, b, opened=opened) is one
    native = wave.packing_mask(cache, kinds, b, 2, "cpu")
    assert pair.mask_for_arm(native, cache, kinds, b, 512, opened=False) is native
    off = pair.mask_for_arm(native, cache, kinds, b, 512, opened=True)
    for kind, mask in native.items():
        delta = torch.zeros_like(mask)
        delta[..., :b, -b:] = True
        assert torch.equal(mask ^ off[kind], delta)
        assert off[kind].all() and not mask[..., :b, -b:].any()
    bad = {**native, "full_attention": off["full_attention"]}
    with pytest.raises(ValueError, match="isolate head"):
        pair.mask_for_arm(bad, cache, kinds, b, 512, opened=False)
    with pytest.raises(ValueError, match="shape or dtype"):
        pair.mask_for_arm({k: v.float() for k, v in native.items()}, cache, kinds, b, 512, opened=True)
    with pytest.raises(ValueError, match="one or two"):
        pair.mask_for_arm(native, cache, kinds, b, 768, opened=True)


def gemma_modules():
    # The standard repository test env need not include the pinned Gemma fork.
    return pytest.importorskip("transformers.models.diffusion_gemma.modeling_diffusion_gemma")


def tiny_text_config(gemma):
    cfg = gemma.DiffusionGemmaTextConfig(
        vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_experts=4, top_k_experts=2, moe_intermediate_size=8,
        num_attention_heads=2, num_key_value_heads=1, num_global_key_value_heads=1,
        head_dim=8, global_head_dim=8, sliding_window=12,
        layer_types=["sliding_attention", "full_attention"])
    cfg._attn_implementation = "sdpa"
    return cfg


@pytest.mark.parametrize("layer", (0, 1), ids=("sliding", "full"))
def test_real_gemma_sdpa_tail_influence(layer):
    gemma = gemma_modules()
    cfg = tiny_text_config(gemma)
    torch.manual_seed(25)
    attn = gemma.DiffusionGemmaDecoderTextAttention(cfg, layer).eval()
    b = 256
    cache = cache_with_lengths(5, 13)
    native = wave.packing_mask(cache, cfg.layer_types, b, 2, "cpu")
    opened = pair.mask_for_arm(native, cache, cfg.layer_types, b, 2 * b, opened=True)
    kind = cfg.layer_types[layer]
    hidden = torch.randn(1, 2 * b, cfg.hidden_size)
    changed = hidden.clone()
    changed[:, b:] = torch.randn_like(changed[:, b:])
    rot = torch.ones(1, 2 * b, 8), torch.zeros(1, 2 * b, 8)
    with torch.inference_mode():
        on_a = attn(hidden, rot, native[kind], cache)[0]
        on_b = attn(changed, rot, native[kind], cache)[0]
        off_a = attn(hidden, rot, opened[kind], cache)[0]
        off_b = attn(changed, rot, opened[kind], cache)[0]
    torch.testing.assert_close(on_a[:, :b], on_b[:, :b], atol=0, rtol=0)
    assert (off_a[:, :b] - off_b[:, :b]).abs().max() > 1e-4
    torch.testing.assert_close(on_a[:, b:], off_a[:, b:], atol=0, rtol=0)


@pytest.mark.parametrize("cap", (8, 24), ids=("single", "dual"))
def test_real_active_w2_paired_prefix_and_observed_work(cap):
    gemma = gemma_modules()
    from transformers import Gemma4VisionConfig

    torch.manual_seed(37)
    vision = Gemma4VisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                               num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                               position_embedding_size=32)
    cfg = gemma.DiffusionGemmaConfig(text_config=tiny_text_config(gemma), vision_config=vision,
                                     canvas_length=8)
    cfg._attn_implementation = "sdpa"
    model = gemma.DiffusionGemmaForBlockDiffusion(cfg).eval()
    model.generation_config.max_denoising_steps = 4
    model.generation_config.pad_token_id = 0
    model.generation_config.bos_token_id = 2
    # Avoid random early EOS; the test must exercise the intended widths.
    model.generation_config.eos_token_id = [-1]
    data = dict(input_ids=torch.tensor([[2, 10, 11]]), attention_mask=torch.ones(1, 3))
    plan = dict(block_size=8, cap=cap, flowblock_spawn_threshold=0)
    task = dict(index=0, task_id="cpu", seed=19, dataset="unit")
    tokenizer = SimpleNamespace(decode=lambda ids, **_: str(ids))
    rows, probes = [], []
    with torch.inference_mode():
        # Serialization mirrors the run receipt before the protected handoff.
        effective, _ = model._prepare_generation_config(None, max_new_tokens=cap)
        sampling = pair.generation_settings(effective)
        json.dumps(sampling)
        assert sampling["sampler_config"]["entropy_bound"] > 0
        for policy in ("active_w2", *pair.POLICIES):
            torch.manual_seed(task["seed"])
            probe_ctx = (nullcontext() if policy == "active_w2"
                         else pair.decoder_probe(model, 8, opened=policy == pair.MASK_OFF))
            with wave.observe(model) as events, probe_ctx as probe:
                ids, trace = wave.generate(model, data, "active_w2", plan)
            rows.append(pair.make_row(task, policy, ids, trace, events, 3, tokenizer, {-1}, {0}, 8))
            probes.append(probe)
    pair.equal_row(rows[1], rows[0], "unhooked CPU baseline")
    audit = pair.check_pair(*rows[1:], *probes[1:], 8)
    assert audit["no_second_block_exact"] == (cap == 8)
    if cap == 24:
        assert audit["first_intervention_step"] == 1
        assert min(audit["dual_block_steps_on"], audit["dual_block_steps_off"]) >= 1
    else:
        assert audit["first_intervention_step"] is None
        assert audit["dual_block_steps_on"] == audit["dual_block_steps_off"] == 0
        with pytest.raises(ValueError, match="no-second-block"):
            pair.check_pair(rows[1], dict(rows[2], generation_query_rows=0), *probes[1:], 8)
    with pytest.raises(ValueError, match="before first intervention"):
        pair.check_pair(*rows[1:], probes[1], dict(probes[2], pre_intervention_sha256="bad"), 8)
    assert not model.model.decoder._forward_pre_hooks and not model.model.encoder._forward_pre_hooks


def test_directional_probes_report_leak_without_changing_request_rng(tmp_path, capsys):
    rows = []

    def measure(model, data, task, *, expert_layout):
        assert model == "shared" and task["index"] == 0 and data["input_ids"].device.type == "cpu"
        rows.append((expert_layout, torch.rand(1).item()))
        return dict(results=[
            dict(mask="block_causal", changed="tail_first", read="head", max_abs=0.1),
            dict(mask="bidirectional", changed="tail_first", read="head", max_abs=1.0),
        ], diagnostics={"block_causal": {"same_input": {"max_abs": 0}}})

    rng = torch.get_rng_state()
    pair.run_directional_probes("shared", {"input_ids": torch.zeros(1, 2)}, {"index": 0},
                                tmp_path, measure, lambda p, report: p.write_text(json.dumps(report)))
    assert torch.equal(torch.get_rng_state(), rng)
    assert rows == [("packed", rows[0][1]), ("per_canvas", rows[0][1])]
    log = capsys.readouterr().out
    for layout in ("packed", "per_canvas"):
        assert (tmp_path / f"directional-{layout}.json").is_file()
        assert f"WARNING directional {layout}" in log


def test_launcher_guard_and_callsite_only(tmp_path, monkeypatch):
    launcher = tmp_path / "launch_prefix90.py"
    src = b"""protected = True
if False:
        run_wave(args, before_cuda=prepare)
"""
    launcher.write_bytes(src)
    with pytest.raises(ValueError, match="reviewed"):
        pair.protected_source(launcher)
    monkeypatch.setattr(pair, "LAUNCHER_SHA", hashlib.sha256(src).hexdigest())
    code = pair.protected_source(launcher)
    ns = {}
    exec(code, ns)
    assert ns["protected"] is True
    monkeypatch.setattr(pair.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    with pytest.raises(RuntimeError, match="still active"):
        pair.require_switch_finished(tmp_path)


def test_installed_protected_source_compiles():
    repo = Path(pair.__file__).resolve().parents[3]
    launcher = repo.parent / "t2t-native-1125/launch_prefix90.py"
    if not launcher.is_file():
        pytest.skip("n164 protected launcher not installed here")
    assert pair.protected_source(launcher).co_filename == str(launcher)


def test_predecessor_metadata_and_protected_completion(tmp_path):
    from bdlm_serving.methods.t2t.wave_offer import TRACES

    def put(path, payload):
        path.write_text(json.dumps(payload))

    active = pair.active_settings(dict(suite="frozen90", samples_sha256=pair.SAMPLES_SHA))
    ref = tmp_path / pair.REFERENCE
    switch = tmp_path / "gemma-external-256-512"
    ref.mkdir()
    switch.mkdir()
    put(ref / "run.json", active)
    meta = dict(active, policies=list(TRACES), offer_sequences=TRACES, nominal_q=None,
                qualification="flat W1 and flat active W2 output, events and original wave trace")
    put(switch / "run.json", meta)
    for folder, labels in ((ref, active["policies"]), (switch, TRACES), (tmp_path, pair.POLICIES)):
        for policy in labels:
            put(folder / f"complete-{policy}.json", dict(policy=policy, requests=90))
    pair.require_switch_artifacts(tmp_path, active)
    put(switch / "run.json", dict(meta, seed_offset=1))
    with pytest.raises(ValueError, match="Switch predecessor settings"):
        pair.require_switch_artifacts(tmp_path, active)
    marker = tmp_path / "protected-complete.json"
    put(tmp_path / "qualified.json", dict(requests=90, historical_replayed=90, exposed=4,
                                         no_second_block_exact=87))
    with pytest.raises(ValueError, match="Missing qualified"):
        pair.mark_protected_complete(tmp_path)
    assert not marker.exists()
    put(tmp_path / "qualified.json", dict(requests=90, historical_replayed=90, exposed=4,
                                         no_second_block_exact=86))
    pair.mark_protected_complete(tmp_path)
    assert pair.read_json(marker) == dict(status="complete", requests=90, gpu=0, holder_restored=True)

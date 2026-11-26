import pytest
import torch

from bdlm_serving.runtime import diffusiongemma_wave as wave
from bdlm_serving.runtime.diffusiongemma import observe
from experiments.t2t.serving_slots import gemma_trigger_probe as trigger
from experiments.t2t.serving_slots import mask_pair_gemma as pair


@pytest.fixture
def tiny():
    gemma = pytest.importorskip("transformers.models.diffusion_gemma.modeling_diffusion_gemma")
    from transformers import Gemma4VisionConfig

    block = 8
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(37)
        text = gemma.DiffusionGemmaTextConfig(
            vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_experts=4, top_k_experts=2, moe_intermediate_size=8,
            num_attention_heads=2, num_key_value_heads=1, num_global_key_value_heads=1,
            head_dim=8, global_head_dim=8, sliding_window=12,
            layer_types=["sliding_attention", "full_attention"])
        vision = Gemma4VisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                   num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                                   position_embedding_size=32)
        cfg = gemma.DiffusionGemmaConfig(text_config=text, vision_config=vision, canvas_length=block)
        cfg._attn_implementation = "sdpa"
        model = gemma.DiffusionGemmaForBlockDiffusion(cfg).eval()
    model.generation_config.max_denoising_steps = 4
    model.generation_config.pad_token_id = 0
    model.generation_config.bos_token_id = 2
    model.generation_config.eos_token_id = [-1]
    data = dict(input_ids=torch.tensor([[2, 10, 11]]), attention_mask=torch.ones(1, 3))
    return model, data, dict(index=0, seed=19), dict(
        block_size=block, cap=3 * block, flowblock_spawn_threshold=0)


def test_real_trigger_replays_rng_cache_and_effective_self_conditioning(tiny):
    model, data, task, plan = tiny
    block = plan["block_size"]

    def scored():
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.manual_seed(task["seed"])
            with observe(model) as work, pair.decoder_probe(model, block, opened=False) as audit:
                sequence, trace = wave.generate(model, data, "active_w2", plan)
            return sequence, trace, work, audit, torch.get_rng_state().tolist()

    before = scored()
    outer_rng = torch.get_rng_state()
    observed = []

    def record(_, args, kw):
        if kw["decoder_input_ids"].shape[-1] == 2 * block:
            observed.append(torch.get_rng_state())

    hook = model.model.decoder.register_forward_pre_hook(record, with_kwargs=True)
    try:
        report = trigger.probe(model, data, task, plan)
    finally:
        hook.remove()

    assert report["status"] == "measured"
    assert report["first_dual_step"] == before[3]["first_dual_block_step"] == 1
    assert report["cache_rows_by_layer"] == [3, 3]
    assert report["has_sc_logits"] and report["effective_tail_soft_is_zero"]
    assert report["effective_soft_nonzero_rows"] == [block]
    assert report["anchor"]["exact"]
    assert all(v["exact"] and v["first_layer_diff"] is None for v in report["sham"].values())
    assert report["tail_effect"]["native"]["exact"]
    assert report["tail_effect"]["off"]["first_layer_diff"] == "0/attn"
    assert len(observed) == 7 and all(torch.equal(observed[0], step) for step in observed)
    assert torch.equal(outer_rng, torch.get_rng_state())
    assert before == scored()


def test_bad_sham_is_inconclusive(tiny, monkeypatch):
    model, data, task, plan = tiny
    edit = trigger._edited_mask
    monkeypatch.setattr(trigger, "_edited_mask", lambda mask, block, opened: edit(mask, block, True))
    report = trigger.probe(model, data, task, plan)
    assert report["status"] == "inconclusive"
    assert not report["sham"]["original"]["exact"]
    assert report["opened_tail_positive"]


def test_capture_limit_does_not_replay(tiny):
    model, data, task, plan = tiny
    saved = torch.get_rng_state()
    report = trigger.probe(model, data, task, plan, max_decoder_steps=1)
    assert report["status"] == "no_trigger"
    assert report["capture_decoder_forwards"] == 1
    assert "replay_forwards" not in report
    assert torch.equal(saved, torch.get_rng_state())

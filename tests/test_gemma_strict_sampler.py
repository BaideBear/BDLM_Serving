"""CPU properties against pinned Transformers, never benchmark or weight parity."""
from contextlib import ExitStack
from types import SimpleNamespace

import pytest
import torch

native = pytest.importorskip("transformers.models.diffusion_gemma.generation_diffusion_gemma")
from experiments.t2t.serving_slots.gemma_strict_sampler import SelectedRowSampler
from experiments.t2t.serving_slots.gemma_wave_kv import WaveKV, wave_attention


def sharp(ids, token, vocab=8):
    return 40.0 * torch.nn.functional.one_hot(torch.full_like(ids, token), vocab).float()


def test_full_width_native_step_and_default_rng_control():
    from transformers.generation.logits_process import LogitsProcessorList

    # Same real toy LM head for the selected adapter and native dense reference.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(10)
        head = torch.nn.Embedding(8, 8)

    def project(rows, ids, sc):
        logits = head(ids).float()
        return logits if sc is None else logits + 0.25*sc.float()

    s = SelectedRowSampler(vocab_size=8, cap=256, seed=17)
    sampler = native.EntropyBoundSampler(native.EntropyBoundSamplerConfig(0.1), 256, 8, 48)
    temp = LogitsProcessorList([native.LinearTemperatureScheduleLogitsProcessor(0.4, 0.8, 48)])
    model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(vocab_size=8)),
                            model=SimpleNamespace(decoder=SimpleNamespace(
                                embed_tokens=SimpleNamespace(weight=torch.empty(0, dtype=s.dtype)))))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        ids = draft = sampler.initialize_canvas(1, "cpu")
        assert torch.equal(ids, s.ids)
        sc = None
        for t in (48, 47):
            before = torch.get_rng_state()
            s.step(range(256), project)
            assert torch.equal(before, torch.get_rng_state())
            ids, draft, sc, _ = native.DiffusionGemmaGenerationMixin._denoising_step(
                model, decoder_forward=lambda **kw: SimpleNamespace(logits=project(
                    range(256), kw["decoder_input_ids"], kw["self_conditioning_logits"])),
                current_canvas=ids, argmax_canvas=draft, input_ids=ids[:, :1], decoder_position_ids=None,
                self_conditioning_logits=sc, mask_mapping={}, past_key_values=None,
                finished_denoising=torch.tensor([False]), cur_step=t, sampler=sampler,
                logits_processor=temp, diffusion_stopping_criteria=None)
            assert torch.equal(s.ids, ids) and torch.equal(s.accepted, sampler.accepted_token_mask)
            assert torch.equal(torch.stack(list(s.sc.values()))[None], sc)
            assert torch.equal(s.rng.get_state(), torch.get_rng_state())


def test_unseen_row_changes_native_acceptance_and_stop():
    # Two fully specified two-position native examples, not imputations passed
    # to the selected sampler. Position 0 is identical across both examples.
    ids = torch.tensor([[0, 0]])
    sampler = native.EntropyBoundSampler(native.EntropyBoundSamplerConfig(0.1), 2, 2, 48)
    for off_row, accepted in (([30., 0.], True), ([2., 0.], False)):
        scores = torch.tensor([[[0., 0.], off_row]])
        sampler.accept_canvas(ids, ids, scores, 48)
        assert sampler.accepted_token_mask[0, 0].item() == accepted
    for off_row, stopped in (([30., 0.], True), ([2., 0.], False), ([0., 30.], False)):
        stop = native.StableAndConfidentStoppingCriteria(1, 0.005)
        previous = torch.tensor([[[30., 0.], [30., 0.]]])
        assert not stop(previous.argmax(-1), previous).item()
        current = torch.tensor([[[30., 0.], off_row]])
        assert stop(current.argmax(-1), current).item() == stopped


def test_sparse_t2t_and_revocation_leave_off_q_untouched():
    s = SelectedRowSampler(cap=2, vocab_size=8, seed=91)
    initial = s.ids.clone()
    assert s.step([0, 1], lambda r, ids, sc: sharp(ids, 2)).admitted == (0, 1)
    changes = s.step([0, 1], lambda r, ids, sc: sharp(ids, 5))
    assert changes.clock == 47 and changes.edited == (0, 1)
    changes = s.step([0, 1], lambda r, ids, sc: torch.zeros((1, 2, 8)))
    assert len(changes.revoked) == 1
    assert s.accepted.sum().item() == 1 and torch.equal(s.ids[:, 2:], initial[:, 2:])
    assert set(s.sc) == {0, 1}
    for rows in ([0, 2], [0, 0], [0, 1, 2], [-1], [256]):
        with pytest.raises(ValueError):
            s.step(rows, lambda *_: pytest.fail("invalid selection reached decoder"))
    with pytest.raises(RuntimeError, match="every position"):
        s.begin_close()


def test_existing_sampler_cannot_start_sparse_and_selected_rng_follows_q():
    from experiments.t2t.serving_slots.gemma_unified import _sample

    s = SelectedRowSampler(cap=32, vocab_size=8, seed=19)
    # This rejection is before the old full-width sampler or model is invoked.
    with pytest.raises(RuntimeError, match="full initialization"):
        _sample(None, SimpleNamespace(ids=s.ids, raw=None, sc=None), [0], sharp(s.ids[:, :1], 1),
                None, None, None, 48, torch.device("cpu"))
    replica = SelectedRowSampler(cap=32, vocab_size=8, seed=19)
    wide = SelectedRowSampler(cap=256, vocab_size=8, seed=19)
    assert torch.equal(s.ids, wide.ids)
    for state, rows in ((s, range(32)), (replica, range(32)), (wide, range(256))):
        state.step(rows, lambda r, ids, sc: sharp(ids, 1))
    assert torch.equal(s.rng.get_state(), replica.rng.get_state())
    assert torch.equal(s.ids, replica.ids)
    assert not torch.equal(s.rng.get_state(), wide.rng.get_state())


@torch.inference_mode()
def test_real_tiny_decoder_starts_and_finishes_b256_under_q32():
    from transformers import DiffusionGemmaConfig, DiffusionGemmaForBlockDiffusion

    cfg = DiffusionGemmaConfig(
        canvas_length=256,
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                           num_attention_heads=2, num_key_value_heads=2, head_dim=8),
        text_config=dict(vocab_size=128, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_experts=4, top_k_experts=2, moe_intermediate_size=8,
                         num_attention_heads=2, num_key_value_heads=1, num_global_key_value_heads=1,
                         head_dim=8, global_head_dim=8, sliding_window=12,
                         layer_types=["sliding_attention", "full_attention"]))
    cfg._attn_implementation = "sdpa"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        model = DiffusionGemmaForBlockDiffusion(cfg).eval()
    prompt = torch.tensor([[2, 8, 9]])
    gen, _ = model._prepare_generation_config(None, max_new_tokens=256)
    cache = model._prepare_cache_for_generation(gen, 1, 3)
    pos = torch.arange(3, dtype=torch.int32)[None]
    enc_ids, masks = model._prepare_encoder_inputs(prompt, torch.ones_like(prompt), pos, cache, True, 256, 1)
    cache = model.model.encoder(input_ids=enc_ids, attention_mask=masks, past_key_values=cache,
                                position_ids=pos).past_key_values
    bank = WaveKV(256)
    s = SelectedRowSampler(cap=32, vocab_size=128, seed=17, embedding_dtype=model.dtype)
    widths = {}
    drafts = {}

    def project(rows, ids, sc):
        absolute = [3+r for r in rows]
        bank.begin(absolute, 3, "cpu")
        logits = model(decoder_input_ids=ids, decoder_position_ids=torch.tensor([absolute], dtype=torch.int32),
                       self_conditioning_logits=sc, decoder_attention_mask=dict.fromkeys(cfg.text_config.layer_types),
                       past_key_values=cache).logits
        assert len(bank.visited) == 2
        if s.phase == "closing":
            drafts.update(zip(rows, logits.argmax(-1)[0].tolist(), strict=True))
        return logits

    def record(name):
        def hook(module, args):
            widths.setdefault(name, []).append(args[0].shape[1])
        return hook

    chunks = [range(i, i+32) for i in range(0, 256, 32)]
    with wave_attention(model, bank), ExitStack() as stack:
        mods = [layer.self_attn.q_proj for layer in model.model.decoder.layers] + [model.lm_head]
        for j, mod in enumerate(mods):
            stack.callback(mod.register_forward_pre_hook(record(j)).remove)
        for i, chunk in enumerate(chunks):
            s.step(chunk, project)
            assert bank.live == list(range(3, 3+32*(i+1)))
            assert len(s.sc) == len(bank.live)
        s.step(chunks[0], project)
        frozen_ids, rng = s.ids.clone(), s.rng.get_state()
        s.begin_close()
        with pytest.raises(RuntimeError, match="Incomplete"):
            s.commit()
        for chunk in reversed(chunks):
            s.close_chunk(chunk, project)
        with pytest.raises(ValueError, match="Repeated"):
            s.close_chunk(chunks[0], project)
        assert s.commit() == [drafts[i] for i in range(256)]
        assert torch.equal(s.ids, frozen_ids) and torch.equal(s.rng.get_state(), rng)
    assert (s.age, s.sample_q, s.close_calls, s.close_q) == (9, 288, 8, 256)
    assert all(widths[j] == [32]*17 for j in range(3))

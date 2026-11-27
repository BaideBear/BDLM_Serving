"""Strict single-canvas CPU row accounting; no weights, GPU or encoder."""
from types import SimpleNamespace as NS

import pytest
import torch

from experiments.t2t.serving_slots.gemma_strict_rows import OFFERS, StrictRows
from experiments.t2t.serving_slots.gemma_wave_kv import WaveKV


def publish(s, p, accepted=(), changed=()):
    s.finish(p, accepted_after=[r in accepted for r in p.sampled],
             input_changed=[r in changed for r in p.sampled])


@pytest.mark.parametrize("q", OFFERS)
def test_empty_bank_to_complete_coverage_at_actual_q(q):
    s, bank = StrictRows(256 // q), WaveKV(256)
    origin = 41
    for i in range(0, 256, q):
        p = s.plan(q, prefix_quota=q)
        assert p.rows == p.cold == p.sampled == tuple(range(i, i + q))
        assert not p.prefix and not p.drain_only
        pos = [origin + r for r in p.rows]
        bank.begin(pos, origin, "cpu")
        x = torch.tensor(pos, dtype=torch.float32).view(1, 1, -1, 1)
        k, _, mask = bank.extend(0, x, x, x)
        assert k.flatten().tolist() == bank.live == list(range(origin, pos[-1] + 1))
        assert mask.shape[-2:] == (q, i + q) and mask.all()
        publish(s, p)
    assert s.status == "limit" and s.tick == s.age == 256 // q
    assert s.initialized == set(range(256)) and s.plan(q) is None


def test_dirty_first_even_on_offer_drop_then_cold_only_sample_gather():
    s = StrictRows(48)
    p = s.plan(128)
    publish(s, p, accepted=range(90))
    for i in (0, 32):
        p = s.plan(32, prefix_quota=32)
        assert p.drain_only and p.rows == tuple(range(i, i + 32))
        s.finish(p)
        assert s.dirty == set(range(i + 32, 90))
        assert s.age == 1 and s.accepted == set(range(90))
    p = s.plan(32, prefix_quota=32)
    assert p.rows == tuple(range(64, 90)) + tuple(range(128, 134))
    assert p.cold == p.sampled == tuple(range(128, 134)) and not p.prefix
    assert tuple(p.rows[i] for i in p.sample_indices) == p.sampled
    publish(s, p)
    assert not s.dirty and s.age == 2 and s.tick == 4


@pytest.mark.parametrize("mode", ["right", "oldest", "random"])
def test_warm_dirty_mask_prefix_and_selected_t2t(mode):
    s = StrictRows(48, seed=17, ordinal=2)
    accepted = set(range(250)) - {246}  # Hole cuts the physical prefix at 246.
    publish(s, s.plan(256), accepted=accepted)
    while len(s.dirty) >= 32:
        p = s.plan(32)
        assert p.drain_only
        s.finish(p)
    p = s.plan(64, prefix_quota=4, prefix_mode=mode)
    unresolved = set(range(256)) - s.accepted
    assert set(p.sampled) == set(p.rows) == s.dirty | unresolved | set(p.prefix)
    assert len(p.prefix) == 4 and set(p.prefix) <= set(range(246)) - s.dirty
    if mode == "right":
        assert p.prefix == (220, 221, 222, 223)
    elif mode == "oldest":
        assert p.prefix == (28, 29, 30, 31)
    # Revoke one accepted row, edit another, admit a MASK without changing ID,
    # and change the noisy ID of a still-unaccepted row. All four owe refresh.
    revoke, edit = p.prefix[:2]
    after = (accepted - {revoke}) | {246}
    publish(s, p, after, changed={edit, 255})
    assert s.accepted == after and s.dirty == {revoke, edit, 246, 255}
    assert s.age == 2


def test_terminal_drain_and_incomplete_draft_are_not_sampling_steps():
    s = StrictRows(1)
    publish(s, s.plan(64), changed=range(63))
    for n in (32, 31):
        p = s.plan(32)
        assert len(p.rows) == n and p.drain_only
        s.finish(p)
    assert s.age == 1 and s.tick == 3 and len(s.initialized) == 64
    assert s.status == "incomplete" and s.plan(256) is None

    s = StrictRows(48)
    publish(s, s.plan(256), accepted=range(256))
    for _ in range(8):
        p = s.plan(32, prefix_quota=32)
        assert p.drain_only and not p.prefix
        s.finish(p)
    assert s.status == "resolved" and s.age == 1 and s.plan(32) is None


def test_reject_invalid_or_stale_plan_and_nonlocal_updates_atomically():
    s = StrictRows(2)
    for q, k in [(512, 0), (31, 1), (32, 33), (True, 0), (32, -1)]:
        with pytest.raises(ValueError):
            s.plan(q, prefix_quota=k)
    p = s.plan(32)
    with pytest.raises(RuntimeError):
        s.plan(32)
    for accepted, changed, stop in [([False]*256, [False]*32, False),
                                    ([False]*32, [False]*32, True),
                                    ([False]*32, [0]*32, False)]:
        with pytest.raises(ValueError):
            s.finish(p, accepted_after=accepted, input_changed=changed, native_stop=stop)
    assert s.age == s.tick == 0 and not s.initialized
    publish(s, p)
    with pytest.raises(RuntimeError):
        s.finish(p)


def test_pinned_native_denoising_step_only_sees_sampled_logits():
    native = pytest.importorskip("transformers.models.diffusion_gemma.generation_diffusion_gemma")
    from transformers.generation.logits_process import LogitsProcessorList

    b, v, steps = 256, 7, 48
    s = StrictRows(steps)
    idx = list(range(b))
    # Force cold rows 0,1 cleanly accepted, others remain noisy. Next call
    # mixes old dirty with cold; only its measured cold logits enter denoising.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(117)
        ids = torch.randint(v, (1, b))
        arg = ids.clone()
        cfg = native.EntropyBoundSamplerConfig(0.1)
        for call in range(3):
            p = s.plan(32)
            old = ids.clone()
            before_rng = torch.get_rng_state()
            if p.drain_only:
                assert call == 1
                s.finish(p)
                assert torch.equal(before_rng, torch.get_rng_state()) and s.age == 1
                continue
            measured = torch.zeros(1, len(p.rows), v)
            measured[:, :2, 1] = 100
            selected = list(p.sampled)
            logits = measured[:, p.sample_indices]
            sampler = native.EntropyBoundSampler(cfg, len(selected), v, steps)
            ids_new, draft, _, _ = native.DiffusionGemmaGenerationMixin._denoising_step(
                NS(config=NS(text_config=NS(vocab_size=v)),
                   model=NS(decoder=NS(embed_tokens=NS(weight=logits)))),
                decoder_forward=lambda **_: NS(logits=logits),
                current_canvas=ids[:, selected], argmax_canvas=arg[:, selected],
                input_ids=torch.tensor([[4]]), decoder_position_ids=None,
                self_conditioning_logits=None, mask_mapping={}, past_key_values=None,
                finished_denoising=torch.tensor([False]), cur_step=steps-s.age,
                sampler=sampler, logits_processor=LogitsProcessorList(),
                diffusion_stopping_criteria=None)
            ids[:, selected], arg[:, selected] = ids_new, draft
            others = [r for r in idx if r not in selected]
            assert torch.equal(ids[:, others], old[:, others])
            assert sampler.accepted_token_mask.shape == (1, len(selected))
            # First SC publication is a changed decoder input for every cold row.
            s.finish(p, accepted_after=sampler.accepted_token_mask[0].tolist(),
                     input_changed=[True] * len(selected))
    assert s.age == 2 and s.tick == 3 and len(s.initialized) == 64

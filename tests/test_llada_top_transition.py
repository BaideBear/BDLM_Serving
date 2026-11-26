"""CPU state and projection contracts; this harness does not score generation."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bdlm_serving.methods.t2t.llada_dual import DualBlockKernel
from experiments.t2t.serving_slots.llada_top_transition import TopTransitionKernel


class CPUModel(nn.Module):
    """Record every physical invocation and expose the prior block through KV."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
        for layer in self.model.layers:
            layer.attention = nn.Module()
            layer.attention.query_key_value = nn.Identity()
        self.embedding = nn.Embedding(6, 1)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(6).view(-1, 1))
        self.calls = []
        self.fill = {1, 2, 3}
        self.propose = lambda block, token, prior: token or 1
        self.propose_at = {}
        self.draws = []

    @property
    def device(self):
        return self.embedding.weight.device

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, inputs_embeds, position_ids, attention_mask,
                past_key_values, use_cache, output_attentions):
        positions = position_ids[0]
        q, end = positions.numel(), attention_mask.shape[-1]
        assert use_cache and output_attentions
        expected = positions[:, None] // 32 >= torch.arange(end)[None, :] // 32
        torch.testing.assert_close(attention_mask, expected[None, None])
        self.calls.append((positions.tolist(), end))
        hidden = inputs_embeds
        for layer_id, layer in enumerate(self.model.layers):
            kv = layer.attention.query_key_value(hidden).unsqueeze(1)
            key, value = past_key_values.update(kv, kv + 100, layer_id)
            assert key.shape[-2] == value.shape[-2] == end
            torch.testing.assert_close(key.index_select(-2, positions), kv)
            if layer_id == 0:
                keys = key[0, 0, :, 0]
                prior = [int(keys[p // 32 * 32 - 1]) if p >= 32 else -1
                         for p in positions.tolist()]
                hidden = hidden + torch.tensor(prior)[None, :, None]
        logits = torch.zeros(1, q, 6)
        tokens = inputs_embeds[0, :, 0].int().tolist()
        for i, (p, token, context) in enumerate(zip(positions.tolist(), tokens, prior)):
            block = p // 32
            draft = self.propose_at.get(p, self.propose(block, token, context))
            logits[0, i, draft] = 12.0 if token or block in self.fill else 0.05
        return SimpleNamespace(logits=logits)

    def _sample_with_temperature_topk_topp(self, logits, *, temperature, top_k, top_p):
        assert temperature == 0.0 and top_k is top_p is None
        self.draws.append(torch.rand(logits.shape[1]))
        confidence, draft = logits.softmax(-1).max(-1)
        return draft, confidence


def create(kernel, prompt=32, length=64, max_post=16, eos=(), seed=11):
    config = dict(confidence_threshold=.8, editing_threshold=.8, max_post_steps=max_post)
    return kernel.create(torch.full((1, prompt), 5), config, length, seed, 0, eos)


def assert_work(model, state, offers):
    assert len(state.events) == len(model.calls)
    generation = [e for e in state.events if e["phase"] != "prefill"]
    assert len(generation) == state.offer_tick == len(offers)
    assert [e["offer_tick"] for e in generation] == list(range(len(offers)))
    assert [e["q_budget"] for e in generation] == offers
    assert [e["previous_offer"] for e in generation] == [None] + offers[:-1]
    for e, (positions, visible_end) in zip(state.events, model.calls, strict=True):
        assert e["F"] == 1
        assert e["absolute_positions"] == positions
        assert e["physical_q"] == len(positions) == e["actual_q"]
        assert e["layer_physical_q"] == [len(positions)] * 2
        assert e["visible_kv"] == visible_end
        if e["phase"] != "prefill":
            assert 0 < len(positions) <= 64
            assert e["overflow"] == max(0, len(positions) - e["q_budget"])
            assert e["overflow"] <= len(e["refresh_only_positions"])
            assert set(e["refresh_only_positions"]) <= set(e["refreshed_pending"]) <= set(positions)
            assert e["sampled_q"] == (0 if e["phase"] == "commit" else
                                      len(positions) - len(e["refresh_only_positions"]))


def test_consecutive_low_offers_refresh_m2t_and_sparse_strict_edits_once():
    model = CPUModel().eval()
    model.fill = {2}
    kernel = TopTransitionKernel(model)
    state = create(kernel)
    hi = kernel.step(state, 64)
    assert hi["absolute_positions"] == list(range(32, 96))
    assert hi["layer_physical_q"] == [64, 64]
    strict = [64, 75, 95]
    for dirty in (list(range(64, 96)), strict):
        resident = state.resident
        history = state.right_history
        post = state.right_post_steps
        clocks = (state.last_query[32:], state.last_write[32:])
        tail_tokens = state.x[:, 64:].clone()
        clean = [p for p in range(64, 96) if p not in dirty]
        dirty0 = [p - 64 for p in dirty]
        saved_tail = [(k[..., 64:, :].clone(), v[..., 64:, :].clone()) for k, v in resident]
        for projected in (dirty, []):
            lo = kernel.step(state, 32)
            assert lo["absolute_positions"] == list(range(32, 64)) + projected
            assert lo["blocks"] == [1] and lo["visible_kv"] == (96 if projected else 64)
            assert lo["refresh_only_positions"] == projected
            assert lo["layer_physical_q"] == [32 + len(projected)] * 2
            assert lo["overflow"] == len(projected)
            assert [p for p in lo["refreshed_pending"] if p >= 64] == projected
            assert model.draws[-1].numel() == lo["sampled_q"] == 32
            assert state.resident is resident and state.right_history is history
            assert state.right_post_steps == post and not state.right_done
            assert state.last_write[32:] == clocks[1]
            assert all(state.last_query[p - 32] == clocks[0][p - 64] for p in clean)
            if projected:
                assert all(state.last_query[p - 32] == lo["hard_clock"] for p in dirty)
            else:
                assert all(state.last_query[p - 32] == lo["hard_clock"] - 1 for p in dirty)
            torch.testing.assert_close(state.x[:, 64:], tail_tokens)
            for (k, v), (old_k, old_v) in zip(resident, saved_tail, strict=True):
                torch.testing.assert_close(k[..., clean, :], old_k[..., [p - 64 for p in clean], :])
                torch.testing.assert_close(v[..., clean, :], old_v[..., [p - 64 for p in clean], :])
            torch.testing.assert_close(resident[0][0][0, 0, dirty, 0],
                                       tail_tokens[0, dirty0].float())
        if dirty != strict:
            model.propose_at = dict.fromkeys(strict, 2)
        hi = kernel.step(state, 64)
        assert hi["top_action"] == "pair_denoise"
        assert hi["absolute_positions"] == list(range(32, 96))
        assert not any(p >= 64 for p in hi["refreshed_pending"])
        assert hi["refresh_only_positions"] == []
        assert hi["hard_strict_edits"] == (len(strict) if dirty != strict else 0)
        assert model.draws[-1].numel() == 64
        assert state.resident is resident
    assert state.committed_end == state.cache.get_seq_length() == 32
    assert_work(model, state, [64, 32, 32, 64, 32, 32, 64])


def test_low_commit_promotes_speculative_done_tail_only_after_new_prefix_reencode():
    model = CPUModel().eval()
    model.propose = lambda b, t, prior: (
        (3 if prior == 2 else 4) if b == 2 else (2 if t else 1))
    kernel = TopTransitionKernel(model)
    state = create(kernel, max_post=1, eos=(4,))
    kernel.step(state, 64)
    kernel.step(state, 64)
    assert state.phase == "commit" and state.right_done and state.right_post_steps == 1
    assert state.x[0, 64:].tolist() == [4] * 32
    prior_history = state.right_history
    prior_tail_clocks = (state.last_query[32:], state.last_write[32:])
    with pytest.raises(RuntimeError, match="not final"):
        kernel.output(state)
    before_commit_rng = state.rng_cpu.clone()
    event = kernel.step(state, 32)
    assert event["top_action"] == "commit_head"
    assert state.block == 2 and state.phase == "reencode" and state.context_version == 1
    assert state.resident is None and state.left_history is prior_history
    assert state.left_post_steps == 0 and not state.left_done
    assert (state.last_query[:32], state.last_write[:32]) == prior_tail_clocks
    assert state.x[0, 64:].tolist() == [4] * 32
    torch.testing.assert_close(state.rng_cpu, before_commit_rng)
    prefix = state.cache.to_legacy_cache()
    assert state.cache.get_seq_length() == 64
    assert prefix[0][0][0, 0, 32:, 0].tolist() == [2.] * 32
    event = kernel.step(state, 64)
    assert event["top_action"] == "promote_verify" and event["context_version"] == 1
    assert event["absolute_positions"] == list(range(64, 96))
    assert event["hard_strict_edits"] == 32 and state.phase == "commit"
    assert state.cache.get_seq_length() == 64
    assert state.x[0, 64:].tolist() == [3] * 32
    kernel.step(state, 32)
    assert state.phase == "done" and state.cache.get_seq_length() == 96
    assert kernel.output(state).tolist() == [[2] * 32 + [3] * 32]
    assert_work(model, state, [64, 64, 32, 64, 32])


@pytest.mark.parametrize("terminal", [False, True])
def test_low_commit_refreshes_dirty_tail_but_stores_only_head(terminal):
    model = CPUModel().eval()
    kernel = TopTransitionKernel(model)
    state = create(kernel, max_post=1, eos=(4,) if terminal else ())
    kernel.step(state, 64)
    dirty = [66, 90]
    model.propose_at = dict.fromkeys(dirty, 3)
    if terminal:
        model.propose_at[32] = 4
    kernel.step(state, 64)
    assert state.phase == "commit" and state.right_done
    resident = state.resident
    tail_before = state.x[:, 64:].clone()
    old_tail_history = state.right_history
    old_keys = resident[1][0][..., 64:, :].clone()
    draws = len(model.draws)
    event = kernel.step(state, 32)
    assert event["top_action"] == "commit_head"
    assert event["absolute_positions"] == list(range(32, 64)) + dirty
    assert event["refresh_only_positions"] == dirty and event["overflow"] == 2
    assert event["layer_physical_q"] == [34, 34] and event["sampled_q"] == 0
    assert len(model.draws) == draws
    torch.testing.assert_close(state.x[:, 64:], tail_before)
    assert resident[0][0][0, 0, dirty, 0].tolist() == [3, 3]
    clean = [i for i in range(32) if i + 64 not in dirty]
    torch.testing.assert_close(resident[1][0][..., [64 + i for i in clean], :],
                               old_keys[..., clean, :])
    assert state.resident is None and state.cache.get_seq_length() == 64
    assert state.cache.to_legacy_cache()[0][0][0, 0, 32, 0].item() == (4 if terminal else 1)
    offers = [64, 64, 32]
    if terminal:
        assert state.phase == "done"
        assert kernel.output(state).tolist() == [[4]]
    else:
        assert state.phase == "reencode" and state.left_history is old_tail_history
        event = kernel.step(state, 32)
        assert event["top_action"] == "promote_verify" and event["physical_q"] == 32
        assert event["refresh_only_positions"] == event["refreshed_pending"] == []
        assert state.cache.get_seq_length() == 64
        offers += [32]
    assert_work(model, state, offers)


def test_high_offer_refreshes_done_tail_without_restarting_sampling():
    model = CPUModel().eval()
    model.fill = {2}
    kernel = TopTransitionKernel(model)
    state = create(kernel, max_post=1)
    kernel.step(state, 64)
    model.propose_at = {72: 2}
    kernel.step(state, 64)
    assert state.phase == "denoise" and state.right_done
    tail_history = state.right_history
    event = kernel.step(state, 64)
    assert event["refresh_only_positions"] == [72] and event["overflow"] == 0
    assert event["layer_physical_q"] == [33, 33] and event["sampled_q"] == 32
    assert state.right_done and state.right_history is tail_history
    assert_work(model, state, [64, 64, 64])


def test_fused_growth_projects_head_and_tail_instead_of_separate_bootstrap():
    def arm(fuse_growth):
        model = CPUModel().eval()
        kernel = TopTransitionKernel(model, fuse_growth=fuse_growth)
        state = create(kernel)
        calls = []
        rng = torch.get_rng_state().clone()

        def offer(info):
            assert info.tick == len(calls) == state.offer_tick
            calls.append(info)
            model.fill = set() if not info.tick else {1, 2}
            return 32 if not info.tick else 64

        assert kernel.run(state, offer) is state
        torch.testing.assert_close(torch.get_rng_state(), rng)
        assert kernel.step(state, 64) is None
        assert_work(model, state, [32] + [64] * (len(calls) - 1))
        assert not calls[1].tail_resident and calls[2].tail_resident
        return kernel.output(state), state.events[1:]

    fused_tokens, fused = arm(True)
    split_tokens, split = arm(False)
    torch.testing.assert_close(fused_tokens, split_tokens)
    assert fused[1]["top_action"] == "fused_growth" and fused[1]["physical_q"] == 64
    assert split[1]["top_action"] == "split_growth" and split[1]["physical_q"] == 33
    assert len(split[1]["refresh_only_positions"]) == 1
    assert split[1]["refresh_only_positions"] == split[1]["refreshed_pending"]
    assert split[1]["sampled_q"] == 32
    assert len(fused) + 1 == len(split)
    assert sum(e["physical_q"] for e in fused) + 1 == sum(e["physical_q"] for e in split)


def test_fixed_64_matches_pinned_dual_on_partial_prompt_two_block_fixture():
    models = [CPUModel().eval(), CPUModel().eval()]
    base = DualBlockKernel(models[0], mask_slots=56, policy="hard_frontier")
    top = TopTransitionKernel(models[1])
    a, b = (create(kernel, prompt=44, length=52) for kernel in (base, top))
    base.run(a)
    top.run(b, lambda _: 64)
    torch.testing.assert_close(base.output(a), top.output(b))
    torch.testing.assert_close(a.rng_cpu, b.rng_cpu)
    assert models[0].calls == models[1].calls
    assert models[1].calls[2][0] == list(range(44, 96))  # Frozen prompt not selected.
    assert_work(models[1], b, [64] * b.offer_tick)


def test_invalid_first_offer_does_not_prefill_and_eos_never_commits_tail():
    model = CPUModel().eval()
    model.propose = lambda b, t, prior: (4 if b == 1 else 1) if not t else t
    kernel = TopTransitionKernel(model)
    state = create(kernel, eos=(4,))
    for invalid in (True, 32.0, 16, 128):
        with pytest.raises(ValueError, match="integer offer"):
            kernel.step(state, invalid)
        assert state.events == [] and state.phase == "prefill" and not model.calls
    kernel.run(state, lambda tick: [64, 32, 64][tick.tick])
    assert state.cache.get_seq_length() == 64
    assert kernel.output(state).tolist() == [[4]]
    assert_work(model, state, [64, 32, 64])


def test_tail_edit_back_to_mask_is_refreshed_before_the_next_high_offer():
    model = CPUModel().eval()
    model.fill = {2}
    kernel = TopTransitionKernel(model)
    state = create(kernel)
    kernel.step(state, 64)
    dirty = [64, 79]
    model.propose_at = dict.fromkeys(dirty, 0)
    edited = kernel.step(state, 64)
    assert edited["edited"] == 2 and edited["hard_strict_edits"] == 0
    assert not state.right_history.fresh and not state.right_done
    old_history, old_post = state.right_history, state.right_post_steps
    assert state.resident[0][0][0, 0, dirty, 0].tolist() == [1, 1]
    parked = kernel.step(state, 32)
    assert parked["refresh_only_positions"] == dirty and parked["overflow"] == 2
    assert parked["layer_physical_q"] == [34, 34]
    assert state.x[0, dirty].tolist() == [0, 0]
    assert state.resident[0][0][0, 0, dirty, 0].tolist() == [0, 0]
    assert state.right_history is old_history and state.right_post_steps == old_post
    model.propose_at.clear()
    resumed = kernel.step(state, 64)
    assert resumed["top_action"] == "pair_denoise"
    assert not set(dirty) & set(resumed["refreshed_pending"])
    assert resumed["block_counts"]["2"]["accepted"] == 2
    assert_work(model, state, [64, 64, 32, 64])


def test_rolls_again_after_promotion_with_a_new_tail_and_keeps_surviving_history():
    model = CPUModel().eval()
    model.fill = {1, 3}
    kernel = TopTransitionKernel(model)
    state = create(kernel, length=96)
    kernel.step(state, 64)
    kernel.step(state, 32)
    kernel.step(state, 32)  # Commit head; roll an incomplete parked tail to head.
    event = kernel.step(state, 32)
    assert event["top_action"] == "promote_verify" and state.phase == "denoise"
    assert state.resident[0][0].shape[-2] == 96
    assert state.left_history.resolved.count(True) == 2
    assert state.right_history is None and state.last_query[32:] == [-1] * 32
    event = kernel.step(state, 64)
    assert event["top_action"] == "fused_growth"
    assert event["absolute_positions"] == list(range(64, 128))
    assert state.left_history.resolved.count(True) == 3
    assert state.right_history.resolved.count(True) == 32
    assert state.committed_end == state.cache.get_seq_length() == 64
    assert_work(model, state, [64, 32, 32, 32, 64])

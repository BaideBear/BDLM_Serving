"""CPU coverage of the native adapter contract; no model download required."""
import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bdlm_serving.methods.t2t.llada_sparse import LLaDASparseKernel, _GateEvidence, qualify_full_budget
from bdlm_serving.methods.t2t.planner import History


MASK = 0
CFG = dict(confidence_threshold=.8, editing_threshold=.8, max_post_steps=4)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Module()
        self.attention.query_key_value = nn.Identity()


class _Toy(nn.Module):
    """Exercise gathered embeddings, per-layer projection hooks and KV updates."""

    def __init__(self, *, edit=True):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer(), _Layer()])
        self.embedding = nn.Embedding(4, 1)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(4).view(4, 1))
        self.edit, self.widen = edit, False
        self.embedding_rows = []
        self.embedding.register_forward_pre_hook(
            lambda _m, args: self.embedding_rows.append(args[0].shape[1]))

    @property
    def device(self):
        return self.embedding.weight.device

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, inputs_embeds, position_ids, attention_mask, past_key_values,
                use_cache, output_attentions):
        positions = position_ids[0]
        q, end = positions.numel(), attention_mask.shape[-1]
        assert attention_mask.shape == (1, 1, q, end)
        assert torch.equal(attention_mask[0, 0],
                           positions[:, None] // 32 >= torch.arange(end)[None, :] // 32)
        assert use_cache and output_attentions
        for i, layer in enumerate(self.model.layers):
            hidden = inputs_embeds + i * 10
            if self.widen and q < 32:
                hidden = torch.cat([hidden, hidden], dim=1)
            kv = layer.attention.query_key_value(hidden).unsqueeze(1)
            key, value = past_key_values.update(kv, kv + 100, i)
            assert key.shape[-2] == value.shape[-2] == end
            assert torch.equal(key.index_select(-2, positions), kv)
        tokens = inputs_embeds.squeeze(-1).long()
        draft = torch.where(tokens == MASK, 1, tokens)
        if self.edit:
            draft = torch.where(tokens == 1, 2, draft)
        conf = torch.where((tokens == MASK) & (positions % 32 >= 4), 1.0, 12.0)
        logits = torch.zeros(1, q, 4).scatter_(-1, draft[..., None], conf[..., None])
        return SimpleNamespace(logits=logits)

    @staticmethod
    def _sample_with_temperature_topk_topp(logits, *, temperature, top_k, top_p):
        assert temperature == 0.0 and top_k is top_p is None
        draft = logits.argmax(-1)
        return draft, logits.softmax(-1).gather(-1, draft[..., None]).squeeze(-1)


def _state(kernel, *, cfg=CFG, prompt_length=34, gen_length=1, eos_ids=()):
    return kernel.create(torch.full((1, prompt_length), 3), cfg, gen_length, 29, MASK, eos_ids)


@pytest.mark.parametrize("q", [6, 32])
def test_physical_prefix_same_state_swap_gathers_real_nonprompt_rows(q):
    plans = []
    for mode, mask_slots in (("hard_prefix_zero", q), ("hard_prefix_frontier_right", q - 2)):
        model = _Toy(edit=False).eval()
        # Only retire rows 2 and 3; select one remaining MASK per step.
        kernel = LLaDASparseKernel(model, q, 0, mask_slots, policy=mode)
        state = _state(kernel)
        for _ in range(4):
            kernel.step(state)
        e = state.events[-1]
        p = e["prefix_plan"]
        assert e["stage"] == "selected" and e["actual_q"] == e["physical_q"] == p["actual_width"]
        assert e["absolute_positions"] == [32 + i for i in p["selected_rows"]]
        assert len(p["fresh_rows"]) == 1 and p["fresh_rows"][0] in range(4, 32)
        assert p["eligible_prefix_rows"] == [2, 3]
        assert p["first_mask"] == (5 if p["fresh_rows"] == [4] else 4)
        assert p["first_mask"] in p["selected_rows"]
        plans.append(p)
    zero, right = plans
    assert zero["selected_rows"] == right["zero_rows"] and zero["actual_width"] == right["actual_width"]
    assert right["added_rows"] == [2, 3] and right["evicted_rows"] == zero["zero_mask_rows"][-2:]
    if q == 32:
        assert zero["actual_width"] < q


def test_selected_gather_and_final_commit_are_separately_charged():
    model = _Toy().eval()
    kernel = LLaDASparseKernel(model, 5, 1, 2)
    state = _state(kernel, eos_ids=(2,))
    assert kernel.step(state) == "prefill"
    prefix = state.cache.to_legacy_cache()
    assert kernel.step(state) == "denoise"
    assert state.events[-1]["stage"] == "bootstrap"
    before = [(k.clone(), v.clone()) for k, v in state.resident]
    assert kernel.step(state) == "denoise"
    event = state.events[-1]
    assert event["stage"] == "selected"
    assert event["actual_q"] == event["physical_q"] == event["q_tokens"] == 5
    assert event["F"] == 1
    assert event["layer_physical_q"] == [5, 5]
    assert model.embedding_rows == [32, 32, 5]
    pos = torch.tensor(event["absolute_positions"])
    skipped = torch.tensor(sorted(set(range(64)) - set(pos.tolist())))
    assert pos.min() >= 34 and pos.max() < 64
    assert state.cache.get_seq_length() == 32
    for i, ((k, v), (old_k, old_v)) in enumerate(zip(state.resident, before)):
        torch.testing.assert_close(k.index_select(-2, skipped), old_k.index_select(-2, skipped))
        torch.testing.assert_close(v.index_select(-2, skipped), old_v.index_select(-2, skipped))
        assert (k.index_select(-2, pos) != old_k.index_select(-2, pos)).any()
        assert torch.equal(state.cache.to_legacy_cache()[i][0], prefix[i][0])
        assert torch.equal(state.cache.to_legacy_cache()[i][1], prefix[i][1])

    kernel.run(state)
    assert state.phase == "done" and state.resident is None and state.history is None
    assert [e["stage"] for e in state.events].count("bootstrap") == 1
    assert [e["stage"] for e in state.events][-3:] == ["verify", "verify", "commit"]
    assert all(e["masked_before"] == 0 and e["actual_q"] == 32
               for e in state.events if e["stage"] == "verify")
    assert state.events[-1]["store_kv"] and state.events[-1]["physical_q"] == 32
    assert sum(e["F"] for e in state.events) == len(model.embedding_rows)
    assert sum(e["physical_q"] for e in state.events) == sum(model.embedding_rows)
    for i, (k, v) in enumerate(state.cache.to_legacy_cache()):
        assert torch.equal(k.flatten(), state.x[0, :64] + i * 10)
        assert torch.equal(v.flatten(), state.x[0, :64] + i * 10 + 100)


def test_edit_only_selection_is_not_convergence_and_next_pass_gets_mask():
    kernel = LLaDASparseKernel(_Toy(edit=False).eval(), 1, 1, 0)
    state = _state(kernel)
    for _ in range(3):
        kernel.step(state)
    event = state.events[-1]
    assert event["masked_before"] > 0 and event["selected_masked_before"] == 0
    assert event["accepted"] == event["edited"] == state.post_steps == 0
    assert state.phase == "denoise"
    kernel.step(state)
    assert state.events[-1]["selected_masked_before"] == state.events[-1]["accepted"] == 1


def test_controls_share_fresh_ordering_and_noedit_backfill_still_writes():
    model = _Toy().eval()
    kernels = [LLaDASparseKernel(model, 4, 1, mask_slots, policy=policy)
               for policy, mask_slots in (("targeted", 1), ("random", 1),
                                           ("undirected", 1), ("noedit", 3))]
    # Three stable rows, two fresh rows with reversed positional/confidence
    # ordering, one MASK. Both unused mask seats go to fresh then neutral edit.
    history = History([True, True, True, True, True, False],
                      [.9, .6, .1, .8, .2, None], [1] * 6, {3, 4}, {3, 4})
    for kernel in kernels:
        state = _state(kernel)
        state.history = history
        picked = kernel._pick(state, range(6), 32)
        assert len(picked) == 4 and 4 in picked and 5 in picked
        assert picked == kernel._pick(state, range(6), 32)
        if kernel.policy == "noedit":
            assert picked == [0, 3, 4, 5]
        # With enough masks to occupy the other seats, only the higher-ranked
        # fresh row fits; positional fresh ordering would choose 3 instead of 4.
        state.history = History(history.resolved + [False] * 2,
                                history.certainty + [None] * 2,
                                history.streak + [0] * 2, history.changed, history.fresh)
        assert 4 in kernel._pick(state, range(8), 32)
        assert 3 not in kernel._pick(state, range(8), 32)

    # Leave exactly one MASK, deliberately seed valid scratch KV via bootstrap,
    # and ensure the next sparse noedit pass can edit both types of resolved row.
    noedit = kernels[-1]
    state = _state(noedit)
    noedit.step(state)
    state.x[:, 34:63] = 1
    noedit.step(state)  # bootstrap may edit the preseeded 1s and admit the mask
    state.x[:, 34:63] = 1
    state.x[:, 63] = MASK
    state.history = History([True] * 31 + [False], [None] * 32, [0] * 32, {5}, {5})
    assert noedit.step(state) == "denoise"
    event = state.events[-1]
    assert event["absolute_positions"] == [34, 35, 37, 63]
    assert event["edited"] == 3
    assert state.x[0, [34, 35, 37]].tolist() == [2, 2, 2]


def test_physical_projection_audit_rejects_hidden_padding():
    model = _Toy().eval()
    kernel = LLaDASparseKernel(model, 5, 1, 2)
    state = _state(kernel)
    kernel.step(state)
    kernel.step(state)
    model.widen = True
    with pytest.raises(RuntimeError, match="physical Q shape"):
        kernel.step(state)
    assert len(state.events) == 2
    assert all(not projection._forward_pre_hooks for projection in kernel.projections)


@pytest.mark.parametrize("max_post_steps", [0, 4])
def test_full_budget_qualification_against_native_batch_one(max_post_steps):
    model = _Toy().eval()
    result = qualify_full_budget(model, torch.full((1, 34), 3),
                                 dict(CFG, max_post_steps=max_post_steps),
                                 33, 29, MASK, [])
    assert result["commits"] == 2
    assert result["actual_q"] == result["physical_q"] == result["forwards"] * 32
    assert result["max_q"] == 32 and len(result["output_ids"]) == 33


@pytest.mark.parametrize("policy", ["near_miss", "gate_random"])
def test_gate_allocation_from_last_observed_rows(policy):
    kernel = LLaDASparseKernel(_Toy().eval(), 6, 1, 2, policy=policy)
    state = _state(kernel, prompt_length=32)
    before = [1] * 12 + [MASK] * 20
    confidence = [.8, .7, .9, .1, .75, .2, .7, .1, .1, .7, .3, .1] + [None] * 20
    state.history = History([x != MASK for x in before], confidence, [0]*32, {3, 4}, {3, 4})
    drafts = {i: (2 if i in (0, 2, 3, 4, 5, 7, 8) else 1, 1) for i in range(31)}
    del drafts[8]  # unobserved row must not become a low-confidence disagreement
    evidence = state.gate_evidence = _GateEvidence(drafts, {4})
    tags = evidence.categories(state, before, 32)
    assert [i for i, tag in enumerate(tags) if tag == "near_miss"] == [0, 5, 7]
    sample_pool = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11]
    py_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    for seed in (7, 29, 113):
        state.selection_seed = seed
        picked, reserved, released = kernel._gate_pick(state, range(32), 32, tags)
        expected = ([5, 7, 0] if policy == "near_miss"
                    else random.Random(f"gate_random:{seed}:32:0").sample(sample_pool, 3))
        assert reserved == expected and released == 0
        assert picked == sorted([3, 12, 13] + reserved)
    assert random.getstate() == py_rng and torch.equal(torch_rng, torch.get_rng_state())

    # Distance ties: ascending certainty first, then position. No previous edit:
    # all distances tied. Prompt, fresh, unseen and strictly above threshold excluded.
    state.prompt_length = 33
    evidence.observed[1] = (2, 1)
    evidence.previous_t2t = {3}
    tags = evidence.categories(state, before, 32)
    assert tags[0] == "prompt"
    if policy == "near_miss":
        assert kernel._gate_pick(state, range(1, 32), 32, tags)[1] == [5, 1, 7]
        state.history.certainty[1] = .2
        assert kernel._gate_pick(state, range(1, 32), 32, tags)[1] == [1, 5, 7]
        evidence.previous_t2t.clear()
        assert kernel._gate_pick(state, range(1, 32), 32, tags)[1] == [7, 1, 5]

    # One out of three seats supported. Then none: release seats to masks.
    for i in (5, 7, 1):
        evidence.observed[i] = (1, 1)
        tags = evidence.categories(state, before, 32)
        picked, reserved, released = kernel._gate_pick(state, range(1, 32), 32, tags)
        n = tags.count("near_miss")
        assert len(reserved) == n and released == 3-n
        assert sum(tags[i] == "mask" for i in picked) == 2+released
        assert 3 in picked and 4 not in picked and len(picked) == 6
    # Scarce masks/fresh: ordinary neutral non-fresh backfill, not reservations.
    picked, reserved, released = kernel._gate_pick(state, range(1, 13), 32, tags)
    assert (picked, reserved, released) == ([1, 2, 3, 4, 5, 12], [], 3)


class _GateToy(_Toy):
    def forward(self, **kw):
        result = super().forward(**kw)
        pos, tokens = kw["position_ids"][0], kw["inputs_embeds"][0, :, 0]
        # The real bootstrap observes three weak disagreements and two strong
        # writes. One strong write is a revocation, not a strict T2T anchor.
        weak = ((pos == 34) | (pos == 36) | (pos == 39)) & (tokens == 1)
        if len(self.embedding_rows) == 2:
            result.logits[0, weak] = torch.tensor([0., 0., 1., 0.])
            result.logits[0, pos == 38] = torch.tensor([12., 0., 0., 0.])
            result.logits[0, pos == 40] = torch.tensor([0., 20., 0., 0.])
        result.logits[0, pos == 35] = torch.tensor([0., 12., 0., 0.])
        return result


def test_gate_live_sparse_step_audit_ages_and_strict_writes():
    model = _GateToy().eval()
    kernel = LLaDASparseKernel(model, 3, 1, 1, policy="near_miss")
    state = _state(kernel, gen_length=33)
    kernel.step(state)
    assert "gate" not in state.events[-1]
    state.x[0, 34:40] = 1
    kernel.step(state)
    bootstrap = state.events[-1]
    assert bootstrap["edited"] == 2
    assert bootstrap["gate"]["strict_edit_positions"] == [37]
    assert bootstrap["gate"]["selected_evidence_age"] == [None]*32
    kernel.step(state)
    e = state.events[-1]
    assert e["absolute_positions"] == [36, 37, 38]
    assert e["gate"]["previous_strict_edit_positions"] == [37]
    assert e["gate"]["available_near_miss_count"] == 3
    assert e["gate"]["reserved_positions"] == e["gate"]["reserved_strict_edit_positions"] == [36]
    assert e["gate"]["selected_evidence_age"] == [1, 1, 1]
    assert e["gate"]["selected_category_counts"] == dict(
        prompt=0, mask=1, fresh=1, near_miss=1, other_resolved=0)
    kernel.step(state)
    e = state.events[-1]
    assert e["gate"]["previous_strict_edit_positions"] == [36]
    assert e["gate"]["reserved_positions"] == [34]
    assert e["gate"]["selected_evidence_age"][e["absolute_positions"].index(34)] == 2
    assert [ev["physical_q"] for ev in state.events] == model.embedding_rows == [32, 32, 3, 3]
    assert all(ev["F"] == 1 for ev in state.events)
    kernel.run(state)
    assert state.gate_evidence is None
    for ev in state.events:
        if ev["phase"] == "denoise":
            assert sum(ev["gate"]["selected_category_counts"].values()) == ev["physical_q"]
            assert len(ev["gate"]["selected_evidence_age"]) == ev["physical_q"]
            assert set(ev["gate"]["reserved_positions"]) <= set(ev["absolute_positions"])
            if ev["stage"] == "bootstrap":
                assert ev["gate"]["previous_strict_edit_positions"] == []
                assert ev["gate"]["selected_evidence_age"] == [None]*32
        else:
            assert "gate" not in ev


def test_gate_random_full_run_no_free_sampling():
    model = _Toy().eval()
    kernel = LLaDASparseKernel(model, 5, 1, 2, policy="gate_random")
    state = kernel.run(_state(kernel))
    assert [e["actual_q"] for e in state.events] == model.embedding_rows
    assert any(e.get("gate", {}).get("released_edit_slots_to_mask") == 2 for e in state.events)
    assert all("gate" not in e or len(e["gate"]["selected_evidence_age"]) == e["actual_q"]
               for e in state.events)

"""CPU coverage of the native adapter contract; no model download required."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bdlm_serving.methods.t2t.llada_sparse import LLaDASparseKernel, qualify_full_budget
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

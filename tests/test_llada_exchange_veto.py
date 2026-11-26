"""CPU synthetic coverage of hard_exchange_edit4_veto; no model download or GPU."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.cache_utils import DynamicCache

from bdlm_serving.methods.t2t.exchange import ONLINE_MODES, VETO, choose
from bdlm_serving.methods.t2t.llada_sparse import LLaDASparseKernel, _VETO_POLICY
from bdlm_serving.methods.t2t.planner import History
from bdlm_serving.methods.t2t.protocol import Arm


MASK = 0
VOCAB = 8
CFG = dict(confidence_threshold=.8, editing_threshold=.8, max_post_steps=4)
RATES = {("old",): .2, ("mask",): .5}


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Module()
        self.attention.query_key_value = nn.Identity()


class _VetoToy(nn.Module):
    """Per-position drafts, otherwise the same adapter contract as the native toy."""

    def __init__(self, drafts=None, draft_fn=None):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer(), _Layer()])
        self.embedding = nn.Embedding(VOCAB, 1)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(VOCAB, dtype=torch.float32).view(VOCAB, 1))
        self.drafts = dict(drafts or {})
        self.draft_fn = draft_fn
        self.calls = 0
        self.eval()

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
        assert use_cache and output_attentions
        self.calls += 1
        for i, layer in enumerate(self.model.layers):
            hidden = inputs_embeds + i * 10
            kv = layer.attention.query_key_value(hidden).unsqueeze(1)
            key, value = past_key_values.update(kv, kv + 100, i)
            assert key.shape[-2] == value.shape[-2] == end
        tokens = inputs_embeds.squeeze(-1).long()
        draft = tokens.clone()
        conf = torch.full((1, q), 12.0)
        for j, pos in enumerate(positions.tolist()):
            spec = self.drafts.get(pos)
            if spec is None and self.draft_fn is not None:
                spec = self.draft_fn(pos, int(tokens[0, j]))
            if spec is not None:
                draft[0, j], conf[0, j] = spec
        logits = torch.zeros(1, q, VOCAB).scatter_(-1, draft[..., None], conf[..., None])
        return SimpleNamespace(logits=logits)

    @staticmethod
    def _sample_with_temperature_topk_topp(logits, *, temperature, top_k, top_p):
        assert temperature == 0.0 and top_k is top_p is None
        draft = logits.argmax(-1)
        return draft, logits.softmax(-1).gather(-1, draft[..., None]).squeeze(-1)


def _resident(length=64):
    key = torch.arange(length, dtype=torch.float32).view(1, 1, length, 1)
    return tuple((key + 10 * i, key + 100 + 10 * i) for i in range(2))


def _state(kernel, tokens=None):
    state = kernel.create(torch.full((1, 34), 3), CFG, 1, 29, MASK, [])
    state.phase, state.committed_end = "denoise", 32
    state.last_write = state.last_query = [-1] * 32
    state.previous_strict = set()
    state.resident = _resident()
    tokens = [MASK] * 32 if tokens is None else list(tokens)
    assert len(tokens) == 32
    state.x[0, 32:64] = torch.tensor(tokens)
    state.history = History([t != MASK for t in tokens], [None] * 32, [0] * 32, set(), set())
    return state


def _select(kernel, picked, reserved):
    kernel._hard_pick = lambda state, editable, resolved, before: (list(picked), list(reserved))


def _rows():
    # Local 0,1 are prompt; local 2,3,4 are abs 34,35,36; local 5 is abs 37.
    return [3, 3, 2, 2, 2, MASK] + [MASK] * 26


def _veto(drafts):
    model = _VetoToy(drafts)
    kernel = LLaDASparseKernel(model, 4, 0, 3, policy=_VETO_POLICY, exchange_rates=RATES)
    return model, kernel


def test_veto_suppresses_only_the_reserved_strict_write():
    model, kernel = _veto({36: (3, 12.0), 37: (2, 12.0)})
    state = _state(kernel, _rows())
    _select(kernel, [2, 3, 4, 5], [4])
    assert kernel.step(state) == "denoise"
    event = state.events[-1]
    assert event["stage"] == "selected"
    assert event["absolute_positions"] == [34, 35, 36, 37]
    assert event["reserved_extra_edit_positions"] == [36]
    assert event["attempted_strict_veto_positions"] == [36]
    assert event["edited"] == 0 and event["accepted"] == 1
    assert event["hard_strict_edits"] == 0
    assert event["reserved_strict_edit_positions"] == []
    assert state.x[0, 34:38].tolist() == [2, 2, 2, 2]
    assert state.history.fresh == {5}
    assert state.previous_strict == set()
    assert model.calls == 1
    assert event["actual_q"] == event["physical_q"] == 4
    assert event["layer_physical_q"] == [4, 4]


def test_write_enabled_arm_commits_the_same_write():
    model = _VetoToy({36: (3, 12.0), 37: (2, 12.0)})
    kernel = LLaDASparseKernel(model, 4, 0, 3, policy="hard_exchange_edit4",
                               exchange_rates=RATES)
    state = _state(kernel, _rows())
    _select(kernel, [2, 3, 4, 5], [4])
    kernel.step(state)
    event = state.events[-1]
    assert event["edited"] == 1 and event["accepted"] == 1
    assert event["hard_strict_edits"] == 1
    assert event["reserved_strict_edit_positions"] == [36]
    assert "attempted_strict_veto_positions" not in event
    assert state.x[0, 34:38].tolist() == [2, 2, 3, 2]
    assert state.history.fresh == {4, 5}


def test_reserved_revocation_to_mask_is_not_vetoed():
    model, kernel = _veto({36: (MASK, 12.0)})
    state = _state(kernel, _rows())
    _select(kernel, [2, 3, 4, 5], [4])
    kernel.step(state)
    event = state.events[-1]
    assert event["attempted_strict_veto_positions"] == []
    assert event["edited"] == 1
    assert event["hard_strict_edits"] == 0
    assert event["reserved_strict_edit_positions"] == []
    assert state.x[0, 36].item() == MASK


def test_unreserved_strict_write_is_untouched():
    model, kernel = _veto({34: (3, 12.0)})
    state = _state(kernel, _rows())
    _select(kernel, [2, 3, 4, 5], [4])
    kernel.step(state)
    event = state.events[-1]
    assert event["attempted_strict_veto_positions"] == []
    assert event["edited"] == event["hard_strict_edits"] == 1
    assert state.x[0, 34].item() == 3


def test_mask_admission_is_untouched():
    model, kernel = _veto({37: (2, 12.0)})
    state = _state(kernel, _rows())
    _select(kernel, [2, 3, 4, 5], [])
    kernel.step(state)
    event = state.events[-1]
    assert event["attempted_strict_veto_positions"] == []
    assert event["accepted"] == 1 and event["edited"] == 0
    assert state.x[0, 37].item() == 2


def test_veto_does_not_add_model_work():
    def run(policy):
        model = _VetoToy({36: (3, 12.0), 37: (2, 12.0)})
        kernel = LLaDASparseKernel(model, 4, 0, 3, policy=policy, exchange_rates=RATES)
        state = _state(kernel, _rows())
        _select(kernel, [2, 3, 4, 5], [4])
        kernel.step(state)
        return model.calls, state.events[-1]

    calls_a, event_a = run("hard_exchange_edit4")
    calls_b, event_b = run(_VETO_POLICY)
    assert calls_a == calls_b == 1
    for key in ("stage", "actual_q", "physical_q", "layer_physical_q", "F",
                "absolute_positions", "reserved_extra_edit_positions"):
        assert event_a[key] == event_b[key]
    assert event_a["edited"] == 1 and event_b["edited"] == 0


def test_stopping_uses_applied_edits():
    model, kernel = _veto({36: (3, 12.0)})
    state = _state(kernel, _rows())
    state.cache = DynamicCache.from_legacy_cache(_resident(32))
    _select(kernel, [2, 3, 4, 5], [4])
    assert kernel.step(state) == "denoise"
    event = state.events[-1]
    assert event["edited"] == 0 and event["attempted_strict_veto_positions"] == [36]
    assert state.phase == "denoise"  # masks remain; the veto is not an applied edit
    model.drafts = {}
    state.x[0, 34:64] = 2
    state.history = History([True] * 32, [None] * 32, [0] * 32, set(), set())
    assert kernel.step(state) == "denoise"
    event = state.events[-1]
    assert event["stage"] == "verify"
    assert event["attempted_strict_veto_positions"] == []
    assert event["edited"] == 0
    assert state.phase == "commit"  # stopping saw zero applied edits


def test_selection_matches_hard_exchange_edit4_on_identical_state():
    model = _VetoToy()
    a = LLaDASparseKernel(model, 4, 0, 3, policy="hard_exchange_edit4", exchange_rates=RATES)
    b = LLaDASparseKernel(model, 4, 0, 3, policy=_VETO_POLICY, exchange_rates=RATES)
    state = _state(a)
    resolved = [False] * 32
    for i in range(2, 10):
        resolved[i] = True
    state.history = History(resolved, [None] * 32, [0] * 32, {2}, {2})
    state.previous_strict = {4}
    before = state.x[0, 32:64].tolist()
    picked_a, reserved_a = a._hard_pick(state, range(2, 32), resolved, before)
    picked_b, reserved_b = b._hard_pick(state, range(2, 32), resolved, before)
    assert (picked_a, reserved_a) == (picked_b, reserved_b)
    assert reserved_a  # this state really admits an entering row


def test_candidate_entering_by_mask_revocation_is_still_vetoed():
    model, kernel = _veto({})
    state = _state(kernel)
    resolved = [False] * 32
    for i in range(2, 10):
        resolved[i] = True
    state.x[0, 34:42] = 2
    state.history = History(resolved, [None] * 32, [0] * 32, {2}, {2})
    state.previous_strict = {4}
    before = state.x[0, 32:64].tolist()
    picked, reserved = kernel._hard_pick(state, range(2, 32), resolved, before)
    assert reserved == [3]  # entrant 3 displaces a MASK, not an old backfill row
    model.drafts = {35: (3, 12.0)}
    kernel.step(state)
    event = state.events[-1]
    assert event["absolute_positions"] == [32 + i for i in picked]
    assert event["reserved_extra_edit_positions"] == [35]
    assert event["attempted_strict_veto_positions"] == [35]
    assert event["edited"] == 0
    assert state.x[0, 35].item() == 2


def test_veto_arm_is_b32_q4_q8_only():
    assert Arm.parse("4:0:3:hard_exchange_edit4_veto", 32).policy == "t2t_q4_f0_m3_hard_exchange_edit4_veto"
    assert Arm.parse("8:0:7:hard_exchange_edit4_veto", 32).policy == "t2t_q8_f0_m7_hard_exchange_edit4_veto"
    assert Arm.parse("4:0:3:hard_exchange_edit4_veto", 32).edit == 1
    assert ONLINE_MODES[_VETO_POLICY] == ONLINE_MODES["hard_exchange_edit4"] == 4.
    assert VETO == {_VETO_POLICY: 4.}
    with pytest.raises(ValueError):
        Arm.parse("4:0:3:hard_exchange_edit4_veto", 256)
    with pytest.raises(ValueError):
        Arm.parse("4:1:2:hard_exchange_edit4_veto", 32)


def _stable_drafts(pos, token):
    if token == MASK:
        return 1, 12.0 if pos % 32 < 4 else 1.0
    if token in (1, 2) and pos % 2 == 0:
        return token + 1, 12.0
    return token, 12.0


@pytest.mark.parametrize("budget,mask_slots", [(4, 3), (8, 7)])
def test_full_synthetic_run_q4_q8(budget, mask_slots):
    arm = Arm.parse(f"{budget}:0:{mask_slots}:hard_exchange_edit4_veto", 32)
    model = _VetoToy(draft_fn=_stable_drafts)
    kernel = LLaDASparseKernel(model, budget, arm.fresh, arm.mask,
                               policy=arm.selection, exchange_rates=RATES)
    state = kernel.run(kernel.create(torch.full((1, 34), 3), CFG, 8, 29, MASK, []))
    assert state.phase == "done"
    events = state.events
    assert all(e["F"] == 1 and e["actual_q"] == e["physical_q"] for e in events)
    denoise = [e for e in events if e["phase"] == "denoise"]
    assert denoise
    for event in denoise:
        assert isinstance(event["attempted_strict_veto_positions"], list)
        assert all(type(p) is int for p in event["attempted_strict_veto_positions"])
    for event in events:
        if event["phase"] != "denoise":
            assert "attempted_strict_veto_positions" not in event
    work = [e for e in events if e["phase"] != "prefill"]
    assert sum(e["F"] for e in work) == len(work)
    assert sum(e["physical_q"] for e in work) == sum(e["actual_q"] for e in work)

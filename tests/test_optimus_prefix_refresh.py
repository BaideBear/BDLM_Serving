"""CPU contracts for the optional KV-only prefix refresh.

The intervention is a budget control, not a new sampler: it may only consume
physical-Q slack left by the original hard-capped LLaDA2.1 IBS planner, and the
injected rows are projected as real Q rows but never sampled or T2T-edited.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import bdlm_serving.methods.optimus.fixed as fixed_module
from bdlm_serving.methods.optimus.fixed import FixedOptimus
from bdlm_serving.methods.optimus.prefix_refresh import plan_prefix_refresh
from bdlm_serving.methods.optimus.upstream import SelectedResidentCache


class _Attention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)


class TinyKVModel(nn.Module):
    """Small real CPU model: embedding, two cached attention layers, readout.

    ``controlled=True`` zeroes the readout head so logits are exactly the
    token/position proposal; this keeps the original planner trajectory fixed
    while the attention/K/V path is still real.  ``controlled=False`` keeps a
    small nonzero head for a KV-sensitive smoke check.
    """
    def __init__(self, *, vocab=5, d=8, layers=2, controlled=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([_Attention(d) for _ in range(layers)])
        self.head = nn.Linear(d, vocab)
        with torch.no_grad():
            if controlled:
                self.head.weight.zero_()
                self.head.bias.zero_()
            else:
                self.head.weight.mul_(0.01)
                self.head.bias.zero_()
        self.calls = []

    def forward(self, tokens, *, position_ids, attention_mask, past_key_values,
                use_cache, output_attentions):
        positions = position_ids[0]
        q, end = positions.numel(), attention_mask.shape[-1]
        assert use_cache and output_attentions
        assert attention_mask.shape == (1, 1, q, end)
        self.calls.append((positions.tolist(), end, attention_mask[0, 0].clone()))
        hidden = self.embedding(tokens)
        for i, layer in enumerate(self.layers):
            key, value = past_key_values.update(layer.k(hidden), layer.v(hidden), i)
            assert key.shape[-2] == value.shape[-2] == end
            scores = layer.q(hidden) @ key.transpose(-2, -1) / math.sqrt(hidden.shape[-1])
            scores = scores.masked_fill(~attention_mask[0, 0], -1e9)
            hidden = hidden + layer.o(torch.softmax(scores, dim=-1) @ value)
        propose = torch.where(tokens == 0, 2,
                              torch.where(positions % 7 == 0, 3, tokens))
        logits = self.head(hidden).scatter(-1, propose[..., None], 18.0)
        return SimpleNamespace(logits=logits)


class RecordingCache(SelectedResidentCache):
    instances = []

    def __init__(self, capacity):
        super().__init__(capacity)
        self.writes = []
        RecordingCache.instances.append(self)

    def update(self, key, value, layer_idx, cache_kwargs=None):
        self.writes.append((layer_idx, self.selected.tolist()))
        return super().update(key, value, layer_idx, cache_kwargs)


@pytest.fixture(autouse=True)
def recording_cache(monkeypatch):
    RecordingCache.instances.clear()
    monkeypatch.setattr(fixed_module, "SelectedResidentCache", RecordingCache)
    yield RecordingCache


def _steps(events):
    return [e for e in events if e["phase"] != "prefill"]


def _check_event(e, prompt_len):
    original = [e["block"] * 32 + i for i in e["selected_local"]]
    assert e["original_q_tokens"] == len(original) == len(e["selected_masked"])
    assert e["q_tokens"] == e["original_q_tokens"] + e["prefix_refresh_q_tokens"]
    assert 0 < e["q_tokens"] <= e["physical_cap"]
    assert e["prefix_refresh_added"] == len(e["prefix_refresh_absolute"]) == e["prefix_refresh_q_tokens"]
    assert e["prefix_refresh_added"] <= e["prefix_refresh_quota"]
    assert e["prefix_refresh_added"] <= e["prefix_refresh_slack"]
    assert e["prefix_refresh_slack"] == e["physical_cap"] - e["original_q_tokens"]
    assert e["prefix_refresh_projection_order"] == e["absolute_positions"]
    assert e["absolute_positions"] == sorted(e["absolute_positions"])
    assert e["prefix_refresh_original_absolute"] == original
    assert set(original) <= set(e["absolute_positions"])
    assert len(set(e["absolute_positions"])) == e["q_tokens"]
    assert e["prefix_refresh_absolute"] == [e["block"] * 32 + i for i in e["prefix_refresh_local"]]
    assert e["prefix_refresh_absolute"] == sorted(e["prefix_refresh_absolute"])
    assert e["prefix_refresh_eligible"] == sorted(e["prefix_refresh_eligible"])
    assert set(e["prefix_refresh_absolute"]) <= set(e["prefix_refresh_eligible"])

    lower = max(prompt_len, e["block"] * 32)
    upper = e["block"] * 32 + e["cached_prefix_before"]
    largest_original = max(original)
    for pos in e["prefix_refresh_eligible"]:
        assert lower <= pos < upper
        assert pos < largest_original
    for pos in e["prefix_refresh_absolute"]:
        assert lower <= pos < upper
        assert pos < largest_original
        assert pos not in e["dirty_before"]
        assert pos not in e["edited_positions"]
        assert pos not in e["written_absolute"]
    assert e["prefix_refresh_visible_end"] == e["visible_kv"] == largest_original + 1
    assert e["prefix_refresh_visible_span"] == [0, e["visible_kv"]]
    assert e["accepted"] + e["edited"] <= e["original_q_tokens"]
    assert set(e["edited_positions"]) <= set(original)
    assert set(e["prefix_refresh_absolute"]).isdisjoint(e["edited_positions"])
    assert set(e["prefix_refresh_absolute"]).isdisjoint(e["written_absolute"])


def test_plan_bounds_orders_and_slack():
    old = np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=np.uint8)
    dirty = {33}
    initialized = set(range(32, 40))
    original = [38, 39]
    extra, eligible, slack = plan_prefix_refresh(
        prompt_len=20, block_start=32, cached_prefix_before=6,
        old_mask=old, dirty=dirty, initialized=initialized,
        original_positions=original, offer=4, quota=5, order="right")
    assert slack == 2
    assert eligible.tolist() == [32, 34, 35, 36, 37]
    assert extra.tolist() == [36, 37]  # slack limits the quota
    extra, _, _ = plan_prefix_refresh(
        prompt_len=20, block_start=32, cached_prefix_before=6,
        old_mask=old, dirty=dirty, initialized=initialized,
        original_positions=original, offer=6, quota=3, order="right")
    assert extra.tolist() == [35, 36, 37]
    extra, _, _ = plan_prefix_refresh(
        prompt_len=20, block_start=32, cached_prefix_before=6,
        old_mask=old, dirty=dirty, initialized=initialized,
        original_positions=original, offer=6, quota=3, order="oldest")
    assert extra.tolist() == [32, 34, 35]


def test_plan_excludes_prompt_dirty_uninitialized_masked_and_original():
    old = np.array([1, 0, 1, 1, 1, 1, 1, 1], dtype=np.uint8)
    extra, eligible, _ = plan_prefix_refresh(
        prompt_len=32, block_start=32, cached_prefix_before=6,
        old_mask=old, dirty={33}, initialized=set(range(32, 40)) - {35},
        original_positions=[36, 37], offer=8, quota=8, order="right")
    assert eligible.tolist() == [32, 34]
    assert extra.tolist() == [32, 34]
    # Prompt rows and the strict upper edge are never eligible.
    extra, eligible, _ = plan_prefix_refresh(
        prompt_len=35, block_start=32, cached_prefix_before=6,
        old_mask=np.ones(8, dtype=np.uint8), dirty=set(),
        initialized=set(range(32, 40)), original_positions=[38],
        offer=8, quota=8, order="right")
    assert eligible.tolist() == [35, 36, 37]
    assert extra.tolist() == [35, 36, 37]
    # Everything at or after the largest original picked row is excluded.
    extra, eligible, _ = plan_prefix_refresh(
        prompt_len=32, block_start=32, cached_prefix_before=8,
        old_mask=np.ones(8, dtype=np.uint8), dirty=set(),
        initialized=set(range(32, 40)), original_positions=[36],
        offer=8, quota=8, order="right")
    assert eligible.tolist() == [32, 33, 34, 35]
    assert extra.tolist() == [32, 33, 34, 35]
    # No physical slack, no quota, no original rows, or no cap means no injection.
    for kwargs in (dict(offer=2, quota=5), dict(offer=8, quota=0),
                   dict(offer=None, quota=5), dict(offer=8, quota=5, original_positions=[])):
        args = dict(prompt_len=32, block_start=32, cached_prefix_before=6,
                    old_mask=np.ones(8, dtype=np.uint8), dirty=set(),
                    initialized=set(range(32, 40)), original_positions=[38, 39],
                    order="right")
        args.update(kwargs)
        extra, _, slack = plan_prefix_refresh(**args)
        assert extra.size == 0
    with pytest.raises(ValueError):
        plan_prefix_refresh(prompt_len=0, block_start=0, cached_prefix_before=0,
                            old_mask=np.ones(1, dtype=np.uint8), dirty=set(),
                            initialized={0}, original_positions=[0],
                            offer=1, quota=1, order="middle")


def test_refresh_requires_hard_capped_llada_ibs():
    model = TinyKVModel(controlled=True)
    with pytest.raises(ValueError, match="prefix_refresh"):
        FixedOptimus(model, "llada21", 32, "ibs", prefix_refresh=1)
    with pytest.raises(ValueError, match="prefix_refresh"):
        FixedOptimus(model, "llada21", 32, "obs", physical_caps=(32,), prefix_refresh=1)
    with pytest.raises(ValueError, match="prefix_order"):
        FixedOptimus(model, "llada21", 32, "ibs", physical_caps=(32,),
                     prefix_refresh=1, prefix_order="middle")
    with pytest.raises(ValueError, match="prefix_refresh"):
        FixedOptimus(model, "llada21", 32, "ibs", physical_caps=(32,), prefix_refresh=-1)
    FixedOptimus(model, "llada21", 32, "ibs", physical_caps=(32,), prefix_refresh=0)


def test_zero_quota_is_identical_to_baseline():
    model = TinyKVModel(controlled=True).eval()
    prompt = torch.ones(1, 31, dtype=torch.long)
    baseline = FixedOptimus(model, "llada21", 8, "ibs",
                            physical_caps=(8,)).run(prompt, 96, 17, 0, [4])
    zero = FixedOptimus(model, "llada21", 8, "ibs", physical_caps=(8,),
                        prefix_refresh=0).run(prompt, 96, 17, 0, [4])
    assert torch.equal(baseline[0], zero[0])
    assert baseline[1] == zero[1]
    assert baseline[2] == zero[2]


def test_right_refresh_uses_only_slack_and_preserves_original_rows():
    model = TinyKVModel(controlled=True).eval()
    prompt = torch.ones(1, 31, dtype=torch.long)
    baseline = FixedOptimus(model, "llada21", 8, "ibs",
                            physical_caps=(8,)).run(prompt, 96, 17, 0, [4])
    refresh = FixedOptimus(model, "llada21", 8, "ibs", physical_caps=(8,),
                           prefix_refresh=4, prefix_order="right").run(
                               prompt, 96, 17, 0, [4])
    assert torch.equal(baseline[0], refresh[0])
    base_steps, refresh_steps = _steps(baseline[1]), _steps(refresh[1])
    assert len(base_steps) == len(refresh_steps)
    assert any(e["prefix_refresh_added"] for e in refresh_steps)
    for before, after in zip(base_steps, refresh_steps, strict=True):
        for key in ("block", "step", "cached_prefix_before", "cached_prefix_after",
                    "selected_local", "selected_masked", "accepted", "edited",
                    "edited_positions", "candidate_local", "dirty_before",
                    "written_absolute", "visible_kv", "active_width"):
            assert before[key] == after[key], key
        _check_event(after, 31)


def test_oldest_refresh_picks_the_oldest_eligible_rows():
    model = TinyKVModel(controlled=True).eval()
    prompt = torch.ones(1, 31, dtype=torch.long)
    right = _steps(FixedOptimus(model, "llada21", 8, "ibs", physical_caps=(8,),
                                prefix_refresh=4, prefix_order="right").run(
                                    prompt, 96, 17, 0, [4])[1])
    oldest = _steps(FixedOptimus(model, "llada21", 8, "ibs", physical_caps=(8,),
                                 prefix_refresh=4, prefix_order="oldest").run(
                                     prompt, 96, 17, 0, [4])[1])
    assert len(right) == len(oldest)
    assert any(e["prefix_refresh_added"] for e in oldest)
    for r, o in zip(right, oldest, strict=True):
        assert r["prefix_refresh_eligible"] == o["prefix_refresh_eligible"]
        _check_event(o, 31)
        if o["prefix_refresh_added"]:
            assert o["prefix_refresh_absolute"] == o["prefix_refresh_eligible"][:o["prefix_refresh_added"]]
            assert r["prefix_refresh_absolute"] == sorted(
                o["prefix_refresh_eligible"][-o["prefix_refresh_added"]:])


def test_quota_is_limited_by_slack_and_eligible_rows():
    model = TinyKVModel(controlled=True).eval()
    prompt = torch.ones(1, 31, dtype=torch.long)
    events = FixedOptimus(model, "llada21", 8, "ibs", physical_caps=(8,),
                          prefix_refresh=999, prefix_order="right").run(
                              prompt, 96, 17, 0, [4])[1]
    steps = _steps(events)
    assert any(e["prefix_refresh_added"] for e in steps)
    for e in steps:
        _check_event(e, 31)
        assert e["prefix_refresh_added"] == min(
            e["prefix_refresh_quota"], e["prefix_refresh_slack"],
            len(e["prefix_refresh_eligible"]))


def test_refresh_writes_extras_through_the_real_kv_path_and_never_samples_them():
    model = TinyKVModel(controlled=True).eval()
    prompt = torch.ones(1, 31, dtype=torch.long)
    events = FixedOptimus(model, "llada21", 8, "ibs", physical_caps=(8,),
                          prefix_refresh=4, prefix_order="right").run(
                              prompt, 96, 17, 0, [4])[1]
    steps = _steps(events)
    injected = [e for e in steps if e["prefix_refresh_added"]]
    assert injected
    for e in injected:
        _check_event(e, 31)
        assert set(e["prefix_refresh_absolute"]).isdisjoint(e["edited_positions"])
        assert set(e["prefix_refresh_absolute"]).isdisjoint(e["written_absolute"])

    # Every projected row is written through past_key_values in ascending
    # order, and each injection forward rewrites its own added rows.
    cache = RecordingCache.instances[-1]
    layers = len(model.layers)
    assert len(cache.writes) == layers * len(events)
    per_forward = [set().union(*(set(w[1]) for w in cache.writes[i * layers:(i + 1) * layers]))
                   for i in range(len(events))]
    for e, writes in zip(events, per_forward, strict=True):
        assert writes == set(e["absolute_positions"])
    for i, e in enumerate(events):
        if e["phase"] != "prefill" and e["prefix_refresh_added"]:
            assert set(e["prefix_refresh_absolute"]) <= per_forward[i]

    # The original optimus mask is all-visible through the unchanged cache end;
    # prefill keeps its block-causal mask.
    for e, (positions, end, mask) in zip(events, model.calls, strict=True):
        assert mask.shape == (len(positions), end)
        if e["phase"] == "prefill":
            expected = positions[:, None] // 32 >= torch.arange(end)[None, :] // 32
            assert torch.equal(mask, expected)
        else:
            assert bool(mask.all())

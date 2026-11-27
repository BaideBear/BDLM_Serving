"""CPU only: real random Gemma layers and sampler; no checkpoint."""
from contextlib import ExitStack

import pytest
import torch

import test_gemma_strict_request as baseline
from test_gemma_strict_request import B, PROMPT, inputs, model
from experiments.t2t.serving_slots import gemma_strict_close as arm
from experiments.t2t.serving_slots.gemma_strict_request import StrictRequest
from experiments.t2t.serving_slots.gemma_unified_audit import projectors


def request(model):
    return arm.ClosingRequest(model, inputs(), seed=19, max_new_tokens=B)


@pytest.mark.parametrize("q,debt", [
    (32, tuple(range(137, 208))),  # Two all-dirty calls, then a mixed call.
    (64, ()), (128, (17, 243)), (256, tuple(range(B))),
])
def test_dirty_first_physical_writes_and_frozen_inputs(model, monkeypatch, q, debt):
    r = request(model)
    r.step(B)  # Real initialized resident bank, conditional scores, and sampled IDs.
    rows, s, bank = r.rows, r.sampler, r.lane.bank
    rows.max_steps = rows.age  # Controlled terminal boundary, not a quality experiment.
    rows.dirty = frozenset(debt)
    remaining, unseen = set(debt), set(range(B))
    age, ids, rng = s.age, s.ids.clone(), s.rng.get_state().clone()
    sc = torch.stack([s.sc[i].clone() for i in range(B)])[None]
    projected, sealed = [], {}
    live_extend = bank.extend

    def extend(layer, q_, k, v, mask):
        result = live_extend(layer, q_, k, v, mask)
        slots = torch.tensor(bank.positions) - bank.base
        for store, src in zip(bank.buffers[layer], (k, v), strict=True):
            assert torch.equal(store.index_select(-2, slots), src[..., -len(slots):, :])
        projected.append((layer, tuple(bank.positions)))
        return result

    def check_input(_, args, kw):
        idx = kw["decoder_position_ids"][0].long() - len(PROMPT)
        assert torch.equal(kw["decoder_input_ids"], ids.index_select(1, idx))
        assert torch.equal(kw["self_conditioning_logits"], sc.index_select(1, idx))

    def save_logits(_, args, out):
        pos = [i-len(PROMPT) for i in bank.positions]
        assert not sealed.keys() & set(pos)
        sealed.update(zip(pos, out.logits[0].argmax(-1).tolist(), strict=True))

    monkeypatch.setattr(bank, "extend", extend)
    n = 0
    with ExitStack() as stack:
        widths, pos = stack.enter_context(projectors(model))
        stack.enter_context(model.model.decoder.register_forward_pre_hook(check_input, with_kwargs=True))
        stack.enter_context(model.register_forward_hook(save_logits))
        while r.status == "ready":
            before = sorted(remaining)
            picked = before[:q]
            picked += sorted(unseen - remaining)[:q-len(picked)]
            e = r.step(q)
            assert e["phase"] == "verify" and e["rows"] == sorted(picked)
            assert e["dirty_before"] == before and e["q"] == len(picked) <= e["offer"] == q
            assert not any(e[k] for k in ("prefix", "sampled", "admitted", "edited", "revoked", "encoded"))
            absolute = tuple(len(PROMPT)+i for i in e["rows"])
            assert pos[-1] == list(absolute)
            assert projected == [(i, absolute) for i in range(len(model.model.decoder.layers))]
            projected.clear()
            remaining.difference_update(picked)
            unseen.difference_update(picked)
            assert rows.dirty == remaining
            assert s.age == age and torch.equal(s.rng.get_state(), rng) and torch.equal(s.ids, ids)
            assert torch.equal(torch.stack([s.sc[i] for i in range(B)])[None], sc)
            n += 1
    assert not remaining and not unseen and all(t == [q]*n for t in widths.values())
    assert s.close_q == B and n == B//q
    assert r.tokens == [sealed[i] for i in range(B)]


@pytest.mark.parametrize("q", [32, 64, 128, 256])
def test_paired_only_terminal_drains_removed(model, q):
    kw = dict(seed=19, max_new_tokens=B, prefix_quota=4)
    _, old = baseline.generate(model, inputs(), offers=(q,), **kw)
    _, new = arm.generate(model, inputs(), offers=(q,), **kw)
    start = max(i for i, e in enumerate(old) if e["phase"] == "sample") + 1
    assert old[:start] == new[:start]
    assert len(new) > start and all(e["phase"] == "verify" for e in new[start:])
    drains = [e for e in old[start:] if e["phase"] == "drain"]
    assert set(new[start]["dirty_before"]) == {i for e in drains for i in e["rows"]}
    assert len(old)-len(new) == len(drains)
    assert sum(e["q"] for e in old)-sum(e["q"] for e in new) == len(new[start]["dirty_before"])
    assert set(new[start]) == set(old[-1])


def test_cyclic_multiblock_eos_encoder_billing(model, monkeypatch):
    # Reuse the independent hook audit, not its generator. Checks encoder input IDs
    # against terminal argmax and charges exactly D[, E] per event, excluding prefill.
    monkeypatch.setattr(baseline, "generate", arm.generate)
    offers = (128, 32, 64, 32, 64)
    tokens, trace = baseline.audited(model, offers, limit=2*B, blocks=2, quota=4)
    assert sum(e["encoded"] for e in trace) == 1
    eos = next(t for t in tokens[1:B] if t != tokens[0])
    monkeypatch.setattr(model.generation_config, "eos_token_id", [eos])
    short, trace = baseline.audited(model, offers, limit=2*B, blocks=1, quota=4)
    assert short == tokens[:tokens.index(eos)+1] and not any(e["encoded"] for e in trace)
    assert sum(e["q"] for e in trace if e["phase"] == "verify") == B


def test_incomplete_and_failure_never_seal_or_retire_dirty(model, monkeypatch):
    r = request(model)
    r.step(32)
    r.rows.max_steps = r.rows.age
    with pytest.raises(RuntimeError, match="incompletely"):
        r.step(32)
    assert r.status == "failed" and not r.tokens
    r = request(model)
    r.step(B)
    r.rows.max_steps = r.rows.age
    debt = r.rows.dirty
    assert debt

    def fail(*args, **kwargs):
        raise RuntimeError("layer failed")

    monkeypatch.setattr(r.lane.bank, "extend", fail)
    with pytest.raises(RuntimeError, match="layer failed"):
        r.step(64)
    assert r.status == "failed" and r.rows.dirty == debt and r.cursor == 0 and r.sampler.close_q == 0
    with pytest.raises(RuntimeError, match="failed"):
        r.step(64)


def test_public_api_inherits_init_and_step(model):
    assert arm.ClosingRequest.__init__ is StrictRequest.__init__
    assert arm.ClosingRequest.step is StrictRequest.step
    r = request(model)
    with pytest.raises(ValueError):
        r.step(512)
    assert r.tick == 0 and r.status == "ready"

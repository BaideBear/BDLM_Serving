"""CPU contracts for physical prefix refresh versus editable denoising on B256."""
from contextlib import ExitStack

import pytest
import torch

from test_gemma_strict_request import B, PROMPT, V, inputs, model
from experiments.t2t.serving_slots.gemma_prefix_action import PrefixActionRequest
from experiments.t2t.serving_slots.gemma_strict_request import StrictRequest
from experiments.t2t.serving_slots.gemma_unified_audit import projectors


def request(model, quota, action=None):
    kw = dict(seed=19, max_new_tokens=B, prefix_mode="right", prefix_quota=quota)
    if action is None:
        return StrictRequest(model, inputs(), **kw)
    return PrefixActionRequest(model, inputs(), prefix_action=action, **kw)


def replay(r, offers):
    trace = []
    while r.status == "ready":
        trace.append(r.step(offers[len(trace) % len(offers)]))
    assert r.status == "done" and len(r.tokens) == B
    return trace


@pytest.mark.parametrize("offers", [(64,), (32, 128, 64, 256)])
@pytest.mark.parametrize("quota", [0, 32])
def test_no_optional_rows_reproduce_registered_request(model, offers, quota):
    # Unmodified tiny-model logits with this seed never have slack + a prefix.
    control = request(model, 0)
    old_sampler = control.sampler
    expected = replay(control, offers)
    for action in ("kv_only", "editable"):
        r = request(model, quota, action)
        s = r.sampler
        actual = replay(r, offers)
        assert all(not e["prefix"] for e in actual)
        assert len(actual) == len(expected)
        for event, ref in zip(actual, expected, strict=True):
            assert {k: event[k] for k in ref} == ref
        assert r.tokens == control.tokens and s.sample_q == old_sampler.sample_q
        assert torch.equal(s.rng.get_state(), old_sampler.rng.get_state())
        assert torch.equal(s.ids, old_sampler.ids) and torch.equal(s.accepted, old_sampler.accepted)
        assert all(torch.equal(s.sc[i], old_sampler.sc[i]) for i in range(B))


@pytest.mark.parametrize("offers", [(64,), (64, 128, 32), (32, 64, 128, 256)])
@pytest.mark.parametrize("action", ["editable", "kv_only"])
def test_k32_physical_rows_sampler_and_sc(model, monkeypatch, action, offers):
    r = request(model, 32, action)
    s, bank = r.sampler, r.lane.bank
    all_rows = set(range(B))
    positions_seen, sampled, draws, writes, head = [], [], [], [], []
    first_warm_age = None

    def scripted_head(_, args, kw, out):
        # Change only returned scores, after real CPU decoder/LM projections.
        # Cold uniform tail leaves unresolved rows. First warm sample keeps a
        # few open; on re-entry mandatory accepted rows can really change ID.
        nonlocal first_warm_age
        pos = kw["decoder_position_ids"][0].long() - len(PROMPT)
        ids = kw["decoder_input_ids"]
        assert ids.device.type == "cpu" and torch.equal(ids, s.ids[:, pos])
        history = [s.sc.get(i) for i in pos.tolist()]
        sc = kw["self_conditioning_logits"]
        if all(x is None for x in history):
            assert sc is None
        else:
            for i, previous in enumerate(history):
                if previous is not None:
                    assert torch.equal(sc[0, i], previous)
        warm = len(r.rows.initialized) == B
        if warm and first_warm_age is None:
            first_warm_age = s.age
        scores = torch.full_like(out.logits, -64.)
        scores.scatter_(-1, ((ids + 1) % V).unsqueeze(-1), 64.)
        if not warm or s.age == first_warm_age:
            scores[:, pos >= (252 if warm else 240)] = 0.
        scores += pos[None, :, None] / B  # Row signature even for two uniform rows.
        out.logits = scores
        positions_seen.append((pos.tolist(), scores))

    live_consume, live_draw, live_extend = s.consume, torch.multinomial, bank.extend

    def consume(rows, logits):
        physical, scores = positions_seen[-1]
        assert torch.equal(logits, scores[:, [physical.index(i) for i in rows]])
        sampled.append(list(rows))
        return live_consume(rows, logits)

    def draw(probs, *args, **kw):
        if kw.get("generator") is s.rng:
            draws.append(probs.shape[0])
        return live_draw(probs, *args, **kw)

    def extend(layer, q, k, v, mask):
        result = live_extend(layer, q, k, v, mask)
        slots = torch.tensor(bank.positions) - bank.base
        for buf, src in zip(bank.buffers[layer], (k, v), strict=True):
            assert torch.equal(buf.index_select(-2, slots), src[..., -len(slots):, :])
        writes.append((layer, list(bank.positions)))
        return result

    monkeypatch.setattr(s, "consume", consume)
    monkeypatch.setattr(torch, "multinomial", draw)
    monkeypatch.setattr(bank, "extend", extend)
    cold = mixed_cold = full_prefix = dirty_t2t = editable_t2t = False
    with ExitStack() as hooks:
        widths, absolute = hooks.enter_context(projectors(model))
        hooks.enter_context(model.lm_head.register_forward_pre_hook(lambda _, a: head.append(a[0].shape[1])))
        hooks.enter_context(model.register_forward_hook(scripted_head, with_kwargs=True))
        while r.status == "ready":
            offer = offers[r.tick % len(offers)]
            init, debt = set(r.rows.initialized), set(r.rows.dirty)
            before = set(s.accepted[0].nonzero().flatten().tolist())
            ids = s.ids.clone()
            old_sc = {i: x.clone() for i, x in s.sc.items()}
            age, sample_q = s.age, s.sample_q
            sampled.clear(); draws.clear(); writes.clear()
            e = r.step(offer)
            rows, pref = e["rows"], set(e["prefix"])
            assert e["dirty_before"] == sorted(debt) and e["offer"] == offer
            assert rows == positions_seen[-1][0] == sorted(set(rows))
            assert e["q"] == len(rows) <= offer and set(sorted(debt)[:offer]) <= set(rows)
            assert len(head) == len(absolute) == len(positions_seen) == r.tick
            assert absolute[-1] == [len(PROMPT) + i for i in rows]
            assert head[-1] == len(rows) and all(tape == head for tape in widths.values())
            assert writes == [(layer, absolute[-1]) for layer in range(len(model.model.decoder.layers))]
            if e["phase"] != "sample":
                assert not (pref or e["sampled"] or sampled or draws)
                assert (s.age, s.sample_q) == (age, sample_q)
                continue
            if init != all_rows:
                cold = True
                mixed_cold |= bool(debt)
                expected = sorted(all_rows - init)[:offer - len(debt)]
                assert not pref and rows == sorted(debt.union(expected))
            else:
                missing = all_rows - before
                ordinary = debt.union(sorted(missing - debt)[:offer - len(debt)])
                pool = sorted(set(range(min(missing))) - debt, reverse=True)
                assert pref == set(pool[:min(32, offer - len(ordinary))])
                assert rows == sorted(ordinary | pref)
                expected = sorted(ordinary) if action == "kv_only" else rows
                full_prefix |= len(pref) == 32
            assert sampled == [expected] and e["sampled"] == expected
            assert draws == [len(expected)] and (s.age, s.sample_q) == (age + 1, sample_q + len(expected))
            after = set(s.accepted[0].nonzero().flatten().tolist())
            changed = set((s.ids != ids)[0].nonzero().flatten().tolist())
            edits = before & after & changed
            assert edits == set(e["edited"])
            dirty_t2t |= bool(edits & debt - pref)
            editable_t2t |= bool(edits & pref)
            untouched = all_rows - set(expected)
            assert not ((before ^ after) | changed) & untouched
            assert s.sc.keys() == old_sc.keys() | set(expected)
            assert all(torch.equal(s.sc[i], old_sc[i]) for i in old_sc.keys() & untouched)
            if action == "editable" and pref:
                assert all(not torch.equal(s.sc[i], old_sc[i]) for i in pref)

    assert r.status == "done" and s.close_q == len(r.tokens) == B
    assert cold and full_prefix and dirty_t2t
    assert editable_t2t == (action == "editable")
    if len(offers) > 1:
        assert mixed_cold

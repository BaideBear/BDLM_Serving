"""Planner and event contract only: no checkpoint, model or attention forward."""
from types import SimpleNamespace

import pytest
import torch

from bdlm_serving.methods.t2t.budget_family import Law
from bdlm_serving.methods.t2t.llada_budget_request import LLaDABudgetRequest


def request(arm, origin=58, end=131):
    r = object.__new__(LLaDABudgetRequest)
    r.ablation, r.law, r.tail_rule = arm, Law(1, 8), "eager"
    r.origin, r.head, r.end = origin, origin//32*32, end
    r.mask_id, r.eos_ids, r.threshold, r.edit_threshold = 0, {9}, .9, .5
    r.x = torch.zeros(1, end, dtype=torch.long)
    r.x[:, :origin] = 1
    r.bank = SimpleNamespace(initialized=set(range(origin)), last_end=origin)
    r.pending_build, r.next_side = None, 0
    r.stopping, r.dirty, r.records, r.tokens = False, set(), [], []
    r.sample_clock = r._stall = 0
    r.status = "ready"

    def project(rows, cap, phase):
        assert rows and len(rows) <= cap and tuple(sorted(set(rows))) == rows
        end = max(r.bank.last_end, rows[-1]+1)
        assert set(range(r.bank.last_end, end)) <= set(rows)
        r.bank.initialized.update(rows)
        r.bank.last_end = end
        return {"phase": phase}, torch.tensor([rows])  # addresses, not model logits

    def proposals(addresses, **_):
        return torch.full_like(addresses, 2), torch.ones_like(addresses, dtype=torch.float)

    r._project, r.model = project, SimpleNamespace(_sample_with_temperature_topk_topp=proposals)
    return r


def test_q64_three_fresh_requests_identical_including_partial_blocks_and_debt():
    results = []
    for arm in ("00", "10", "11"):
        r = request(arm)
        while r.status == "ready":
            e = r.step(64)
            assert e["phase"] == "dense_joint"
        first, second = r.records[:2]
        assert first["rows"] == first["sampled"] == list(range(58,96))
        assert first["position_updates"] == [[i,2] for i in range(58,96)]
        assert second["dirty"] == second["rows"] == first["rows"]
        assert not second["sampled"]
        assert r.records[-1]["rows"][-1] == 130
        results.append((r.records, r.tokens))
    assert results[0] == results[1] == results[2]


def test_downshift_00_refreshes_parked_rows_without_discretionary_tail():
    r = request("00")
    r.step(64)
    while r.records[-1]["dirty_after"] and r.head == 32:
        assert r.step(4)["phase"] == "drain"
    # A deliberately unresolved head and dirty parked row on a later small call.
    r = request("00")
    r.bank.initialized.update(range(58,96))
    r.bank.last_end, r.dirty = 96, {70}
    r.x[0,70] = 2
    e = r.step(4)
    assert e["dirty"] == [70] and 70 not in e["sampled"]
    assert all(p < 64 for p in e["rows"] if p != 70)


@pytest.mark.parametrize("q", (4, 8, 16, 32))
def test_gate_effective_length_build_samples_then_dirty_first(q):
    a, b, c = (request(arm) for arm in ("00", "10", "11"))
    for r in (a, b, c):
        # Effective width is six; two masks exceed a quarter; one qualifies.
        r.x[0,58:62] = 2
        assert max(r._ablation_plan(q)[0].rows) < 64  # not resident yet
        r.bank.initialized.update(range(58,64))
        r.bank.last_end = 64
        assert max(r._ablation_plan(q)[0].rows) < 64
        r.x[0,62] = 2
    assert max(a._ablation_plan(q)[0].rows) < 64
    plan, sampled, _, _ = b._ablation_plan(q)
    assert 63 in sampled and 64 in sampled and len(plan.rows) <= q
    c.dirty = {62}
    assert c.step(q)["rows"] == [62] and not c.records[-1]["sampled"]
    e = c.step(q)
    assert e["phase"] == "build_sample" and e["rows"] == e["sampled"] == list(range(64,96))
    assert e["dirty_after"] == e["admitted"] == e["rows"]
    assert c.step(q)["dirty"] == list(range(64,64+q))


def test_gate_eos_and_prefix_independent_in_both_blocks():
    r = request("10", 32)
    r.bank.initialized.update(range(32,96))
    r.bank.last_end = 96
    r.x[0,32:58] = r.x[0,64:70] = 2
    p = r._ablation_plan(32)[0]
    assert set(p.prefix) == {56,57,68,69}
    r.x[0,55] = 9
    for q in (4,32,64):
        assert max(r._ablation_plan(q)[0].rows) < 64


def test_dense_joint_edits_and_eos_invalidate_positioned_tail():
    r = request("00")
    r.x[0,59] = 3
    r.bank.initialized.update(range(58,61))
    r.bank.last_end, r.dirty = 61, {59}
    def proposals(addresses, **_):
        ids = torch.where(addresses == 58, 9, 2)
        return ids, torch.ones_like(ids, dtype=torch.float)
    r.model._sample_with_temperature_topk_topp = proposals
    e = r.step(64)
    assert e["dirty"] == e["edited"] == [59] and [65,2] in e["position_updates"]
    e = r.step(64)
    assert e["invalidate_from"] == 59 and r.tokens == [9] and r.status == "done"
    assert 65 in e["dirty"] and not e["sampled"]

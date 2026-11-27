"""Model-free contracts for the opt-in warm-head sample single factor.

No checkpoint, tokenizer, GPU, or CPU inference of a real model: a planner-only
request drives the real planner/ledger through a fake projector that records
the one physical call per step. These are state/plan contracts, not generation
qualification.
"""
import inspect
from types import SimpleNamespace

import pytest
import torch

from bdlm_serving.methods.t2t.budget_family import Law
from bdlm_serving.methods.t2t.llada_budget_request import B, LLaDABudgetRequest


def planner(*, origin=32, end=131, cross="half", switchable=False, top_gate="half",
            warm_head_sample=False, resident=(), resolved=(), masks=(), dirty=(),
            last_end=None, accept=False):
    r = object.__new__(LLaDABudgetRequest)
    r.ablation, r.law, r.tail_rule = None, Law(1, 8), "eager"
    r.cross, r.borrow = cross, False
    r.switchable, r.top_gate, r.lower_borrow = switchable, top_gate, False
    r.coverage = "head"
    r.warm_head_sample = warm_head_sample
    r.audit_admission = False
    r._cross_evidence = r._switch_evidence = None
    r.pending_build, r.next_side = None, 0
    r.origin, r.head, r.end = origin, origin // B * B, end
    r.mask_id, r.eos_ids, r.threshold, r.edit_threshold = 0, {9}, 0.9, 0.5
    r.x = torch.zeros(1, end, dtype=torch.long)
    r.x[:, :origin] = 1
    r.bank = SimpleNamespace(initialized=set(range(origin)), last_end=origin, capacity=end)
    r.stopping, r.dirty, r.records, r.tokens = False, set(), [], []
    r.sample_clock = r._stall = 0
    r.status = "ready"
    r.calls, r.sampler_widths = [], []
    if resident:
        r.bank.initialized.update(resident)
        r.bank.last_end = max(r.bank.last_end, max(resident) + 1)
    if last_end is not None:
        r.bank.last_end = last_end
    for p in resolved:
        r.x[0, p] = 2
    for p in masks:
        r.x[0, p] = 0
    if dirty:
        r.dirty = set(dirty)

    def project(rows, cap, phase):
        assert rows and len(rows) <= cap and tuple(sorted(set(rows))) == rows
        stop = max(r.bank.last_end, rows[-1] + 1)
        assert tuple(p for p in rows if p >= r.bank.last_end) == tuple(range(r.bank.last_end, stop))
        r.bank.initialized.update(rows)
        r.bank.last_end = stop
        r.calls.append((tuple(rows), cap, phase, len(rows)))
        return {"phase": phase, "layer_physical_q": [len(rows)] * 2}, torch.tensor([rows])

    def proposals(addresses, **_):
        r.sampler_widths.append(addresses.shape[-1])
        return torch.full_like(addresses, 2), torch.full(addresses.shape, 1.0 if accept else 0.5)

    r._project = project
    r.model = SimpleNamespace(_sample_with_temperature_topk_topp=proposals)
    return r


def warm_state():
    """Resident head with 12 warm MASKs plus a fully cold tail; debt is empty."""
    return dict(origin=32, end=131, resident=range(32, 64),
                resolved=(*range(32, 40), *range(52, 64)),
                masks=(*range(40, 52), *range(64, 96)))


HEAD_WARM = set(range(40, 52))


def snapshot(r):
    return dict(x=r.x.clone(), head=r.head, end=r.end, dirty=sorted(r.dirty),
                last_end=r.bank.last_end, sample_clock=r.sample_clock,
                status=r.status, stopping=bool(r.stopping))


def same_state(a, b):
    return (torch.equal(a["x"], b["x"])
            and all(a[k] == b[k] for k in ("head", "end", "dirty", "last_end",
                                           "sample_clock", "status", "stopping")))


def test_default_off_keeps_the_old_control_plan_and_record_schema():
    old = planner(**warm_state())
    e = old.step(64)
    assert set(e["sampled"]) == set(e["cold"]) == set(range(64, 96))
    assert set(e["rows"]) == set(range(32, 96))
    assert "warm_head_sample" not in e and "warm_head_rows" not in e
    assert len(e["rows"]) == e["q"] <= 64


def test_opt_in_adds_only_selected_warm_head_masks_without_extra_q():
    old = planner(**warm_state())
    new = planner(warm_head_sample=True, **warm_state())
    a, b = old.step(64), new.step(64)
    assert b["rows"] == a["rows"]                      # no extra Q
    assert b["q"] == a["q"] == len(a["rows"])
    assert set(b["sampled"]) == set(a["sampled"]) | HEAD_WARM
    assert b["warm_head_sample"] == len(HEAD_WARM)
    assert set(b["warm_head_rows"]) == HEAD_WARM
    assert set(b["warm_head_rows"]) <= set(b["rows"])
    assert not set(b["warm_head_rows"]) & (set(b["prefix"]) | set(b["dirty"]) | set(b["cold"]))
    assert len(b["sampled"]) == len(b["cold"]) + len(HEAD_WARM)
    assert b["layer_physical_q"] == [len(b["rows"])] * 2  # one forward, reused logits
    assert len(new.sampler_widths) == 1 and new.sampler_widths[0] == len(b["sampled"])


def test_first_intervention_has_identical_pre_state_and_rows_then_may_diverge():
    old = planner(**warm_state())
    new = planner(warm_head_sample=True, **warm_state())
    first = None
    for _ in range(8):
        if old.status != "ready" or new.status != "ready":
            break
        so, sn = snapshot(old), snapshot(new)
        eo, en = old.step(64), new.step(64)
        if en.get("warm_head_sample", 0):
            first = so, sn, eo, en
            break
    assert first is not None
    so, sn, eo, en = first
    assert same_state(so, sn)                      # pre-state identical
    assert eo["rows"] == en["rows"]               # physical Q identical
    assert eo["dirty_before"] == en["dirty_before"]
    assert set(en["sampled"]) == set(eo["sampled"]) | set(en["warm_head_rows"])
    # Subsequent trajectories are allowed to differ; no equality is asserted here.


def test_cold_head_mask_disables_the_intervention():
    r = planner(warm_head_sample=True, resident=range(32, 63),
                resolved=(*range(32, 40), *range(52, 63)),
                masks=(*range(40, 52), 63, *range(64, 96)), last_end=63)
    e = r.step(64)
    assert e["warm_head_sample"] == 0 and e["warm_head_rows"] == []
    assert 63 in e["sampled"] and 63 in e["cold"]  # cold head is still sampled


def test_no_warm_head_mask_disables_the_intervention():
    r = planner(warm_head_sample=True, resident=range(32, 64),
                resolved=range(32, 64), masks=range(64, 96))
    e = r.step(64)
    assert e["warm_head_sample"] == 0 and e["warm_head_rows"] == []
    assert set(e["sampled"]) == set(range(64, 96))


def test_full_debt_drain_disables_the_intervention():
    debt = set(range(32, 96))
    r = planner(warm_head_sample=True, resident=debt, resolved=debt,
                masks=range(64, 96), dirty=debt)
    e = r.step(64)
    assert e["drain_only"] and not e["sampled"]
    assert e["warm_head_sample"] == 0 and e["warm_head_rows"] == []


def test_unpayable_global_debt_disables_the_intervention():
    r = planner(warm_head_sample=True, **{**warm_state(), "dirty": set(range(96, 100))})
    e = r.step(64)
    assert e["warm_head_sample"] == 0 and e["warm_head_rows"] == []
    assert set(e["sampled"]) == set(range(64, 96))


def test_switchable_full_head_also_honours_the_flag():
    r = planner(switchable=True, cross=None, top_gate="half",
                warm_head_sample=True, **warm_state())
    e = r.step(64)
    assert e["warm_head_sample"] == len(HEAD_WARM)
    assert set(e["sampled"]) == set(e["cold"]) | HEAD_WARM
    assert e["coverage"] == "head" and e["head_full"] is True


def test_flag_is_default_off_and_rejects_non_head_coverage():
    sig = inspect.signature(LLaDABudgetRequest.__init__)
    assert sig.parameters["warm_head_sample"].default is False
    from test_llada_budget_request import Measured

    def make(**kw):
        return LLaDABudgetRequest(Measured(), torch.ones((1, 32), dtype=torch.long),
                                  mask_id=0, max_new_tokens=69, **kw)

    make(cross="half", coverage="head", warm_head_sample=True)
    with pytest.raises(ValueError):
        make(cross="half", coverage="sparse", warm_head_sample=True)
    with pytest.raises(ValueError):
        make(cross="half", coverage="head", warm_head_sample=1)

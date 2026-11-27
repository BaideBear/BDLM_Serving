"""Registered C64/C16/C8 cross-block contracts on the measured fake model.

No checkpoint, no tokenizer, no CPU inference of a real model: the fake issues
known token proposals through the real executor and the real KV accounting.
"""
import pytest
import torch

from bdlm_serving.methods.t2t.llada_budget_request import LLaDABudgetRequest
from test_llada_budget_request import Measured, drive

LEVELS = ("tick", "q", "rows", "sampled", "dirty_before", "dirty", "dirty_after",
          "cold", "mask", "prefix", "written_absolute", "admitted", "edited",
          "revoked", "retired_blocks", "drain_only", "phase", "sample_clock")


def make(q, cross, borrow=False, prompt=32, cap=69, mode="high"):
    torch.manual_seed(20)
    model = Measured(mode)
    req = LLaDABudgetRequest(model, torch.ones((1, prompt), dtype=torch.long),
                             mask_id=0, max_new_tokens=cap, cross=cross, borrow=borrow)
    return req, model


def run(q, cross, borrow=False):
    req, model = make(q, cross, borrow)
    drive(req, model, [q])
    return req


def levels(records):
    return [{k: e[k] for k in LEVELS} for e in records]


@pytest.mark.parametrize("q,mode", [(64, "eager"), (32, "head"), (16, "head"), (8, "head")])
def test_cross_modes_reproduce_legacy_family_when_neutral(q, mode):
    legacy = run(q, None)
    cross = run(q, mode)
    assert legacy.tokens == cross.tokens
    assert levels(legacy.records) == levels(cross.records)
    assert all("cross_open" not in e for e in legacy.records)


def test_c64_off_never_projects_the_next_block_and_equals_c32():
    off = run(64, "off")
    c32 = run(32, None)
    assert off.tokens == c32.tokens
    assert levels(off.records) == levels(c32.records)
    assert all(e["tail_q"] == 0 and e["head_q"] == e["q"] for e in off.records)


def test_c64_eager_opens_on_the_first_offer():
    req = run(64, "eager")
    first = req.records[0]
    assert first["tail_q"] == 32 and first["head_q"] == 32 and first["tail_opened"]
    assert set(first["tail_read_only"]) == {r for r in first["rows"] if r >= 64} - set(first["sampled"])


def _stall_two_steps(mode, keep, borrow=False):
    """Build a resident head block, then leave exactly `keep` MASKs in it."""
    torch.manual_seed(20)
    model = Measured("stall")
    req = LLaDABudgetRequest(model, torch.ones((1, 32), dtype=torch.long), mask_id=0,
                             max_new_tokens=69, cross=mode, borrow=borrow)
    if mode in ("off", "eager", "half", "quarter"):
        first = req.step(64)
    else:
        req.step(16)
        first = req.step(16)  # both halves of the head block are now resident
    if mode in ("half", "quarter"):
        assert first["tail_q"] == 0 and first["head_q"] == 32  # resident, still gated
    with torch.inference_mode():
        req.x[0, 32:64 - keep] = 3
    req.dirty.clear()
    return req, first


@pytest.mark.parametrize("mode,keep,opens", [("half", 20, False), ("half", 12, True),
                                             ("quarter", 12, False), ("quarter", 6, True)])
def test_half_and_quarter_open_only_after_the_registered_mask_share(mode, keep, opens):
    req, _ = _stall_two_steps(mode, keep)
    e = req.step(64)
    assert e["head_mask_at_plan"] == keep
    assert (e["tail_q"] > 0) is opens
    if opens:
        k = 2 if mode == "half" else 4
        assert e["head_mask_at_plan"] * k <= 32


def test_opening_survives_a_later_mask_revocation():
    req, _ = _stall_two_steps("quarter", 6)
    opened = req.step(64)
    assert opened["tail_q"] > 0
    with torch.inference_mode():
        req.x[0, 32:64] = 0
    req.dirty.clear()
    again = req.step(64)
    assert again["tail_q"] > 0 and again["head_mask_at_plan"] == 32


def test_c16_borrow_keeps_the_head_plan_and_only_reads_the_tail():
    plain, _ = _stall_two_steps("head", 3, borrow=False)
    borrow, _ = _stall_two_steps("head", 3, borrow=True)
    a, b = plain.step(16), borrow.step(16)
    assert a["rows"] == [r for r in b["rows"] if r < 64]
    assert a["sampled"] == b["sampled"]
    assert b["q"] == 16 and b["head_q"] == len(a["rows"]) and b["tail_q"] == 11
    assert b["tail_read_only"] == [r for r in b["rows"] if r >= 64]
    assert not any(r >= 64 for r in b["sampled"])
    assert b["borrowed_rows"] == 11


def test_borrow_waits_for_a_resident_head_block():
    torch.manual_seed(20)
    model = Measured("stall")
    req = LLaDABudgetRequest(model, torch.ones((1, 40), dtype=torch.long), mask_id=0,
                             max_new_tokens=69, cross="head", borrow=True)
    req.step(16)  # only 40..56 resident
    assert req.bank.last_end == 56 < 64
    with torch.inference_mode():
        req.x[0, 40:53] = 3
    req.dirty.clear()
    e = req.step(16)
    assert e["tail_q"] == 0 and e["tail_read_only"] == []
    assert e["q"] == 8  # the cold rows 56..63; no residual reaches the tail


@pytest.mark.parametrize("q", [8, 16])
def test_borrow_never_exceeds_the_offer_or_retires_a_stale_block(q):
    req = run(q, "head", borrow=True)
    for e in req.records:
        assert e["q"] <= q and e["head_q"] + e["tail_q"] == e["q"]
        read_only = set(e["tail_read_only"])
        assert read_only <= set(e["rows"]) - set(e["sampled"])
        assert len(read_only) == e["borrowed_rows"]
    assert len(req.tokens) == 69 and 0 not in req.tokens


def test_borrowed_tail_kv_is_cached_but_never_emitted():
    req, _ = _stall_two_steps("head", 3, borrow=True)
    e = req.step(16)
    roots = [r for r in e["rows"] if r >= 64]
    assert roots and all(r in req.bank.initialized for r in roots)
    assert not set(roots) & set(req.tokens)


def test_invalid_cross_registrations_are_rejected():
    with pytest.raises(ValueError):
        LLaDABudgetRequest(Measured(), torch.ones((1, 32), dtype=torch.long), mask_id=0,
                           cross="sideways")
    with pytest.raises(ValueError):
        LLaDABudgetRequest(Measured(), torch.ones((1, 32), dtype=torch.long), mask_id=0,
                           cross="half", ablation="00")
    req, _ = make(16, "head")
    with pytest.raises(ValueError):
        req.step(64)  # head is registered strictly below one block
    req, _ = make(64, "eager")
    with pytest.raises(ValueError):
        req.step(16)  # eager is registered at the two-block offer only
    req, _ = make(64, "eager", borrow=True)
    with pytest.raises(ValueError):
        req.step(64)  # borrow is a residual-seat option below one block
    req, _ = make(64, "quarter")
    with pytest.raises(ValueError):
        req.step(8)  # the fixed-offer race stays fixed; switchable is the opt-in


# ---------------------------------------------------------------------------
# Switchable registration: one request, changing offers, no KV loss.
# ---------------------------------------------------------------------------

from types import SimpleNamespace

from bdlm_serving.methods.t2t.budget_family import Law

ALL_TIERS = (4, 8, 16, 32, 64)


def planner(origin=58, end=131, top_gate="eager", lower_borrow=True, accept=False):
    """Planner-only request: the fake project checks hole-free contiguous KV."""
    r = object.__new__(LLaDABudgetRequest)
    r.ablation, r.law, r.tail_rule = None, Law(1, 8), "eager"
    r.cross, r.borrow = None, False
    r.switchable, r.top_gate, r.lower_borrow = True, top_gate, lower_borrow
    r._cross_evidence = r._switch_evidence = None
    r.pending_build, r.next_side = None, 0
    r.origin, r.head, r.end = origin, origin // 32 * 32, end
    r.mask_id, r.eos_ids, r.threshold, r.edit_threshold = 0, {9}, .9, .5
    r.x = torch.zeros(1, end, dtype=torch.long)
    r.x[:, :origin] = 1
    r.bank = SimpleNamespace(initialized=set(range(origin)), last_end=origin, capacity=end)
    r.stopping, r.dirty, r.records, r.tokens = False, set(), [], []
    r.sample_clock = r._stall = 0
    r.status = "ready"

    def project(rows, cap, phase):
        assert rows and len(rows) <= cap and tuple(sorted(set(rows))) == rows
        stop = max(r.bank.last_end, rows[-1] + 1)
        assert tuple(p for p in rows if p >= r.bank.last_end) == tuple(range(r.bank.last_end, stop))
        r.bank.initialized.update(rows)
        r.bank.last_end = stop
        return {"phase": phase}, torch.tensor([rows])

    def proposals(addresses, **_):
        return torch.full_like(addresses, 2), torch.full(addresses.shape, 1.0 if accept else .5)

    r._project = project
    r.model = SimpleNamespace(_sample_with_temperature_topk_topp=proposals)
    return r


def audit_step(r, e, q):
    assert e["offer"] == q and 4 <= q <= 64
    assert e["q"] == e["head_q"] + e["tail_q"] <= q
    assert e["rows"] == sorted(set(e["rows"]))
    assert set(e["sampled"]) <= set(e["rows"])
    assert set(e["written_absolute"]) <= set(e["sampled"])
    assert set(e["dirty"]) == set(sorted(e["dirty_before"])[:q])
    assert set(e["dirty_after"]) == ((set(e["dirty_before"]) - set(e["rows"]))
                                     | set(e["written_absolute"]))
    assert set(e["tail_read_only"]) <= set(e["rows"]) - set(e["sampled"])
    assert r.bank.last_end <= r.end


@pytest.mark.parametrize("a", ALL_TIERS)
@pytest.mark.parametrize("b", ALL_TIERS)
def test_switchable_5x5_matrix_keeps_kv_and_clock(a, b):
    r = planner(accept=False)
    for q in (a, a, b, b, a, b):
        init, end, clock, split = set(r.bank.initialized), r.bank.last_end, r.sample_clock, r.head + 32
        e = r.step(q)
        audit_step(r, e, q)
        assert set(r.bank.initialized) >= init          # no KV is ever dropped
        assert r.bank.last_end >= end                   # high-water mark never rewinds
        assert e["sample_clock"] - clock == int(bool(e["sampled"]))
        if e["parked_tail"]:
            assert e["parked_tail"] == sum(1 for p in e["rows"] if split <= p < end)


def test_switchable_accepts_64_to_8_and_back():
    r = planner(top_gate="eager", lower_borrow=True)
    first = r.step(64)
    assert first["tail_q"] > 0 and first["tail_opened"]
    parked = 0
    for q in (8, 8, 4, 16, 64):
        e = r.step(q)
        audit_step(r, e, q)
        parked += e["parked_tail"]
    assert parked > 0  # the 64 offer re-reads rows that stayed resident while parked


def test_downshift_parks_the_tail_block_without_losing_kv():
    r = planner(top_gate="eager")
    r.step(64)
    split = r.head + 32
    resident = {p for p in r.bank.initialized if p >= split}
    end = r.bank.last_end
    assert resident and end > split
    e = r.step(8)
    audit_step(r, e, 8)
    assert set(r.bank.initialized) >= resident and r.bank.last_end == end
    assert e["tail_resident"] == end - split
    again = r.step(64)
    assert again["tail_opened"] and r.bank.last_end >= end


def test_global_dirty_drains_on_downshift_and_never_advances_the_clock():
    r = planner(accept=True, top_gate="eager")
    first = r.step(64)
    debt = set(first["dirty_after"])
    assert len(debt) >= 8
    clock = r.sample_clock
    e = r.step(8)
    audit_step(r, e, 8)
    assert e["drain_only"] and not e["sampled"] and not e["written_absolute"]
    assert e["sample_clock"] == clock
    assert set(e["dirty"]) == set(sorted(debt)[:8])
    assert set(e["rows"]) == set(sorted(debt)[:8])
    assert set(first["dirty_after"]) - set(e["rows"]) == set(e["dirty_after"])


def test_lower_borrow_fires_under_a_switchable_request():
    torch.manual_seed(7)
    model = Measured("stall")
    req = LLaDABudgetRequest(model, torch.ones((1, 32), dtype=torch.long), mask_id=0,
                             max_new_tokens=69, switchable=True, lower_borrow=True)
    req.step(16)
    req.step(16)
    with torch.inference_mode():
        req.x[0, 32:61] = 3
    req.dirty.clear()
    e = req.step(16)
    assert e["q"] == 16 and e["tail_q"] == 11 and e["borrowed_rows"] == 11
    assert not any(p >= 64 for p in e["sampled"])


def test_switchable_and_fixed_cross_are_mutually_exclusive():
    with pytest.raises(ValueError):
        LLaDABudgetRequest(Measured(), torch.ones((1, 32), dtype=torch.long), mask_id=0,
                           switchable=True, cross="eager")
    with pytest.raises(ValueError):
        LLaDABudgetRequest(Measured(), torch.ones((1, 32), dtype=torch.long), mask_id=0,
                           switchable=True, top_gate="sideways")

"""Model-free contracts for the switchable full-head registration.

No checkpoint, tokenizer, or CPU inference of a real model: every step below is
driven through a planner-only fake projector that enforces the same contiguous
KV high-water rule as ``_Bank``. These are state/plan contracts, not generation
qualification.
"""
import inspect
from types import SimpleNamespace

import pytest
import torch

from bdlm_serving.methods.t2t.budget_family import Law
from bdlm_serving.methods.t2t.llada_budget_request import B, LLaDABudgetRequest

TIERS = (4, 8, 16, 32, 64)
GATES = ("off", "eager", "half", "quarter")
PLAN_FIELDS = ("rows", "dirty", "cold", "mask", "prefix", "drain", "sampled", "phase", "cap")


def planner(*, origin=32, end=131, switchable=False, top_gate="quarter", cross=None,
            coverage="sparse", lower_borrow=False, resident=(), resolved=(), masks=(),
            dirty=(), last_end=None, accept=False):
    """Planner-only request: no model weights, but the real planner and ledger."""
    r = object.__new__(LLaDABudgetRequest)
    r.ablation, r.law, r.tail_rule = None, Law(1, 8), "eager"
    r.cross, r.borrow = cross, False
    r.switchable, r.top_gate, r.lower_borrow = switchable, top_gate, lower_borrow
    r.coverage = coverage
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
        return {"phase": phase}, torch.tensor([rows])

    def proposals(addresses, **_):
        return torch.full_like(addresses, 2), torch.full(addresses.shape, 1.0 if accept else 0.5)

    r._project = project
    r.model = SimpleNamespace(_sample_with_temperature_topk_topp=proposals)
    return r


def head_state(origin=32, end=131):
    """Resident head with warm MASKs plus a fully cold tail.

    Sparse ``choose`` spends the whole offer on the cold tail and skips the
    warm head; full-head coverage must still project every valid head row.
    Works for an aligned head, a mid-block prompt, and a short tail.
    """
    split = origin // B * B + B
    tail = min(split + B, end)
    if origin > split - B:  # mid-block prompt: the valid head is [origin, split)
        return dict(origin=origin, end=end, resident=range(origin, split),
                    resolved=(*range(origin, origin + 2), *range(origin + 12, split)),
                    masks=(*range(origin + 2, origin + 12), *range(split, tail)))
    return dict(origin=origin, end=end, resident=range(origin, split),
                resolved=(*range(origin, origin + 8), *range(origin + 20, split)),
                masks=(*range(origin + 8, origin + 20), *range(split, tail)))


STATES = (dict(origin=32, end=131), dict(origin=40, end=131), dict(origin=32, end=80))


def plan_fields(result):
    plan, sampled, phase, cap = result
    return dict(rows=tuple(plan.rows), dirty=tuple(plan.dirty), cold=tuple(plan.cold),
                mask=tuple(plan.mask), prefix=tuple(plan.prefix), drain=plan.drain_only,
                sampled=tuple(sampled), phase=phase, cap=cap)


def record_plan(e):
    return {k: e[k] for k in ("rows", "dirty", "cold", "mask", "prefix", "sampled",
                              "drain_only", "phase")}


def audit(r, e, q, head_before, end_before):
    """Strict-cap and ledger audit shared by every switch tier."""
    split_before = head_before + B
    rows, sampled = set(e["rows"]), set(e["sampled"])
    before, written = set(e["dirty_before"]), set(e["written_absolute"])
    assert e["offer"] == q
    assert e["q"] == len(rows) <= q
    assert e["rows"] == sorted(rows)
    assert len(e["dirty"]) <= q and len(e["cold"]) <= q
    assert sampled <= rows and written <= sampled
    assert set(e["dirty"]) == set(sorted(before)[:q])
    assert set(e["dirty_after"]) == (before - rows) | written
    assert set(e["tail_read_only"]) <= rows - sampled
    assert set(e["cold"]) <= rows and set(e["mask"]) <= rows and set(e["prefix"]) <= rows
    assert r.bank.last_end <= r.end
    if q == 2 * B:
        head = [p for p in e["rows"] if p < split_before]
        assert head == list(range(max(head_before, r.origin), min(split_before, end_before)))
        assert e["coverage"] == "head" and e["head_full"] is True
    else:
        assert e["head_full"] is False
    if e["parked_tail"]:
        assert e["parked_tail"] == sum(1 for p in e["rows"] if split_before <= p < end_before)


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("state", STATES, ids=["aligned", "midblock", "shorttail"])
def test_switched_head_is_plan_equivalent_to_the_fixed_arm_at_constant_c64(gate, state):
    """Same coverage + gate: the switch plan equals the fixed plan at q=64."""
    fixed = planner(cross=gate, coverage="head", **head_state(**state))
    switch = planner(switchable=True, top_gate=gate, coverage="head", **head_state(**state))
    assert plan_fields(switch._switch_plan(64)) == plan_fields(fixed._cross_plan(64))
    plan, sampled, phase, cap = switch._switch_plan(64)
    split = state["origin"] // B * B + B
    assert [p for p in plan.rows if p < split] == list(range(max(state["origin"], state["origin"] // B * B), split))
    assert len(plan.rows) <= 64 and cap == 64
    # The paired sparse arm is genuinely different on this state, so the
    # equivalence above is not vacuous.
    sparse = planner(switchable=True, top_gate=gate, coverage="sparse", **head_state(**state))
    assert plan_fields(sparse._switch_plan(64)) != plan_fields((plan, sampled, phase, cap))


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("accept", [False, True])
@pytest.mark.parametrize("state", STATES, ids=["aligned", "midblock", "shorttail"])
def test_switched_head_matches_the_fixed_arm_over_repeated_c64_steps(gate, accept, state):
    fixed = planner(cross=gate, coverage="head", accept=accept, **head_state(**state))
    switch = planner(switchable=True, top_gate=gate, coverage="head", accept=accept, **head_state(**state))
    for _ in range(4):
        if fixed.status != "ready" or switch.status != "ready":
            break
        f, s = fixed.step(64), switch.step(64)
        assert record_plan(f) == record_plan(s)
        assert f["dirty_after"] == s["dirty_after"]
        assert f["written_absolute"] == s["written_absolute"]
        assert torch.equal(fixed.x, switch.x)
    assert fixed.status == switch.status
    assert fixed.tokens == switch.tokens


@pytest.mark.parametrize("a", TIERS)
@pytest.mark.parametrize("b", TIERS)
def test_switched_head_5x5_matrix_keeps_strict_cap_kv_and_dirty(a, b):
    """5x5 tier transitions: every offer keeps cap, KV, and the global ledger."""
    r = planner(switchable=True, top_gate="eager", coverage="head", lower_borrow=True)
    for q in (a, a, b, b, a, b):
        init, end, clock = set(r.bank.initialized), r.bank.last_end, r.sample_clock
        head_before = r.head
        e = r.step(q)
        audit(r, e, q, head_before, r.end)
        assert set(r.bank.initialized) >= init
        assert r.bank.last_end >= end
        assert e["sample_clock"] - clock == int(bool(e["sampled"]))


@pytest.mark.parametrize("q", TIERS)
def test_switched_head_respects_a_full_debt_offer(q):
    """Debt >= offer means a pure drain of exactly q dirty rows, never a sample."""
    debt = set(range(32, 32 + q + 10))
    r = planner(switchable=True, top_gate="eager", coverage="head",
                resident=debt, resolved=debt, dirty=debt)
    head_before = r.head
    e = r.step(q)
    audit(r, e, q, head_before, r.end)
    assert e["drain_only"] and not e["sampled"] and not e["written_absolute"]
    assert e["rows"] == sorted(debt)[:q]
    assert set(e["dirty_after"]) == debt - set(e["rows"])


@pytest.mark.parametrize("q", [4, 8, 16, 32])
def test_downshift_retains_parked_tail_dirty_and_the_next_c64_refreshes_it(q):
    """A 64 offer that dirtied the tail may be followed by any lower offer."""
    r = planner(switchable=True, top_gate="eager", coverage="head", accept=True)
    first = r.step(64)
    assert first["tail_q"] > 0
    parked = {p for p in first["dirty_after"] if p >= 64}
    assert parked
    head_before = r.head
    e = r.step(q)
    audit(r, e, q, head_before, r.end)
    assert parked - set(e["rows"]) <= set(e["dirty_after"])
    assert set(r.dirty) == set(e["dirty_after"])
    assert len(e["rows"]) <= q
    head_before = r.head
    again = r.step(64)
    audit(r, again, 64, head_before, r.end)
    assert parked <= set(again["rows"]) | set(again["dirty_after"])
    assert parked - set(again["rows"]) <= set(again["dirty_after"])


def test_downshift_keeps_parked_dirty_when_an_eos_closes_the_head():
    """Tail dirty stays refreshable read-only after the tail loses eligibility."""
    r = planner(switchable=True, top_gate="eager", coverage="head", accept=True)
    first = r.step(64)
    parked = {p for p in first["dirty_after"] if p >= 64}
    assert parked
    with torch.inference_mode():
        r.x[0, 40] = 0          # keep a head MASK so the head is not retired
        r.x[0, 63] = 9          # EOS in the head closes the tail
    head_before = r.head
    e = r.step(8)
    audit(r, e, 8, head_before, r.end)
    assert parked - set(e["rows"]) <= set(e["dirty_after"])
    head_before = r.head
    up = r.step(64)
    audit(r, up, 64, head_before, r.end)
    assert parked <= set(up["rows"]) | set(up["dirty_after"])
    assert not up["tail_opened"] and up["tail_q"] > 0


def test_switched_head_covers_a_mid_block_prompt_and_a_short_tail():
    r = planner(switchable=True, top_gate="eager", coverage="head", origin=58, end=80)
    e = r.step(64)
    assert [p for p in e["rows"] if p < 64] == list(range(58, 64))
    assert set(e["rows"]) <= set(range(58, 80))
    assert e["q"] <= 64 and e["head_full"] is True


def test_switched_head_default_coverage_is_sparse_and_unchanged():
    """The new combination is opt-in; old switch defaults keep their old plan."""
    sig = inspect.signature(LLaDABudgetRequest.__init__)
    assert sig.parameters["coverage"].default == "sparse"
    assert sig.parameters["top_gate"].default == "quarter"
    assert sig.parameters["lower_borrow"].default is False
    default = planner(switchable=True, top_gate="eager", **head_state())
    fixed = planner(cross="eager", coverage="sparse", **head_state())
    assert plan_fields(default._switch_plan(64)) == plan_fields(fixed._cross_plan(64))
    default.records.clear()


def test_switched_head_registration_boundaries():
    from test_llada_budget_request import Measured

    def make(**kw):
        return LLaDABudgetRequest(Measured(), torch.ones((1, 32), dtype=torch.long),
                                  mask_id=0, max_new_tokens=69, **kw)

    # Accepted opt-in combinations.
    for gate in GATES:
        make(switchable=True, top_gate=gate, coverage="head")
        make(cross=gate, coverage="head")
    make(switchable=True, top_gate="eager", coverage="head", lower_borrow=True)
    make(switchable=True, top_gate="eager", coverage="sparse")
    # Rejected: full head still needs a two-block registration.
    for kw in (dict(switchable=True, top_gate="eager", coverage="head", cross="eager"),
               dict(switchable=True, top_gate="eager", coverage="head", ablation="00"),
               dict(switchable=True, top_gate="eager", coverage="head", tail_rule="defer_tail"),
               dict(switchable=True, top_gate="sideways", coverage="head"),
               dict(coverage="head"),
               dict(cross="head", coverage="head"),
               dict(cross="eager", coverage="sideways"),
               dict(cross="eager", coverage="head", ablation="00")):
        with pytest.raises(ValueError):
            make(**kw)

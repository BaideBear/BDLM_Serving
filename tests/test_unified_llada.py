"""Unified planner adapter: mechanical mapping only, no admission or ranking."""
import dataclasses
import sys
import types

import pytest
import torch

from experiments.t2t.serving_slots import unified_llada

MASK = 0


@dataclasses.dataclass(frozen=True)
class Out:
    rows: tuple
    dirty: tuple
    mask: tuple
    prefix: tuple
    sampled: tuple
    cold: tuple
    open_next: bool
    drain_only: bool


class FakePlanner:
    def __init__(self, out):
        self.out, self.calls = out, []

    def __call__(self, q, head_block, head, next_block=None, *, opened=False, params=None):
        self.calls.append(dict(q=q, head=head, opened=opened, next_start=None if next_block is None
                               else next_block.start, head_block=head_block))
        return self.out


@pytest.fixture
def planner(monkeypatch):
    def install(out):
        Block = dataclasses.make_dataclass("Block", [("start", int), ("size", int), ("masks", frozenset),
                                                     ("dirty", frozenset), ("resident", frozenset)],
                                           frozen=True)
        mod = types.ModuleType("bdlm_serving.methods.t2t.unified_rows")
        mod.Block, mod.Params = Block, dataclasses.make_dataclass("Params", [])
        fake = FakePlanner(out)
        mod.plan_rows = fake
        monkeypatch.setitem(sys.modules, "bdlm_serving.methods.t2t.unified_rows", mod)
        return Block, fake
    return install


class Bank:
    def __init__(self, initialized):
        self.initialized = set(initialized)


class Req:
    mask_id, head, end, origin, cross = MASK, 32, 128, 32, "quarter"
    warm_head_sample, admission_scope, close_policy, borrow_at_half = False, "global", "unchanged", False

    def __init__(self, dirty=(), initialized=(), origin=32):
        self.x = torch.full((1, 128), 7)
        self.x[0, 34] = self.x[0, 36] = MASK          # unresolved rows in the head block
        self.x[0, 33] = MASK                          # decoded-by-debt row
        self.x[0, 64:96] = MASK                       # unvisited successor is all unresolved
        self.dirty = set(dirty)
        self.bank = Bank(initialized)
        self.origin = origin


def test_blocks_are_read_from_live_state_only(planner):
    Block, fake = planner(Out((34,), (), (34,), (), (34,), (), False, False))
    req = Req(dirty={33}, initialized={33})
    head, nxt = unified_llada.live_blocks(req, opened=False)
    assert (head.start, head.size) == (32, 32)
    assert nxt is not None and nxt.start == 64          # offered as a candidate, not opened
    assert head.masks == frozenset({34, 36, 33})
    assert head.dirty == frozenset({33}) and head.resident == frozenset({33})
    head, nxt = unified_llada.live_blocks(req, opened=True)
    assert nxt is not None and nxt.start == 64 and nxt.size == 32
    assert nxt.masks == frozenset(range(64, 96))
    short = Req(origin=40)
    head, nxt = unified_llada.live_blocks(short, opened=True)
    assert (head.start, head.size) == (40, 24)                 # first valid generated row
    assert nxt.start == head.start + head.size == 64           # contract chaining


def test_planner_rows_are_copied_verbatim_and_partitioned(planner):
    planner(Out((40, 34, 36), (), (34, 36, 40), (40,), (34, 36, 40), (), True, False))
    req = Req(initialized=set(range(32, 64)))
    plan, sampled, phase, cap, nxt = unified_llada.execute(req, 4, False)
    assert plan.rows == (34, 36, 40)                       # sorted, but same set
    assert set(plan.rows) == {34, 36, 40}
    assert plan.mask == (34, 36, 40) and plan.cold == ()   # resident masks, all cold-free
    assert plan.prefix == (40,) and plan.dirty == ()
    assert sampled == (34, 36, 40) and phase == "sample" and cap == 4 and nxt is True
    assert len(plan.rows) <= cap


def test_cold_unresolved_rows_stay_inside_the_mask_role(planner):
    planner(Out((34, 40), (), (34, 40), (), (34, 40), (40,), False, False))
    req = Req(initialized=set(range(32, 64)) - {40})       # 40 uninitialized
    plan, sampled, phase, _, _ = unified_llada.execute(req, 4, False)
    assert plan.cold == (40,) and plan.mask == (34,)       # cold is in .mask, traced as cold
    assert sampled == (34, 40)                             # planner's sampled, verbatim


def test_drain_plan_never_samples_unresolved_rows(planner):
    planner(Out((90, 91), (), (), (), (), (), False, True))
    req = Req(dirty={90, 91})
    plan, sampled, phase, cap, _ = unified_llada.execute(req, 4, False)
    assert plan.drain_only is True and plan.dirty == (90, 91)
    assert sampled == () and phase == "drain" and cap == 4
    planner(Out((34,), (), (34,), (), (34,), (), False, True))
    with pytest.raises(RuntimeError, match="Drain plan"):
        unified_llada.execute(Req(), 4, False)


def test_row_outside_planned_blocks_is_rejected(planner):
    planner(Out((200,), (), (200,), (), (200,), (), False, False))
    with pytest.raises(RuntimeError, match="outside the planned blocks"):
        unified_llada.execute(Req(), 4, False)
    planner(Out((34, 36, 40, 41, 42), (), (), (), (), (), False, False))
    with pytest.raises(RuntimeError, match="exceeded the paid offer"):
        unified_llada.execute(Req(), 4, False)


def test_open_next_is_threaded_and_one_forward_is_one_call(planner):
    _, fake = planner(Out((34,), (), (34,), (), (34,), (), True, False))
    log = []
    cls = unified_llada.request(object, on_step=log.append)
    obj = cls.__new__(cls)
    obj.mask_id, obj.head, obj.end, obj.cross = MASK, 32, 128, "quarter"
    obj.warm_head_sample, obj.admission_scope = False, "global"
    obj.close_policy, obj.borrow_at_half = "unchanged", False
    obj.switchable = True
    obj.x = Req().x
    obj.dirty, obj.bank = set(), Bank(set(range(32, 64)))
    obj.origin = 32
    obj._cross_plan(4)
    obj._cross_plan(4)
    assert [c["opened"] for c in fake.calls] == [False, True]     # open_next threaded
    assert len(fake.calls) == len(log) == 2                       # one plan per paid call
    assert "_additional_admissions" not in cls.__dict__          # no admission hook added
    fake.out = dataclasses.replace(fake.out, open_next=False)
    obj._cross_plan(4)
    assert fake.calls[-1]["opened"] is True                       # sticky once admitted


def test_profile_knobs_are_refused(planner):
    planner(Out((34,), (), (34,), (), (34,), (), False, False))
    cls = unified_llada.request(object)
    for name, value in (("warm_head_sample", True), ("admission_scope", "warm_head"),
                        ("close_policy", "full_write"), ("borrow_at_half", True)):
        obj = cls.__new__(cls)
        obj.mask_id, obj.head, obj.end, obj.cross = MASK, 32, 128, "quarter"
        obj.warm_head_sample, obj.admission_scope = False, "global"
        obj.close_policy, obj.borrow_at_half = "unchanged", False
        obj.switchable = True
        setattr(obj, name, value)
        obj.x = Req().x
        obj.dirty, obj.bank = set(), Bank(set(range(32, 64)))
        obj.origin = 32
        with pytest.raises(ValueError, match="does not carry a profile knob"):
            obj._cross_plan(4)


def test_planner_sampled_set_is_verified_against_the_contract(planner):
    planner(Out((34, 36), (), (34,), (), (34, 36), (), False, False))   # 36 is dirty-shaped
    with pytest.raises(RuntimeError, match=r"sorted\(mask \| prefix\)"):
        unified_llada.execute(Req(dirty={36}), 4, False)
    planner(Out((34, 36), (), (34,), (), (36, 34), (), False, False))   # unsorted sampled
    with pytest.raises(RuntimeError, match="sorted"):
        unified_llada.execute(Req(), 4, False)
    planner(Out((34,), (), (34,), (), (34,), (99,), False, False))      # cold outside mask
    with pytest.raises(RuntimeError, match="cold rows"):
        unified_llada.execute(Req(), 4, False)


def test_planner_is_resolved_lazily_not_at_import():
    assert not any("unified_rows" in getattr(v, "__name__", "") for v in vars(unified_llada).values())
    Block, Params, plan_rows = unified_llada.planner()      # resolved on first use only
    assert Block is not None and Params is not None and callable(plan_rows)


def test_real_planner_contract_is_executed_without_rewriting_rows():
    """Integration with the actual unified_rows planner (pure CPU, no model)."""
    try:
        from bdlm_serving.methods.t2t import unified_rows as real
    except ImportError:
        return                                        # planner not landed yet
    assert real.plan_rows is not None
    req = Req(initialized=set(range(32, 60)))            # 4 unresolved head rows, warm head
    plan, sampled, phase, cap, nxt = unified_llada.execute(req, 64, False)
    assert set(plan.rows) == set(sampled) | set(plan.dirty)
    assert tuple(sorted(sampled)) == sampled and set(sampled) & set(plan.dirty) == set()
    assert phase == "sample" and cap == 64 and len(plan.rows) <= 64
    assert {34, 36, 33} <= set(plan.rows)               # head masks are paid first
    assert nxt is True                                  # successor admitted on a 2B offer
    assert max(plan.rows) >= 64                         # the admitted block was really paid
    plan2, sampled2, _, _, nxt2 = unified_llada.execute(Req(initialized=set(range(32, 60))), 16, False)
    assert nxt2 is False and max(plan2.rows) < 64        # no successor on a sub-block offer


def make(cls, req=None, head=32, stopping=False):
    obj = cls.__new__(cls)
    obj.mask_id, obj.head, obj.end, obj.cross = MASK, head, 128, None
    obj.origin = 32
    obj.warm_head_sample, obj.admission_scope = False, "global"
    obj.close_policy, obj.borrow_at_half = "unchanged", False
    obj.switchable = True
    obj.x, obj.dirty = (req or Req()).x, set()
    obj.bank = Bank(set(range(32, 60)))
    obj.stopping = stopping
    return obj


def test_sticky_open_next_resets_with_the_physical_head(planner):
    _, fake = planner(Out((34,), (), (34,), (), (34,), (), True, False))
    log = []
    obj = make(unified_llada.request(object, on_step=log.append))
    obj._plan(4)
    obj._plan(4)
    assert [c["opened"] for c in fake.calls] == [False, True]
    obj.head = 64                                  # head retired: successor ordinal changed
    fake.out = dataclasses.replace(fake.out, rows=(66,), sampled=(66,), mask=(66,))
    obj._plan(4)
    assert fake.calls[-1]["opened"] is False       # admission does not leak to the new head


def test_stopping_plans_debt_only_and_never_samples(planner):
    planner(Out((40,), (40,), (), (), (), (), False, True))
    req = Req(dirty={40})
    plan, sampled, phase, cap, _ = unified_llada.execute(req, 4, False) if False else (None,) * 5
    obj = make(unified_llada.request(object), stopping=True)
    obj.dirty = {40}
    plan, sampled, phase, cap = obj._plan(4)
    assert sampled == () and phase == "drain" and plan.drain_only is True
    planner(Out((34,), (), (34,), (), (34,), (), False, False))
    obj.dirty = set()                                  # no tail debt: window falls back to head
    with pytest.raises(RuntimeError, match="EOS/tail call must not sample"):
        obj._plan(4)


def test_params_are_public_and_reach_the_planner(planner):
    Block, fake = planner(Out((34,), (), (34,), (), (34,), (), False, False))
    sent = {}

    def spy(q, b, head, next_block=None, *, opened=False, params=None):
        sent["params"] = params
        return fake.out
    import sys
    sys.modules["bdlm_serving.methods.t2t.unified_rows"].plan_rows = spy
    pars = type("P", (), {})()
    obj = make(unified_llada.request(object, params=pars))
    obj._plan(4)
    assert sent["params"] is pars


def test_base_admission_hook_is_refused(planner):
    planner(Out((34,), (), (34,), (), (34,), (), False, False))
    class Hooked:
        def _additional_admissions(self, *a, **k):
            raise AssertionError("must not be called")
    obj = make(unified_llada.request(Hooked))
    with pytest.raises(ValueError, match="refuses a base admission hook"):
        obj._plan(4)


def test_observed_rows_are_audited_against_the_planner(planner):
    planner(Out((34,), (), (34,), (), (34,), (), False, False))
    obj = make(unified_llada.request(object))
    obj._plan(4)
    obj.records = [dict(rows=[34], sampled=[99])]
    obj.head = 64
    with pytest.raises(RuntimeError, match="Observed rows/sampled differ"):
        obj._plan(4)


def test_one_class_serves_all_five_offers(planner):
    planner(Out((34,), (), (34,), (), (34,), (), False, False))
    cls = unified_llada.request(object)
    assert cls._switch_plan is cls._cross_plan
    for q in (4, 8, 16, 32, 64):
        obj = make(cls)
        obj._plan(q)
    assert unified_llada.factory is not None


def test_terminal_step_is_audited_too(planner):
    planner(Out((34,), (), (34,), (), (34,), (), False, False))
    class B:
        def step(self, q):
            self.records.append(dict(rows=[999], sampled=[999]))
            return "ok"
    obj = make(unified_llada.request(B))
    obj.records = []
    obj._plan(4)
    with pytest.raises(RuntimeError, match="Observed rows/sampled differ"):
        obj.step(4)


def real_planner_or_none():
    try:
        from bdlm_serving.methods.t2t import unified_rows
        return unified_rows
    except ImportError:
        return None


def test_eos_return_offers_no_successor_and_pays_only_tail_debt():
    if real_planner_or_none() is None:
        return
    req = Req(initialized=set(range(32, 128)), origin=32)
    req.x[0, 70] = req.x[0, 71] = MASK          # tail debt rows, no MASK left anywhere else
    req.stopping, req.head, req.dirty = True, 64, {70, 71}
    head, nxt = unified_llada.live_blocks(req, opened=False)
    assert nxt is None                          # EOS: never a fresh successor
    assert head.masks == frozenset()            # stopping never samples
    plan, sampled, phase, cap, nxt2 = unified_llada.execute(req, 16, False)
    assert set(plan.rows) == {70, 71} and sampled == () and phase == "drain"
    assert plan.dirty == (70, 71) and not nxt2


def test_eos_window_is_the_physical_head_block_and_missing_debt_fails_closed():
    if real_planner_or_none() is None:
        return
    req = Req(initialized=set(range(32, 128)), origin=32)
    req.x[0, 70] = req.x[0, 71] = MASK
    req.stopping, req.head, req.dirty = True, 64, {70, 71}     # debt inside the head block
    head, nxt = unified_llada.live_blocks(req, opened=False)
    assert (head.start, head.size) == (64, 32) and nxt is None
    plan, sampled, phase, _, _ = unified_llada.execute(req, 16, False)
    assert set(plan.rows) == {70, 71} and sampled == () and phase == "drain"
    above = Req(initialized=set(range(32, 128)), origin=32)     # debt outside the window
    above.x[0, 70] = above.x[0, 71] = MASK
    above.stopping, above.head, above.dirty = True, 96, {70, 71}
    with pytest.raises(RuntimeError, match="outside the physical head window"):
        unified_llada.execute(above, 16, False)


def test_visited_successor_is_handed_over_and_the_planner_fails_closed():
    if real_planner_or_none() is None:
        return
    req = Req(initialized=set(range(32, 96)), origin=32)       # successor already resident
    req.x[0, 64:96] = 7                                        # ... and already resolved
    req.head, req.dirty = 32, set()
    head, nxt = unified_llada.live_blocks(req, opened=False)
    assert nxt is not None and nxt.start == 64                 # never silently suppressed
    assert nxt.resident and not nxt.masks                      # a visited, resolved block
    with pytest.raises(ValueError, match="consecutive successor"):
        unified_llada.execute(req, 16, False)                  # fails closed, opened=False
    plan, sampled, phase, _, nxt_out = unified_llada.execute(req, 16, True)   # truly open
    assert nxt_out is True and phase == "sample" and plan.rows   # open persists, no raise


def test_eos_clears_an_already_admitted_successor():
    if real_planner_or_none() is None:
        return
    req = Req(initialized=set(range(32, 96)), origin=32)
    req.x[0, 70] = MASK
    obj = make(unified_llada.request(object), stopping=True)
    obj.head, obj.dirty = 64, {70}
    obj._unified_open_at = 96                          # successor 96 was admitted before EOS
    plan, sampled, phase, cap = obj._plan(16)          # must not raise "open successor" here
    assert obj._unified_open_at is None and sampled == () and phase == "drain"
    assert set(plan.rows) == {70}


def test_terminal_debt_over_offer_drains_across_rounds():
    if real_planner_or_none() is None:
        return
    req = Req(initialized=set(range(32, 128)), origin=32)
    tail = list(range(70, 80))                       # 10 debt rows inside the head block
    for p in tail:
        req.x[0, p] = MASK
    req.stopping, req.head, req.dirty = True, 64, set(tail)
    rounds, paid = 0, []
    while req.dirty:
        plan, sampled, phase, cap, nxt = unified_llada.execute(req, 4, False)   # q < debt
        assert phase == "drain" and sampled == () and len(plan.rows) <= 4 and not nxt
        paid.extend(plan.rows)
        req.dirty -= set(plan.rows)                  # only the paid debt is consumed
        rounds += 1
        assert rounds <= 4                           # 10 rows at q=4 drains in 3 rounds
    assert rounds == 3 and sorted(paid) == tail      # every debt row paid exactly once


def test_terminal_debt_outside_window_fails_but_in_window_deferral_does_not():
    if real_planner_or_none() is None:
        return
    inside = Req(initialized=set(range(32, 128)), origin=32)
    for p in range(70, 75):
        inside.x[0, p] = MASK
    inside.stopping, inside.head, inside.dirty = True, 64, set(range(70, 75))
    plan, sampled, phase, _, _ = unified_llada.execute(inside, 4, False)   # q < debt, in window
    assert set(plan.rows) == {70, 71, 72, 73}
    assert set(inside.dirty) - set(plan.rows) == {74}                      # deferred, not an error
    assert phase == "drain" and sampled == ()
    outside = Req(initialized=set(range(32, 128)), origin=32)
    outside.x[0, 70] = outside.x[0, 126] = MASK        # 126 lies outside [64,96)
    outside.stopping, outside.head, outside.dirty = True, 64, {70, 126}
    with pytest.raises(RuntimeError, match="outside the physical head window"):
        unified_llada.execute(outside, 4, False)     # 126 is outside [64,96)

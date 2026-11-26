from itertools import product
import random

import pytest

from bdlm_serving.methods.t2t.physical_prefix import MATCHED_MODES, SLACK_MODES, plan_prefix
from bdlm_serving.methods.t2t.protocol import Arm


ZERO, RANDOM, RIGHT = MATCHED_MODES


@pytest.mark.parametrize("selector", ["right", "random"])
def test_gemma_paired_donors_same_recipient_and_physical_width(selector):
    def pp(state, q, k, donor):
        return plan(state, {0}, q=q, k=k, mode=f"hard_prefix_{donor}_{selector}")

    for state in product((True, False), repeat=7):
        if not state[0]:
            continue
        for q in (3, 5):
            k = q - 1
            old, mask, deploy = [pp(state, q, k, d) for d in
                                 ("paired_backfill", "paired_mask", "backfill")]
            assert old.added_rows == mask.added_rows
            assert len(old.positions) == len(mask.positions) == len(deploy.positions) == len(old.zero_rows)
            assert set(old.evicted_rows) <= set(old.zero_backfill_rows)
            assert set(deploy.evicted_rows) <= set(old.zero_backfill_rows)
            assert set(mask.evicted_rows) <= set(old.zero_mask_rows[1:])
            assert len(old.added_rows) == min(k, len(old.eligible_prefix_rows),
                                              len(old.zero_backfill_rows), len(old.zero_mask_rows[1:]))
            assert len(deploy.added_rows) == min(k, len(old.eligible_prefix_rows), len(old.zero_backfill_rows))
    state = (True, True, True, False, False, True, True)
    old, mask = [pp(state, 5, 2, d) for d in ("paired_backfill", "paired_mask")]
    assert len(old.added_rows) == 1 and (old.evicted_rows, mask.evicted_rows) == ((6,), (4,))
    with pytest.raises(ValueError, match="Gemma"):
        Arm.parse(f"16:0:12:hard_prefix_paired_mask_{selector}", 32)


def plan(accepted, fresh=(), q=5, k=0, mode=ZERO, *, editable=None,
         seed=29, block_start=32, clock=3, last_query=None):
    rows = range(len(accepted)) if editable is None else editable
    return plan_prefix(accepted, rows, fresh, q, k, mode, seed=seed,
                       block_start=block_start, clock=clock,
                       last_query=[0] * len(accepted) if last_query is None else last_query)


def test_strict_physical_prefix_stops_at_first_hole_before_removing_fresh():
    state = [True, True, True, True, False, True, True, False]
    p = plan(state, {2, 5}, k=2, mode=RIGHT, editable=range(2, 8))
    assert p.prefix_end == 4 and p.eligible_prefix_rows == (3,)
    assert p.zero_rows == (2, 4, 5, 6, 7)
    assert p.added_rows == (3,) and p.evicted_rows == (6,)
    assert p.positions == (2, 3, 4, 5, 7)

    # A current acceptance hole may move the frontier back: no high-water mark.
    reopened = plan([True, False, True, True, False, True, True, False], {2, 5})
    assert reopened.prefix_end == 1 and reopened.eligible_prefix_rows == (0,)
    assert 0 not in reopened.positions and 1 in reopened.positions


def test_zero_underfills_instead_of_backfilling_retired_prefix_rows():
    state = [True] * 6 + [False, True]
    zero = plan(state, {3})
    right = plan(state, {3}, k=4, mode=RIGHT)
    assert zero.positions == zero.zero_rows == right.zero_rows == (3, 6, 7)
    assert right.positions == (3, 5, 6)
    assert right.added_rows == (5,) and right.evicted_rows == (7,)
    assert right.audit()["underfill"] == 2
    assert plan([True] * 8).positions == ()


def test_oldest_uses_last_physical_query_then_rightmost_tie():
    tokens = [True] * 6 + [False, True]
    times = [0, 2, 7, 2, 3, 5, 1, 1]
    p = plan(tokens, {2}, q=5, k=2, mode="hard_prefix_slack_oldest",
             editable=range(1, 8), last_query=times)
    assert p.added_rows == (1, 3) and p.evicted_rows == ()
    assert p.audit()["eligible_last_query"] == [2, 2, 3, 5]
    assert plan(tokens, {2}, q=5, k=1, mode="hard_prefix_slack_oldest",
                editable=range(1, 8), last_query=times).added_rows == (3,)
    with pytest.raises(ValueError, match="history"):
        plan(tokens, k=2, mode="hard_prefix_slack_oldest", last_query=[-1] * 8)


def test_first_mask_is_protected_and_all_extra_masks_precede_old_backfill():
    state = [True] * 3 + [False, True, False, True, False]
    p = plan(state, k=3, mode=RIGHT)
    assert p.zero_mask_rows == (3, 5, 7) and p.zero_backfill_rows == (4, 6)
    assert p.added_rows == (0, 1, 2) and p.evicted_rows == (4, 6, 7)
    assert p.positions == (0, 1, 2, 3, 5)
    overflow = plan(state, {0, 1, 2, 4, 6}, q=4, k=3, mode=RIGHT)
    assert overflow.positions == overflow.zero_rows == (0, 1, 2, 4, 6)
    assert not overflow.added_rows and overflow.audit()["overflow"] == 1


def test_random_is_private_reproducible_and_nested_at_the_same_state():
    state = [True] * 12 + [False] * 12
    before = random.getstate()
    r = plan(state, q=8, k=3, mode=RANDOM)
    assert r == plan(state, q=8, k=3, mode=RANDOM)
    assert set(r.added_rows) < set(plan(state, q=8, k=6, mode=RANDOM).added_rows)
    for attr, values in (("seed", range(10)), ("block_start", range(0, 320, 32)), ("clock", range(10))):
        assert len({plan(state, q=8, k=3, mode=RANDOM, **{attr: x}).added_rows for x in values}) > 1
    assert before == random.getstate()


def test_all_small_acceptance_and_fresh_states_match_the_same_state_zero():
    # 0=MASK, 1=ordinary or retired accepted, 2=fresh accepted.
    for state in product(range(3), repeat=5):
        accepted, fresh = [bool(x) for x in state], {i for i, x in enumerate(state) if x == 2}
        for q in range(1, 6):
            zero = plan(accepted, fresh, q=q)
            assert set(zero.positions).isdisjoint(zero.eligible_prefix_rows)
            for k in range(1, q):
                for mode in (RIGHT, RANDOM):
                    p = plan(accepted, fresh, q=q, k=k, mode=mode)
                    rows, reference = set(p.positions), set(zero.positions)
                    assert p.zero_rows == zero.positions and len(rows) == len(reference)
                    assert rows - reference == set(p.added_rows) <= set(zero.eligible_prefix_rows)
                    assert reference - rows == set(p.evicted_rows)
                    assert fresh <= rows and len(p.added_rows) <= k
                    assert zero.first_mask is None or zero.first_mask in rows


@pytest.mark.parametrize("width,q", [(32, 4), (32, 8), (32, 16), (256, 32), (256, 64), (256, 128)])
def test_cli_arm_parsing_accepts_sparse_and_block_sized_capacity(width, q):
    assert Arm.parse(f"{q}:0:{q}:{ZERO}", width).edit == 0
    for mode in (RANDOM, RIGHT):
        assert Arm.parse(f"{q}:0:{q-2}:{mode}", width).edit == 2
        assert Arm.parse(f"{width}:0:{width-2}:{mode}", width).edit == 2
        for form in (f"{width+1}:0:{width-1}:{mode}", f"{q}:1:{q-2}:{mode}",
                     f"{q}:0:{q}:{mode}", f"{q}:0:0:{mode}"):
            with pytest.raises(ValueError):
                Arm.parse(form, width)
    assert Arm.parse(f"{width}:0:{width}:{ZERO}", width).edit == 0
    with pytest.raises(ValueError):
        Arm.parse(f"{q}:0:{q-1}:{ZERO}", width)


def test_b32_q64_alias_keeps_k_and_policy():
    for mode, k in ((ZERO, 0), (RIGHT, 2), (RANDOM, 31), (SLACK_MODES[0], 2)):
        arm = Arm.parse(f"64:0:{64-k}:{mode}", 32)
        effective = Arm.parse(f"32:0:{32-k}:{mode}", 32)
        assert arm.prefix_executable(32) == effective
        assert arm.edit == effective.edit == k
    with pytest.raises(ValueError):
        Arm.parse(f"64:0:32:{RIGHT}", 32)


def test_reject_invalid_editable_or_fresh_positions():
    for fresh, editable in (({4}, range(4)), ({0}, range(8)), ({5}, range(6))):
        with pytest.raises(ValueError):
            plan([True, True, True, True, True, False, False], fresh, editable=editable)


@pytest.mark.parametrize("mode", SLACK_MODES)
def test_slack_only_preserves_all_zero_rows_and_charges_every_addition(mode):
    for state in product(range(3), repeat=5):
        accepted = [bool(x) for x in state]
        fresh = {i for i, x in enumerate(state) if x == 2}
        for q in range(2, 6):
            zero = plan(accepted, fresh, q=q)
            for k in {1, q - 1}:
                p = plan(accepted, fresh, q=q, k=k, mode=mode)
                slots = max(0, q - len(zero.positions)) if zero.positions else 0
                n = min(k, len(zero.eligible_prefix_rows), slots)
                assert p.zero_rows == zero.positions and not p.evicted_rows
                assert set(p.positions) == set(p.zero_rows) | set(p.added_rows)
                assert set(p.added_rows) <= set(zero.eligible_prefix_rows)
                assert len(p.positions) == len(zero.positions) + n
                assert len(p.added_rows) == n
                if mode == "hard_prefix_slack_right" and n:
                    assert p.added_rows == zero.eligible_prefix_rows[-n:]

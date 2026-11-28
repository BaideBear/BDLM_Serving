from dataclasses import replace
from fractions import Fraction

import pytest

from bdlm_serving.methods.t2t.unified_rows import Block, Params, plan_rows


def block(start, size, mask, dirty=(), resident=()):
    return Block(start, size, *({start+i for i in rows} for rows in (mask, dirty, resident)))


@pytest.mark.parametrize("b", (32, 256))
def test_all_normalized_offers_including_two_blocks(b):
    h = block(7, b, range(b))
    s = block(7+b, b, range(b))
    for q in (b//8, b//4, b//2, b, 2*b):
        p = plan_rows(q, b, h, s)
        assert p.rows == p.mask == p.cold == p.sampled == tuple(range(7, 7+q))
        assert p.open_next == (q == 2*b)
        assert not p.prefix and not p.dirty


def test_downshift_drains_both_blocks_before_sampling_again():
    h = block(19, 8, [6, 7], [0, 1], range(8))
    s = block(27, 8, range(8), [0, 1, 2], range(5))
    p = plan_rows(4, 8, h, s, opened=True)
    assert p.dirty == p.rows == (19, 20, 27, 28)
    assert p.open_next and p.drain_only and not p.sampled
    p = plan_rows(8, 8, replace(h, dirty=frozenset()), replace(s, dirty={29}), opened=True)
    assert p.open_next and p.dirty == (29,) and len(p.rows) == 8
    assert set(p.rows) == set(p.dirty).union(p.sampled)
    assert p.cold == (32, 33, 34)
    assert not set(p.sampled).intersection(p.dirty)


def test_prefix_is_physical_contiguous_right_edge_excludes_dirty_and_cold():
    h = block(29, 16, [8, 10, 11, 12, 13, 14, 15], [6], {0, 1, 3, 4, 5, 6, 7, 9})
    p = plan_rows(8, 16, h, params=Params(prefix_share=Fraction(1, 2)))
    assert p.dirty == (35,) and p.prefix == (33, 34, 36)
    assert p.mask == (37, 39, 40, 41)
    assert len(p.rows) == len(set(p.rows)) == 8
    assert 38 not in p.prefix  # Resolved interior beyond the frontier.
    assert 36 not in plan_rows(8, 16, h, params=Params(prefix_share=0)).rows


def test_tail_gate_uses_valid_rows_actual_admission_and_stays_open():
    # Prompt occupies part of the physical head, so the remaining valid size is 4.
    tail = block(23, 8, range(8))
    h = block(19, 4, [2, 3], resident=range(4))
    assert not plan_rows(8, 8, h, tail).open_next
    h = replace(h, masks={22})
    assert not plan_rows(1, 8, h, tail).open_next
    assert plan_rows(2, 8, h, tail).rows == (22, 23)
    assert plan_rows(2, 8, h, tail).open_next
    assert plan_rows(1, 8, h, tail, opened=True).open_next


@pytest.mark.parametrize("shift", (0, 7, 256))
def test_plan_is_invariant_to_prompt_translation(shift):
    args = (8, 16)
    state = lambda k: (block(k, 16, [13, 15], [1], range(16)), block(k+16, 16, range(16)))
    p, s = plan_rows(*args, *state(0)), plan_rows(*args, *state(shift))
    for name in ("rows", "dirty", "mask", "prefix", "sampled", "cold"):
        assert tuple(i+shift for i in getattr(p, name)) == getattr(s, name)
    assert (p.open_next, p.drain_only) == (s.open_next, s.drain_only)


def test_closed_or_badly_translated_tail_cannot_hide_debt():
    h = block(3, 4, [1])
    with pytest.raises(ValueError):
        plan_rows(4, 8, h, block(7, 4, range(4), [0]))
    with pytest.raises(ValueError):
        plan_rows(4, 8, h, block(8, 4, range(4)))
    with pytest.raises(ValueError):
        plan_rows(4, 8, h, opened=True)

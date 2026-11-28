from fractions import Fraction
from random import Random

import pytest
from bdlm_serving.methods.t2t import unified_rows as r
from tools.shared_mask_order import instrument, alternative


def test_revisit_frontier_without_reassigning_debt_or_prefix():
    h = r.Block(30,8,{32,33,34,35,36,37},{31},{30,31,32,33})
    p = r.Params(Fraction(1,2),0)
    with instrument("cold_first") as a:
        old = r.plan_rows(4,8,h,params=p)
    with instrument("left_first") as b:
        new = r.plan_rows(4,8,h,params=p)
    assert old.mask == (34,35) and new.mask == (32,33)
    assert old.dirty == new.dirty == (31,) and old.prefix == new.prefix == (30,)
    assert old.cold == old.mask and not new.cold
    assert a["differing_calls"] == b["differing_calls"] == 1


@pytest.mark.parametrize("size", [32,256])
def test_per_block_quota_and_cache_invariants(size):
    rng = Random(710)
    for case in range(128):
        blocks = []
        for offset in (3,size+3):
            ids = list(range(offset,offset+size))
            def subset():
                return set(rng.sample(ids,rng.randrange(size+1)))
            blocks.append(r.Block(offset,size,subset(),subset(),subset()))
        q = rng.randrange(1,2*size+1)
        base = r.plan_rows(q,size,*blocks,opened=True)
        new = alternative(base,*blocks)
        assert new.dirty == base.dirty and new.prefix == base.prefix and new.open_next == base.open_next
        assert len(new.rows) == len(base.rows) <= q
        assert new == alternative(new,*blocks)
        for block in blocks:
            n = len(set(base.mask) & block.masks)
            assert sorted(set(new.mask) & block.masks) == sorted(block.masks - block.dirty)[:n]
        assert set(new.cold) == set(new.mask) - set.union(*(set(b.resident) for b in blocks))

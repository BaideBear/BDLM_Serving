"""Request protocol only: no torch, checkpoint, CPU inference or GPU required."""
from threading import Thread
from types import SimpleNamespace

import pytest

from bdlm_serving.methods.t2t.budget_family import Law
from bdlm_serving.methods.t2t.serving_profiles import StrictHandle


def make(over=False):
    bank, rng = object(), object()
    req = SimpleNamespace(origin=11, status="ready", tokens=[], bank=bank, rng=rng)
    seen = []
    def step(q):
        seen.append((q, req.law, req.bank, req.rng))
        enc = int(len(seen) == 2)
        if len(seen) == 3:
            req.status, req.tokens = "done", [9, 1]
        return dict(offer=q, q=q+int(over), encoder_F=enc, encoder_Q=enc*256)
    req.step = step
    return StrictHandle(req, (32,128,512), (Law(),Law(1,4),Law()), 256), seen


def test_cache_and_sampler_survive_law_switch_without_hidden_work():
    h, seen = make()
    tape = [h.step(q) for q in (32,128,32)]
    assert [e["prefix_share"] for e in tape] == [(1,8),(1,4),(1,8)]
    assert h.opened_prefill_q == 11 and h.result() == (9,1)
    assert sum(e["generation_F"] for e in tape) == 4
    assert sum(e["history_Q"] for e in tape) == 192
    assert sum(e["physical_generation_Q"] for e in tape) == 448
    assert seen[0][2:] == seen[1][2:] == seen[2][2:]
    h.close()
    assert h.status == "done" and h._request is None


def test_invalid_budget_fails_before_advancing_worker():
    h, seen = make()
    for x in (4, True, 32., None):
        with pytest.raises(ValueError):
            h.step(x)
    assert not seen and h.status == "ready"
    h.close()
    with pytest.raises(RuntimeError):
        h.step(32)


def test_corrupt_physical_receipt_is_terminal_without_retry():
    h, seen = make(over=True)
    with pytest.raises(RuntimeError, match="Paid work"):
        h.step(32)
    assert h.status == "failed" and h._request is None and len(seen) == 1


def test_wrong_lane_cannot_advance():
    h, seen = make()
    errors = []
    def other_lane():
        try:
            h.step(32)
        except RuntimeError as exc:
            errors.append(str(exc))
    t = Thread(target=other_lane)
    t.start()
    t.join()
    assert errors == ["Use the owning serial inference lane"] and not seen


def test_optional_full_head_uses_current_density_only_at_the_one_block_offer():
    req=SimpleNamespace(origin=12,status="ready",rows=SimpleNamespace(accepted=set()))
    flags=[]
    def step(q):
        flags.append(req.full_head_at_b)
        return dict(offer=q,q=q)
    req.step=step
    h=StrictHandle(req,(128,256,512),(Law(),)*3,256,full_head_min_unresolved=128)
    for q,accepted in ((128,0),(256,128),(256,129),(512,10),(256,127)):
        req.rows.accepted=set(range(accepted))
        h.step(q)
    assert flags==[False,True,False,False,True]

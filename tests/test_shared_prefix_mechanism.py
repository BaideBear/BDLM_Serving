from fractions import Fraction
from types import SimpleNamespace

import pytest

from bdlm_serving.methods.t2t import unified_rows as rows
from tools.shared_prefix_mechanism import ARMS, DOSE, check_triplet, instrument, observe_writes


def head():
    return rows.Block(32, 32, frozenset(range(45,64)), frozenset(), frozenset(range(32,50)))


def test_no_op_negative_control_and_hook_restoration():
    original = rows.plan_rows
    calls = []
    for arm in ARMS:
        with instrument(arm) as info:
            calls.append(rows.plan_rows(4,32,head(),params=DOSE))
            assert info["first"] is None
        assert rows.plan_rows is original
    assert calls[0] == calls[1] == calls[2]
    with pytest.raises(ValueError), instrument("reserve_idle"):
        rows.plan_rows(4,32,head(),params=rows.Params(Fraction(1,4),Fraction(1,4)))
    assert rows.plan_rows is original


def test_idle_omits_physical_prefix_without_refilling_mask():
    decisions, starts = {}, []
    for arm in ARMS:
        with instrument(arm) as info:
            rows.plan_rows(4,32,head(),params=DOSE)
            decisions[arm] = rows.plan_rows(8,32,head(),params=DOSE)
            starts.append(info["first"])
            assert (info["exposed_calls"],info["reserved_seats"],info["avoided_mask_rows"]) == (1,1,1)
    assert starts[0] == starts[1] == starts[2] and starts[0]["tick"] == 1
    a,b,c = (decisions[k] for k in ARMS)
    assert set(b.rows) == set(c.rows)-{44}
    assert b.mask == c.mask and set(a.mask) > set(b.mask)
    assert b.prefix == () and len(b.rows) == 7 and len(c.rows) == 8


def test_triplet_must_not_count_idle_seats_as_physical_rows():
    meta = dict(index=0,model="llada21",offer=8,version="t",board="b",seed=1,prompt_sha256="x")
    pack = [dict(meta, arm=a, probe=dict(first={"tick":3}, reserved_seats=2), status="complete",
                 output_ids_sha256="x", writes=dict(prefix_Q=2 if a=="prefix" else 0,
                                                    prefix_t2t=0, prefix_revocations=0),
                 roles={"clean_resolved": 2 if a=="prefix" else 0}) for a in ARMS]
    check_triplet(pack)
    pack[1]["roles"]["clean_resolved"] = 1
    with pytest.raises(ValueError, match="physical"):
        check_triplet(pack)


@pytest.mark.parametrize("model", ["llada21", "gemma"])
def test_write_observer_reads_native_edits_and_restores_worker(model):
    event = dict(prefix=[4, 5], edited=[4], revoked=[5], sampled=[4, 5, 6],
                 admitted=[6], drain_only=False)
    result = dict(trace=[event], output_ids=[2, 5])
    worker = SimpleNamespace(evaluate_fixed=lambda: result, _account=lambda *_: 17)
    original = vars(worker).copy()
    def run():
        if model == "llada21":
            assert worker.evaluate_fixed() is result
        else:
            assert worker._account({}, dict(route="ordinary", parts=[event],
                                            shared_plan=dict(drain_only=False)), 8) == 17
    with observe_writes(model, worker) as stats:
        run()
        assert (stats["prefix_Q"], stats["prefix_t2t"], stats["prefix_revocations"],
                stats["sample_mask_to_token"], stats["drain_F"]) == (2, 1, 1, 1, 0)
        assert ("output_ids_sha256" in stats) == (model == "llada21")
    event["edited"] = [0]
    with pytest.raises(ValueError, match="outside the actual sampler"), observe_writes(model, worker):
        run()
    assert vars(worker) == original


@pytest.mark.parametrize("mismatch", ["unexposed_tape", "output_ids_sha256", "F"])
def test_triplet_rejects_different_unexposed_trajectories(mismatch):
    pack = [dict(arm=a, index=0, model="llada21", offer=4, version="t", board="b", seed=1,
                 prompt_sha256="x", probe=dict(first=None, reserved_seats=0, unexposed_tape="abc"),
                 status="complete", F=5, output_ids_sha256="out", roles={"clean_resolved": 0},
                 writes=dict(prefix_Q=0, prefix_t2t=0, prefix_revocations=0)) for a in ARMS]
    check_triplet(pack)
    target = pack[-1]["probe"] if mismatch == "unexposed_tape" else pack[-1]
    target[mismatch] = 6 if mismatch == "F" else "changed"
    with pytest.raises(ValueError, match="unexposed controls disagree"):
        check_triplet(pack)

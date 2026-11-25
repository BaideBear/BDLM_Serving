import pytest
import torch

from bdlm_serving.methods.t2t.protocol import Arm
from bdlm_serving.methods.t2t.seat_audit import SeatAudit, canvas_digest, digest


def test_fingerprints_are_shape_dtype_and_value_sensitive():
    a = torch.arange(12).reshape(3, 4)
    assert digest(a.T) == digest(a.T.contiguous())
    assert len({digest(a), digest(a.flatten()), digest(a.float()), digest(a + 1)}) == 4
    assert canvas_digest([6, 7], [True, False], accepted_only=True) == canvas_digest(
        [6, 42], [True, False], accepted_only=True)
    assert canvas_digest([6, 7], [True, False]) != canvas_digest([6, 42], [True, False])


def test_one_seat_measures_only_layer_writes():
    a = SeatAudit("hard_once_rank")
    a.current = dict(victim=0, candidate=1, random_candidate=0, executed=[1])
    bank = tuple(torch.ones((1, 2, 7, 3)) for _ in range(2))
    a.begin("same", bank, 3)
    bank[0][..., 4, :] *= 3
    event = {}
    a.end(event, [5, 3], [True, True], [5, 3], [True, True], bank)
    s = event["intervention"]
    assert not s["strict_rows"] and s["resident_l15"][0]["dK"] == 0
    assert s["resident_l15"][1]["dK"] == pytest.approx(2.)
    assert s["resident_l15"][1]["dV"] == 0


def test_diagnostic_arms_do_not_change_normal_quota_validation():
    for b, q in ((32, 4), (256, 32)):
        for mode in ("base", "rank", "random"):
            assert Arm.parse(f"{q}:0:{q}:hard_once_{mode}", b).budget == q
        with pytest.raises(ValueError):
            Arm.parse(f"{q}:0:{q-1}:hard_once_rank", b)


@pytest.mark.parametrize("stratum,expected", [("slack", [0, 1]), ("mask", [0, 2])])
def test_conditional_consumes_first_legal_anchor(monkeypatch, stratum, expected):
    calls = []

    def plan(*args, **kwargs):
        calls.append(1)
        return dict(base=[0, 1], targeted=[0, 2], stratum=stratum)

    monkeypatch.setattr("bdlm_serving.methods.t2t.intervention.plan_seat", plan)
    audit = SeatAudit("hard_once_mask")
    args = ([0, 1], [], [], [], [], [], 0, 32, "gemma", 1, [])
    assert audit.choose(*args)[0] == expected
    assert audit.choose(*args) == ([0, 1], [])
    assert calls == [1]
    assert Arm.parse("32:0:32:hard_once_mask", 256).budget == 32
    with pytest.raises(ValueError):
        Arm.parse("4:0:4:hard_once_mask", 32)


@pytest.mark.parametrize("resolved,accepted", [(False, False), (True, True)])
def test_victim_admission_uses_acceptance_not_renoised_id(resolved, accepted):
    a = SeatAudit("hard_once_base")
    a.current = dict(victim=0, candidate=1, random_candidate=1, executed=[0])
    bank = tuple(torch.ones((1, 2, 2, 3)) for _ in range(2))
    a.begin("same", bank, 0)
    event = {}
    a.end(event, [13, 2], [resolved, True], [12, 2], [False, True], bank)
    s = event["intervention"]
    assert s["victim_queried"] and s["victim_unresolved_before"]
    assert s["victim_accepted"] is accepted

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

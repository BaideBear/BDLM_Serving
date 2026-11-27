"""Factor gating and first-divergence certificates, without running a model."""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from experiments.t2t.serving_slots.gemma_fixed_dual import AblationRequest
from experiments.t2t.serving_slots.gemma_warm_entropy import WarmEntropyRequest
from experiments.t2t.serving_slots.gemma_warm_entropy_eval import first_difference
from experiments.t2t.serving_slots.gemma_warm_entropy_report import bill


@pytest.mark.parametrize("enabled,n,route,sampled,expected", [
    (False, 256, "sparse", (3,), .10), (True, 256, "sparse", (3,), .20),
    (True, 255, "sparse", (255,), .10), (True, 0, "sparse", (0,), .10),
    (True, 256, "drain", (), .10), (True, 256, "verify", (), .10),
    (True, 256, "verify_paid", (), .10),
])
def test_unchanged_parent_call(monkeypatch, enabled, n, route, sampled, expected):
    req = object.__new__(WarmEntropyRequest)
    req.warm_entropy, req.tail = enabled, None
    h = NS(rows=NS(initialized=set(range(n))), sampler=NS(sc=set(range(n)), entropy_bound=.20),
           can_sample=bool(sampled))
    args = ([h], (3, 4, 5), sampled, (), sampled if n < 256 else (), 128, route)
    called = []

    def parent(self, *received):
        assert self is req and received == args and h.sampler.entropy_bound == expected
        called.append(1)
        return {"parent": 42}

    monkeypatch.setattr(AblationRequest, "_apply", parent)
    assert req._apply(*args) == {"parent":42, "warm_sample":bool(sampled and n == 256)}
    assert called == [1]


@pytest.mark.parametrize("fault", ["sc", "cold", "closed", "tail", "offer"])
def test_fail_closed(fault):
    req = object.__new__(WarmEntropyRequest)
    req.warm_entropy, req.tail = True, object() if fault == "tail" else None
    h = NS(rows=NS(initialized=set(range(256))), sampler=NS(sc=set(range(255 if fault == "sc" else 256))),
           can_sample=fault != "closed")
    with pytest.raises((ValueError, RuntimeError)):
        req._apply([h], (3,), (3,), (), (3,) if fault == "cold" else (),
                   64 if fault == "offer" else 128, "sparse")


def event():
    return dict(tick=3, ordinal=0, head_mask_before=31, head_initialized_before=256, warm_sample=True,
                positions=[12, 13], route="sparse", sampled_absolute=[12, 13], dirty_before_absolute=[],
                encoder_F=0, offer=128, parts=[dict(rows=[1, 2], sampled=[1, 2], prefix=[], age_before=4,
                age_after=5, dirty_before=[], admitted=[], revoked=[2])])


@pytest.mark.parametrize("case", ["same", "add", "retain", "misalign", "unexpected_revoke", "cold", "truncated"])
def test_first_real_change(case):
    x = event()
    y = deepcopy(x)
    a, b = [x], [y]
    if case == "add":
        y["parts"][0]["admitted"].append(1)
    elif case == "retain":
        y["parts"][0]["revoked"].clear()
    elif case == "misalign":
        y["positions"] = [22, 23]
    elif case == "unexpected_revoke":
        y["parts"][0]["revoked"].append(1)
    elif case == "cold":
        x["warm_sample"] = False
    elif case == "truncated":
        b = []
    runs = [dict(trace=t, output_ids=[8]) for t in (a,b)]
    if case == "same":
        assert first_difference(*runs) is None
    elif case in ("add", "retain"):
        assert first_difference(*runs)["extra"] == ([1] if case=="add" else [2])
    else:
        with pytest.raises(ValueError):
            first_difference(*runs)


def test_offline_bill_rejects_partial_terminal_block():
    x = dict(encoder_F=0, encoder_Q=0, decoder_F=2, decoder_Q=256, route_F={"verify":2},
             route_Q={"verify":256}, F=2, max_decoder_Q=128, joint_F=0, output_ids=[1, 6], correct=True,
             N=1, phases={"verify_F":2, "verify_Q":256})
    assert bill(x)["total_Q"] == 256
    x.update(decoder_Q=128, route_Q={"verify":128}, phases={"verify_F":2, "verify_Q":128})
    with pytest.raises(ValueError, match="terminal"):
        bill(x)

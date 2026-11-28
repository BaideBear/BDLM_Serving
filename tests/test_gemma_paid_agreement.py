"""Model-free contracts for the Gemma Q128 paid-agreement adapter."""
from pathlib import Path

import numpy as np
import pytest
import torch

from bdlm_serving.methods.t2t.paid_agreement import EvidenceLedger
from experiments.t2t.serving_slots.gemma_ibs_certified_eos import _AFTER, _PLAN, _SIG
from experiments.t2t.serving_slots.gemma_paid_agreement import Agreement, _UNMASK
from experiments.t2t.serving_slots.gemma_paid_agreement_eval import check_fork


@pytest.mark.parametrize("block", [0, 1])
def test_first_fork_converts_sampler_rows_and_preserves_other_decisions(block):
    offset = 256 * block
    pos = [offset+17, offset+133]
    first = dict(tick=2, block=block, extra=[pos[1]])
    base = dict(block=block, positions=pos, offer=128, q=2, admitted=[pos[0]])
    applied = dict(base, admitted=pos)
    obs = dict(first, positions=pos, proposals=[10, 11], native=[True, False],
               probabilities=[.95, .7])
    raw = dict(ids=[1, 2], pos=pos, rng="same",
               sample=dict(rows=[17, 133], proposal=[10, 11], take=[1, 0]))
    modified = dict(raw, sample=dict(raw["sample"], take=[1, 1]))
    check_fork(first, applied, base, modified, raw, obs, obs)
    with pytest.raises((RuntimeError, ValueError)):
        check_fork(first, applied, base, modified, dict(raw, rng="not the same"), obs, obs)


class FakeUnmask:
    def __init__(self, base):
        self.base = base

    def low_confidence_dynamic_sparse(self, raw, proposal, prior, sel, row_width):
        return self.base.clone()


def raw_for(token=3, flat=False):
    raw = torch.full((2, 7), -10.0)
    if flat:
        raw.zero_()
    else:
        raw[:, token] = 10.0
    return raw


def test_ledger_requires_two_adjacent_rejected_votes_above_floor():
    ledger = EvidenceLedger(floor=0.5)
    assert ledger.observe(block=0, tick=0, positions=[10], proposals=[3],
                          probabilities=[0.9], native=[False], excluded_ids=set()) == ()
    assert ledger.observe(block=0, tick=1, positions=[10], proposals=[3],
                          probabilities=[0.9], native=[False], excluded_ids=set()) == (10,)


def test_ledger_rejects_skip_native_special_and_weak_votes():
    base = dict(block=0, positions=[10], proposals=[3], probabilities=[0.9],
                native=[False], excluded_ids=set())
    ledger = EvidenceLedger(floor=0.5)
    ledger.observe(tick=0, **base)
    assert ledger.observe(tick=2, **base) == ()          # skipped tick 1
    ledger = EvidenceLedger(floor=0.5)
    ledger.observe(tick=0, **base)
    assert ledger.observe(tick=1, **{**base, "native": [True]}) == ()
    ledger = EvidenceLedger(floor=0.5)
    ledger.observe(tick=0, **base)
    assert ledger.observe(tick=1, **{**base, "proposals": [6], "excluded_ids": {6}}) == ()
    ledger = EvidenceLedger(floor=0.5)
    ledger.observe(tick=0, **base)
    assert ledger.observe(tick=1, **{**base, "probabilities": [0.5]}) == ()


def test_ledger_tick_must_increase():
    ledger = EvidenceLedger(floor=0.5)
    ledger.observe(block=0, tick=0, positions=[], proposals=[], probabilities=[],
                   native=[], excluded_ids=set())
    with pytest.raises(ValueError, match="strictly increase"):
        ledger.observe(block=0, tick=0, positions=[], proposals=[], probabilities=[],
                       native=[], excluded_ids=set())


def test_agreement_observer_does_not_apply_and_apply_sets_extra_take():
    seen = []
    observer = Agreement(apply=False, excluded_ids={6})
    applier = Agreement(apply=True, excluded_ids={6})
    applier.record_take = seen.append
    for agreement in (observer, applier):
        base = torch.zeros(2, dtype=torch.uint8)
        first = agreement.observe(FakeUnmask(base), raw_for(), torch.tensor([3, 3]),
                                  np.zeros(2, dtype=np.uint8), torch.tensor([0, 1]),
                                  2, 0, 0)
        assert first.tolist() == [0, 0]
        second = agreement.observe(FakeUnmask(base), raw_for(), torch.tensor([3, 3]),
                                   np.zeros(2, dtype=np.uint8), torch.tensor([0, 1]),
                                   2, 0, 1)
        if agreement.apply:
            assert second.tolist() == [1, 1]
            assert agreement.first_extra == dict(tick=1, block=0, extra=[0, 1])
        else:
            assert second.tolist() == [0, 0]
            assert agreement.first_extra is None
    assert seen == [[1, 1]]


def test_agreement_skips_non_adjacent_native_or_special():
    base = torch.zeros(2, dtype=torch.uint8)
    agreement = Agreement(apply=True, excluded_ids={6})
    agreement.observe(FakeUnmask(base), raw_for(), torch.tensor([3, 3]),
                      np.zeros(2, dtype=np.uint8), torch.tensor([0, 1]), 2, 0, 0)
    out = agreement.observe(FakeUnmask(base), raw_for(), torch.tensor([3, 3]),
                            np.zeros(2, dtype=np.uint8), torch.tensor([0, 1]), 2, 0, 2)
    assert out.tolist() == [0, 0] and agreement.first_extra is None
    agreement = Agreement(apply=True, excluded_ids={6})
    agreement.observe(FakeUnmask(torch.ones(2, dtype=torch.uint8)), raw_for(), torch.tensor([3, 3]),
                      np.zeros(2, dtype=np.uint8), torch.tensor([0, 1]), 2, 0, 0)
    out = agreement.observe(FakeUnmask(torch.ones(2, dtype=torch.uint8)), raw_for(), torch.tensor([3, 3]),
                            np.zeros(2, dtype=np.uint8), torch.tensor([0, 1]), 2, 0, 1)
    assert out.tolist() == [1, 1] and agreement.first_extra is None
    agreement = Agreement(apply=True, excluded_ids={6})
    agreement.observe(FakeUnmask(base), raw_for(6), torch.tensor([6, 6]),
                      np.zeros(2, dtype=np.uint8), torch.tensor([0, 1]), 2, 0, 0)
    out = agreement.observe(FakeUnmask(base), raw_for(6), torch.tensor([6, 6]),
                            np.zeros(2, dtype=np.uint8), torch.tensor([0, 1]), 2, 0, 1)
    assert out.tolist() == [0, 0] and agreement.first_extra is None


def test_source_anchors_are_unique():
    root = Path(__file__).resolve().parents[1]
    src = (root / "bdlm_serving/runtime/diffusiongemma_optimus.py").read_text()
    assert src.count(_SIG) == 1
    body = src.split(_SIG)[1].split("\n\ndef ", 1)[0]
    assert body.count(_PLAN) == body.count(_AFTER) == body.count(_UNMASK) == 1

import pytest
from bdlm_serving.methods.t2t.physical_prefix import plan_prefix
from experiments.t2t.prefix_window.grid import MODELS, arms, ratio_caps
from experiments.t2t.prefix_window.ratio_opportunity import tally


def test_gemma_dose_no_reruns_or_missing_cells():
    for q in MODELS["gemma"][1]:
        old, extra = (set(arms("gemma", q, mode)) for mode in ("slack", "dose"))
        assert not old & extra
        assert old | extra == {f"{q}:0:{q-k}:hard_prefix_slack_{selector}"
                               for k in (1, 2, 4) for selector in ("right", "random")}


def test_gemma_ratio_deduplicates_registered_policy_ids():
    count = 0
    for q in MODELS["gemma"][1]:
        extra = set(arms("gemma", q, "ratio"))
        old = set(arms("gemma", q, "dose") + arms("gemma", q, "slack"))
        assert not extra & old
        assert old | extra == {f"{q}:0:{q-k}:hard_prefix_slack_{s}"
                               for k in {1, 2, 4, *ratio_caps(q)} for s in ("right", "random")}
        count += len(extra)
    assert count == 48
    assert ratio_caps(256) == (16, 32, 64, 128, 192, 255)
    with pytest.raises(ValueError):
        arms("llada21", 16, "ratio")


def test_full_slack_protects_every_baseline_row_and_never_creates_a_forward():
    for q in MODELS["gemma"][1]:
        for selector in ("right", "random"):
            plan = plan_prefix([True] * 254 + [False] * 2, range(256), {253}, q, q-1,
                               f"hard_prefix_slack_{selector}", seed=0, block_start=0, clock=1)
            assert len(plan.positions) == q and len(plan.added_rows) == q-3
            assert set(plan.zero_rows) == {253, 254, 255} and not plan.evicted_rows
            empty = plan_prefix([True]*256, range(256), set(), q, q-1,
                                f"hard_prefix_slack_{selector}", seed=0, block_start=0, clock=1)
            assert not empty.positions


def test_counterfactual_counts_are_per_forward_not_pooled_capacity():
    assert tally({0: 2, 1: 3, 8: 1}, 4) == (7, 1, 1)
    assert tally({0: 2, 1: 3, 8: 1}, 0) == (0, 0, 4)

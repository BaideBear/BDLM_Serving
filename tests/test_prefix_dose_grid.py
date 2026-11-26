import pytest
import csv
from bdlm_serving.methods.t2t.physical_prefix import plan_prefix
from experiments.t2t.prefix_window.grid import MODELS, arms, ratio_caps
from experiments.t2t.prefix_window.ratio_opportunity import tally
from experiments.t2t.prefix_window.ratio_report import rows


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


def test_online_shares_count_actual_rows_and_missing_is_not_a_zero(tmp_path):
    def write(name, row):
        with (tmp_path / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerow(row)

    policy = "t2t_q256_f0_m128_hard_prefix_slack_right"
    ex = dict(policy=policy, cohort="all90", requests=90, events=200, mean_q_rows=100.5,
              mask_displaced=0, ordinary_displaced=0, added_rows=640,
              noop_events=190, saturated_events=2, untouched_requests=82)
    write("exposure-gemma-example.csv", ex)
    write("gemma-q256-slack-k128-right.csv", dict(scope="all", window=90, requests=90,
          arm_policy=policy, baseline_correct=83, arm_correct=85, rescued=3, lost=1,
          quality_delta_ci_low=-.05, quality_delta_ci_high=.09,
          macro_tpf_delta_ci_low=-.4, macro_tpf_delta_ci_high=2.4,
          baseline_macro_tpf=10, arm_macro_tpf=11, baseline_pooled_token_efficiency=.11,
          arm_pooled_token_efficiency=.10, arm_physical_Q=36000))
    result, = rows(tmp_path)
    assert result["affected_events"] == 10 and result["affected_requests"] == 8
    assert result["share_selected"] == 640/20100
    assert result["share_all_generation"] == 640/36000
    write("exposure-gemma-example.csv", dict(ex, mask_displaced=1))
    with pytest.raises(ValueError):
        list(rows(tmp_path))

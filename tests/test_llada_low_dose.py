"""Low-dose head-only contracts: C4/8/16/32 x clean-prefix share.

C4 x 1/8 is a registered alias of C4 x 0: choose()'s cap is
``min(room,32)*1//8`` and room <= q = 4, so the cap is identically zero. The
suite records that proof and never spends a seat re-running it.
"""
import pytest
import torch

from bdlm_serving.methods.t2t.budget_family import Law
from bdlm_serving.methods.t2t.llada_budget_request import LLaDABudgetRequest
from test_llada_budget_request import Measured, drive

from experiments.t2t.serving_slots.cross_race_llada import (
    LOW_DOSE_ALIASES, LOW_DOSE_ARMS, LOW_DOSE_SHARES, LOW_DOSE_TIERS,
    audit_low_dose, cohort, low_dose_metrics, prefix_cap,
)

LEVELS = ("tick", "q", "rows", "sampled", "dirty_before", "dirty", "dirty_after",
          "cold", "mask", "prefix", "written_absolute", "admitted", "edited",
          "revoked", "retired_blocks", "drain_only", "phase", "sample_clock")


def run_head(q, num, den, prompt=32, cap=69, mode="high"):
    torch.manual_seed(20)
    model = Measured(mode)
    req = LLaDABudgetRequest(model, torch.ones((1, prompt), dtype=torch.long),
                             mask_id=0, max_new_tokens=cap, cross="head", borrow=False,
                             law=Law(num, den))
    drive(req, model, [q])
    return req


def levels(records):
    return [{k: e[k] for k in LEVELS} for e in records]


def test_low_dose_arm_table_has_exactly_the_registered_16_minus_alias():
    names = [a[0] for a in LOW_DOSE_ARMS]
    assert len(LOW_DOSE_ARMS) == 4 * len(LOW_DOSE_SHARES) - 1 == 15
    assert len(set(names)) == len(names)
    assert "c4-p1of8" not in names
    assert "c4-p0" in names
    for q in LOW_DOSE_TIERS:
        for num, den in LOW_DOSE_SHARES:
            name = f"c{q}-p0" if num == 0 else f"c{q}-p{num}of{den}"
            if (q, num, den) == (4, 1, 8):
                assert name not in names
            else:
                assert name in names
    assert [(a["alias"], a["of"], a["q"], a["share"]) for a in LOW_DOSE_ALIASES] == [
        ("c4-p1of8", "c4-p0", 4, "1/8")]


def test_prefix_cap_floor_is_zero_for_c4_eighth_and_distinct_elsewhere():
    for dirty in range(5):
        assert prefix_cap(4, 0, 1, dirty) == 0
        assert prefix_cap(4, 1, 8, dirty) == 0
    caps = {q: [prefix_cap(q, n, d) for n, d in LOW_DOSE_SHARES]
            for q in LOW_DOSE_TIERS}
    assert caps[4] == [0, 0, 1, 2]
    assert caps[8] == [0, 1, 2, 4]
    assert caps[16] == [0, 2, 4, 8]
    assert caps[32] == [0, 4, 8, 16]
    assert prefix_cap(32, 1, 2, dirty=8) == 12


def test_c4_zero_and_one_eighth_are_the_same_execution():
    zero = run_head(4, 0, 1)
    eighth = run_head(4, 1, 8)
    assert zero.tokens == eighth.tokens
    assert levels(zero.records) == levels(eighth.records)
    assert all(not e["prefix"] for e in zero.records)
    assert all(not e["prefix"] for e in eighth.records)


def test_actual_warm_prefix_seats_never_exceed_the_registered_cap():
    for q in (8, 16, 32):
        for num, den in LOW_DOSE_SHARES:
            req = run_head(q, num, den)
            m = low_dose_metrics(dict(trace=req.records))
            cap = sum(prefix_cap(e["q"], num, den, len(e["dirty"]))
                      for e in req.records if e["sampled"] and not e["cold"])
            assert m["warm_prefix_seats"] <= cap
            if num == 0:
                assert m["warm_prefix_seats"] == 0
                assert m["prefix_seats"] == 0


def test_c4_eighth_is_indistinguishable_from_zero_but_quarter_has_a_cap():
    zero = low_dose_metrics(dict(trace=run_head(4, 0, 1).records))
    eighth = low_dose_metrics(dict(trace=run_head(4, 1, 8).records))
    quarter = run_head(4, 1, 4)
    qm = low_dose_metrics(dict(trace=quarter.records))
    assert zero["warm_prefix_seats"] == eighth["warm_prefix_seats"] == 0
    assert qm["warm_prefix_seats"] <= sum(
        prefix_cap(e["q"], 1, 4, len(e["dirty"]))
        for e in quarter.records if e["sampled"] and not e["cold"])


def test_low_dose_metrics_reports_seats_and_t2t_totals():
    trace = [
        dict(sampled=[1, 2, 3, 4], cold=[], q=4, prefix=[4], mask=[1, 2],
             dirty=[3], dirty_before=[3], admitted=[1], edited=[4], revoked=[],
             drain_only=False),
        dict(sampled=[], cold=[], q=2, prefix=[], mask=[], dirty=[1, 2],
             dirty_before=[1, 2], admitted=[], edited=[], revoked=[], drain_only=True),
        dict(sampled=[5], cold=[5], q=1, prefix=[], mask=[5], dirty=[],
             dirty_before=[], admitted=[5], edited=[], revoked=[], drain_only=False),
    ]
    m = low_dose_metrics(dict(trace=trace))
    assert m["warm_F"] == 1 and m["warm_rows"] == 4
    assert m["warm_prefix_seats"] == 1 and m["warm_prefix_row_share"] == 0.25
    assert m["prefix_seats"] == 1 and m["mask_seats"] == 3
    assert m["dirty_selected_seats"] == 3 and m["dirty_before_seats"] == 3
    assert m["m2t_accepts"] == 2 and m["t2t_edits"] == 1 and m["t2t_revoked"] == 0
    assert m["cold_seats"] == 1 and m["drain_F"] == 1
    assert m["zero_prefix_warm_F"] == 0


def test_audit_enforces_the_registered_prefix_cap():
    row = dict(trace=[dict(rows=[1, 2, 3, 4], sampled=[1, 2, 3, 4], cold=[], q=4,
                           prefix=[4], mask=[1], dirty=[], admitted=[1], edited=[4],
                           revoked=[], drain_only=False)])
    with pytest.raises(RuntimeError, match="exceeds cap"):
        audit_low_dose(row, 0, 1)   # cap 0, one prefix seat
    audit_low_dose(row, 1, 4)       # cap 1, one prefix seat is legal


def test_cohort_split_is_original30_then_new60():
    assert [cohort(i) for i in (0, 29, 30, 89)] == [
        "original30", "original30", "new60", "new60"]


def test_low_dose_summary_splits_original30_new60_all90(tmp_path):
    import json
    from experiments.t2t.serving_slots.cross_race_llada import low_dose_summary

    rows = tmp_path / "rows"
    rows.mkdir()

    def emit(index):
        row = dict(index=index, arm="c8-p1of8", q=8, prefix_share="1/8",
                   N=10, F=2, Q=40, TPF=5.0, correct=1, failure=None,
                   warm_F=1, warm_rows=8, warm_prefix_seats=1, prefix_seats=1,
                   mask_seats=2, dirty_selected_seats=0, dirty_before_seats=0,
                   m2t_accepts=1, t2t_edits=0, t2t_revoked=0, cold_seats=0,
                   drain_F=0, zero_prefix_warm_F=0)
        (rows / f"c8-p1of8-{index:02d}.json").write_text(json.dumps(row))

    emit(0)
    emit(30)
    data = low_dose_summary(tmp_path)
    groups = {(g["arm"], g["cohort"]): g for g in data["by_arm_cohort"]}
    assert groups[("c8-p1of8", "original30")]["requests"] == 1
    assert groups[("c8-p1of8", "new60")]["requests"] == 1
    assert groups[("c8-p1of8", "all90")]["requests"] == 2
    assert groups[("c8-p1of8", "all90")]["Q"] == 80
    assert data["aliases"] == list(LOW_DOSE_ALIASES)

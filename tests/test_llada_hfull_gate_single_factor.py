"""Unit contracts for the Q64 gate single-factor runner's pure pairing logic."""
import pytest

from experiments.t2t.serving_slots import llada_hfull_gate_single_factor as mod


def event(*, rows=range(32, 64), tail_opened=False, q=None,
          layer_physical_q=None, head_mask_at_plan=10, pre_state=None):
    rows = list(rows)
    q = len(rows) if q is None else q
    return dict(tick=0, rows=rows, q=q, tail_opened=tail_opened,
                layer_physical_q=layer_physical_q or [q] * 20,
                head_mask_at_plan=head_mask_at_plan,
                pre_state=pre_state or {"x": "same"})


def make_row(index, *, old_correct=True, new_correct=True, old_trace=None,
             new_trace=None, old_N=100, new_N=110, old_F=10, new_F=9,
             old_Q=500, new_Q=450):
    return dict(
        index=index,
        old=dict(correct=old_correct, N=old_N, F=old_F, Q=old_Q,
                 trace=old_trace if old_trace is not None else [event()]),
        new=dict(correct=new_correct, N=new_N, F=new_F, Q=new_Q,
                 trace=new_trace if new_trace is not None else [event()]),
        compare=None)


def test_compare_pair_reports_first_divergence_and_common_metrics():
    old_trace = [event(), event(rows=range(32, 96), tail_opened=True)]
    new_trace = [event(), event(rows=range(32, 64), tail_opened=False)]
    old_trace[0]["tick"] = new_trace[0]["tick"] = 0
    old_trace[1]["tick"] = new_trace[1]["tick"] = 1
    pair = mod.compare_pair(dict(correct=True, N=100, F=10, Q=500, trace=old_trace),
                            dict(correct=False, N=110, F=9, Q=450, trace=new_trace))
    assert pair["old_correct"] is True and pair["new_correct"] is False
    assert pair["old_TPF"] == pytest.approx(10.0)
    assert pair["new_TE"] == pytest.approx(110 / 450)
    assert pair["old_strict_cap"] is True and pair["new_strict_cap"] is True
    assert pair["pre_state_equal_at_tick0"] is True
    divergence = pair["first_divergence"]
    assert divergence["tick"] == 1
    assert divergence["old_rows"] == 64 and divergence["new_rows"] == 32
    assert divergence["old_tail_opened"] is True
    assert divergence["new_tail_opened"] is False


def test_compare_pair_flags_a_forward_over_the_strict_cap():
    bad = [event(q=65, layer_physical_q=[65] * 20)]
    pair = mod.compare_pair(dict(correct=True, N=10, F=1, Q=65, trace=bad),
                            dict(correct=True, N=10, F=1, Q=32, trace=[event()]))
    assert pair["old_strict_cap"] is False
    assert pair["old_max_Q"] == 65


def test_summarize_cohort_counts_pairs_and_pooled_metrics():
    rows = {
        0: make_row(0, old_correct=True, new_correct=True),
        1: make_row(1, old_correct=False, new_correct=True),
        2: make_row(2, old_correct=True, new_correct=False),
    }
    for row in rows.values():
        row["compare"] = mod.compare_pair(row["old"], row["new"])
    stats = mod.summarize_cohort("all90", tuple(range(3)), rows)
    assert stats["requests"] == 3
    assert stats["old_correct"] == 2 and stats["new_correct"] == 2
    assert stats["old_false_new_true"] == 1
    assert stats["old_true_new_false"] == 1
    assert stats["pre_state_equal_tick0"] == 3
    assert stats["old_pooled_TPF"] == pytest.approx(stats["old_N"] / stats["old_F"])
    assert stats["new_pooled_TE"] == pytest.approx(stats["new_N"] / stats["new_Q"])
    assert stats["old_strict_cap"] is True and stats["new_strict_cap"] is True


def test_incomplete_cohort_is_partial_and_present_only():
    rows = {0: make_row(0), 1: make_row(1)}
    for row in rows.values():
        row["compare"] = mod.compare_pair(row["old"], row["new"])
    stats = mod.summarize_cohort("all90", (0, 1, 2), rows)
    assert stats["coverage"]["present"] == 2
    assert stats["coverage"]["missing"] == [2]
    assert stats["coverage"]["complete"] is False
    assert stats["partial"] is True and stats["metrics_scope"] == "present_only"

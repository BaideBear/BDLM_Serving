"""Model-free contracts for the Q16 active-window paid warm-MASK factor."""
import importlib.util
from pathlib import Path

import pytest

from bdlm_serving.methods.t2t.budget_family import Plan
from experiments.t2t.serving_slots.llada_window_sample import extra_mask, first

MODULE = Path(__file__).resolve().parents[1] / "tools" / "launch_llada_window_sample_q16.py"
SPEC = importlib.util.spec_from_file_location("launch_window_sample_q16", MODULE)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_extra_mask_on_a_mixed_cold_call_at_q16():
    plan = Plan(rows=(32, 33, 34), dirty=(32,), cold=(34,),
                mask=(33,), prefix=(), drain_only=False)
    assert extra_mask(plan, (34,)) == (33,)
    assert extra_mask(plan, (32, 34)) == ()   # not a cold-only call
    assert extra_mask(Plan((), (), (), (), (), True), ()) == ()  # drain


def trace_event(*, tick=0, rows=(32, 33, 34), sampled=(34,), dirty_before=(32,),
                dirty=(32,), prefix=(), mask=(33,), cold=(34,), pre_state=None,
                paid=(), phase="sample", sample_clock=1, visible_kv=64):
    return dict(tick=tick, rows=list(rows), sampled=list(sampled),
                dirty_before=list(dirty_before), dirty=list(dirty),
                prefix=list(prefix), mask=list(mask), cold=list(cold),
                pre_state=pre_state or {"x": "same"}, paid_waiting_mask=list(paid),
                phase=phase, sample_clock=sample_clock,
                q=len(rows), layer_physical_q=[len(rows)] * 20, visible_kv=visible_kv)


def test_first_paid_mask_keeps_the_physical_causal_match():
    old = dict(trace=[trace_event(sampled=(34,), mask=(33,), cold=(34,)),
                      trace_event(tick=1, sampled=(34,), mask=(33,), cold=(34,))])
    new = dict(trace=[trace_event(sampled=(34,), mask=(33,), cold=(34,)),
                      trace_event(tick=1, sampled=(33, 34), mask=(33,),
                                  cold=(34,), paid=(33,))])
    result = first(old, new)
    assert result == dict(tick=1, width=3, added=[33])


def test_first_paid_mask_rejects_a_changed_causal_match():
    old = dict(trace=[trace_event(sampled=(34,), mask=(33,), cold=(34,))])
    new = dict(trace=[trace_event(sampled=(33, 34), mask=(33,), cold=(34,),
                                 paid=(33,), visible_kv=63)])
    with pytest.raises(Exception):
        first(old, new)


def test_summary_uses_canonical_pooled_te():
    rows = {
        0: dict(index=0, old=dict(correct=True, N=10, F=2, Q=40),
                paid=dict(correct=True, N=11, F=2, Q=40),
                first=dict(tick=3, width=4, added=[33]),
                control_replay=True),
        1: dict(index=1, old=dict(correct=False, N=10, F=2, Q=40),
                paid=dict(correct=False, N=10, F=2, Q=40),
                first=None, control_replay=False),
    }
    stats = mod.summarize(rows, (0, 1), 16)
    assert stats["requests"] == 2
    assert stats["exposed"] == 1 and stats["first_checks"] == 1
    assert stats["control_replay"] == 1 and stats["control_replay_failures"] == 1
    assert stats["old_pooled_TE"] == 20 / 80
    assert stats["new_pooled_TE"] == 21 / 80
    assert stats["old_macro_TPF"] == 5.0
    assert stats["new_macro_TPF"] == 5.25
    assert stats["first_tick"]["min"] == 3

"""Model-free contracts for the research KV-horizon branch and its ledger."""
import pytest
import torch

from experiments.t2t.serving_slots.llada_kv_horizon_eval import (
    first_divergence, summarize_cohort)
from experiments.t2t.serving_slots.llada_kv_horizon_research import KVHorizonRequest


class Bank:
    def __init__(self, last_end=0, capacity=96, buffers=None):
        self.last_end = last_end
        self.capacity = capacity
        self.initialized = set(range(last_end))
        self.visible_end = last_end
        self.buffers = buffers or []
        self.selected = None

    def begin(self, positions, rows):
        self.selected = positions
        self.visible_end = max(self.last_end, rows[-1] + 1)

    def get_seq_length(self, layer_idx=0):
        return self.last_end


class Engine:
    def __init__(self):
        self.calls = []

    def _execute(self, state, pos, end, cache, phase, stage, store, budget=None):
        self.calls.append(dict(end=end, cache_visible=getattr(cache, "visible_end", None),
                               cache_type=type(cache).__name__,
                               cache_seq=cache.get_seq_length(), budget=budget))
        state.events.append({"phase": phase})
        return torch.zeros(1, pos.numel(), 2)


def request(*, phase="selected", last_end=64, buffers=True):
    req = object.__new__(KVHorizonRequest)
    req.x = torch.zeros(1, 131, dtype=torch.long)
    req.head = 32
    req.bank = Bank(last_end=last_end,
                    buffers=[(torch.zeros(1, 96, 2), torch.zeros(1, 96, 2))]
                    if buffers else [])
    req.engine = Engine()
    return req


def test_generation_project_exposes_only_the_rightmost_selected_row():
    req = request()
    event, _ = req._project((32, 33), 4, "selected")
    call = req.engine.calls[0]
    assert call["end"] == 34
    assert call["cache_visible"] == 34
    assert call["cache_type"] == "_SelectedCache"
    assert call["cache_seq"] == 34
    assert req.bank.visible_end == 64          # high-water preserved
    assert req.bank.last_end == 64
    assert {32, 33} <= req.bank.initialized
    assert event["kv_horizon"] == 34
    assert event["kv_high_water"] == 64
    assert event["kv_horizon_gap"] == 30


def test_prefill_uses_the_full_prompt_horizon():
    req = request(phase="prefill", last_end=0, buffers=False)
    event, _ = req._project(tuple(range(32)), 32, "prefill")
    call = req.engine.calls[0]
    assert call["end"] == 32
    assert call["cache_type"] == "Bank"
    assert event["kv_horizon"] == 32 and event["kv_horizon_gap"] == 0


def trace_event(*, rows=(32, 33), sampled=(33,), dirty_before=(32,),
                written=(33,), admitted=(33,), edited=(), revoked=(),
                pre_state=None, gap=0):
    return dict(tick=0, rows=list(rows), sampled=list(sampled),
                dirty_before=list(dirty_before), written_absolute=list(written),
                admitted=list(admitted), edited=list(edited), revoked=list(revoked),
                pre_state=pre_state or {"x": "same"}, kv_horizon_gap=gap,
                kv_horizon=max(rows) + 1 if rows else 0,
                kv_high_water=(max(rows) + 1 + gap) if rows else 0)


def test_first_divergence_separates_output_from_plan_change():
    base = trace_event()
    old = [base, base, trace_event(rows=(32, 34), sampled=(34,))]
    new = [base, trace_event(written=(34,), admitted=(34,), gap=7),
           trace_event(rows=(32, 35), sampled=(35,), gap=7)]
    ledger = first_divergence(dict(trace=old, input_sha256="p"),
                              dict(trace=new, input_sha256="p"))
    assert ledger["diverged"] and ledger["visibility_only"]
    assert ledger["same_selection_before"] and ledger["same_input"]
    assert ledger["first_output_tick"] == 1 and ledger["first_plan_tick"] == 2
    assert ledger["first_horizon_tick"] == 1
    assert ledger["horizon_gap_at_first_horizon"] == 7
    assert ledger["horizon_gap_at_first_output"] == 7
    assert ledger["checks"]["rows_equal_at_first"] is True
    assert ledger["checks"]["sampled_equal_at_first"] is True
    assert ledger["checks"]["written_absolute_equal_at_first"] is False


def test_first_divergence_rejects_a_plan_change_before_output():
    base = trace_event()
    old = [base, base, base]
    new = [base, trace_event(rows=(32, 34), sampled=(34,), written=(33,), gap=7),
           trace_event(rows=(32, 34), sampled=(34,), written=(34,), gap=7)]
    ledger = first_divergence(dict(trace=old), dict(trace=new))
    assert ledger["diverged"] and not ledger["visibility_only"]
    assert ledger["checks"]["rows_equal_at_first"] is False


def test_no_divergence_is_recorded():
    base = trace_event()
    ledger = first_divergence(dict(trace=[base]), dict(trace=[base]))
    assert ledger["diverged"] is False and ledger["visibility_only"] is False
    assert ledger["first_horizon_tick"] is None


def test_summarize_cohort_counts_divergence_and_horizon_gap():
    base = trace_event()
    old = [base, base, base]
    new = [base, trace_event(written=(34,), admitted=(34,), gap=7), base]
    rows = {
        0: dict(index=0, old=dict(correct=True, N=10, F=2, Q=20, trace=old,
                                  input_sha256="p"),
                new=dict(correct=True, N=11, F=2, Q=20, trace=new,
                         input_sha256="p"),
                compare=first_divergence(dict(trace=old, input_sha256="p"),
                                         dict(trace=new, input_sha256="p"))),
        1: dict(index=1, old=dict(correct=False, N=10, F=2, Q=20, trace=old,
                                  input_sha256="q"),
                new=dict(correct=False, N=10, F=2, Q=20, trace=old,
                         input_sha256="q"),
                compare=first_divergence(dict(trace=old, input_sha256="q"),
                                         dict(trace=old, input_sha256="q"))),
    }
    stats = summarize_cohort("original30", (0, 1), rows, 8)
    assert stats["requests"] == 2
    assert stats["diverged"] == 1 and stats["visibility_only"] == 1
    assert stats["first_horizon_tick"]["min"] == 1
    assert stats["horizon_gap_at_horizon"]["min"] == 7
    assert stats["horizon_gap_at_output"]["min"] == 7
    assert stats["first_output_tick"]["min"] == 1

"""Unit contracts for the single-factor runner's pure comparison/summary logic."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

MODULE = Path(__file__).resolve().parents[1] / "experiments" / "t2t" / "serving_slots" / "llada_hfull_single_factor.py"
SPEC = importlib.util.spec_from_file_location("llada_hfull_single_factor", MODULE)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def event(*, tick=0, rows=range(32, 96), sampled=(), cold=(), mask=(), prefix=(),
          dirty=(), warm_head_sample=0, warm_head_rows=(), pre_state=None,
          layer_physical_q=None, head_mask_at_plan=12):
    rows = list(rows)
    return dict(tick=tick, rows=rows, q=len(rows), sampled=list(sampled),
                cold=list(cold), mask=list(mask), prefix=list(prefix),
                dirty=list(dirty), dirty_before=[], warm_head_sample=warm_head_sample,
                warm_head_rows=list(warm_head_rows), pre_state=pre_state,
                layer_physical_q=layer_physical_q or [len(rows)] * 20,
                head_mask_at_plan=head_mask_at_plan)


def old_row():
    return dict(trace=[event(sampled=[64, 65], cold=[64, 65], mask=[40, 41],
                             pre_state=dict(x_sha256="same"))])


def new_row(**kw):
    return dict(trace=[event(sampled=[40, 41, 64, 65], cold=[64, 65], mask=[40, 41],
                             warm_head_sample=2, warm_head_rows=[40, 41],
                             pre_state=dict(x_sha256="same"), **kw)])


def test_first_intervention_passes_all_lockstep_checks():
    compared = mod.compare_first(old_row(), new_row())
    assert compared["intervened"] and compared["passed"]
    assert compared["pre_state_verified"] is True
    assert compared["added"] == [40, 41]
    assert all(value is True for value in compared["checks"].values())


def test_first_intervention_fails_on_rows_or_forward_mismatch():
    bad_rows = new_row(rows=range(32, 95))
    compared = mod.compare_first(old_row(), bad_rows)
    assert compared["intervened"] and not compared["passed"]
    assert compared["checks"]["rows_equal"] is False
    bad_forward = new_row(layer_physical_q=[1] * 20)
    compared = mod.compare_first(old_row(), bad_forward)
    assert compared["checks"]["same_forward"] is False


def test_reused_old_control_without_pre_state_is_not_verified_but_not_failed():
    old = dict(trace=[event(sampled=[64, 65], cold=[64, 65], mask=[40, 41])])
    compared = mod.compare_first(old, new_row())
    assert compared["intervened"] and compared["passed"]
    assert compared["pre_state_verified"] is False
    assert compared["checks"]["pre_state_equal"] is None


def test_no_intervention_returns_false():
    compared = mod.compare_first(old_row(), dict(trace=[event(sampled=[64, 65])]))
    assert compared["intervened"] is False


def test_summary_counts_old_new_failures_and_first_checks():
    rows = [
        dict(index=0, old=dict(correct=True, N=10, F=3, Q=30),
             new=dict(correct=False, N=9, F=4, Q=40),
             compare=dict(intervened=True, passed=True, tick=2, added=[40, 41])),
        dict(index=1, old=dict(correct=False, N=0, F=0, Q=0), old_failure="boom",
             new=dict(correct=True, N=5, F=2, Q=20),
             compare=dict(intervened=False, passed=False)),
        dict(index=2, new_failure="cuda"),
    ]
    summary = mod.summarize(rows)
    assert summary["requests"] == 3
    assert summary["intervened"] == 1 and summary["first_checks_passed"] == 1
    assert summary["old_failures"] == 1 and summary["new_failures"] == 1
    assert summary["old_correct"] == 1 and summary["new_correct"] == 1
    assert summary["old_N"] == 10 and summary["new_N"] == 14
    assert summary["per_index"][0]["first_tick"] == 2
    assert summary["per_index"][2]["new_failure"] == "cuda"


def test_cap_claim_needs_an_observed_nonempty_history_projection_on_every_layer():
    e=dict(q=2,physical_q=2,rows=[32,33],absolute_positions=[32,33],
           layer_physical_q=[2]*20,prefix_kv_length=32)
    assert mod._strict_cap(dict(F=1,Q=2,trace=[e]))
    for patch in (dict(layer_physical_q=[]),dict(rows=[]),dict(prefix_kv_length=0),dict(physical_q=32)):
        assert not mod._strict_cap(dict(F=1,Q=2,trace=[dict(e,**patch)]))
    assert not mod._strict_cap(dict(F=1,Q=2))


def test_state_digest_tracks_x_head_and_dirty():
    req = SimpleNamespace(
        x=torch.tensor([[1, 2, 3, 0]]), head=0, end=4, origin=3,
        sample_clock=0, bank=SimpleNamespace(last_end=3), status="ready",
        stopping=False, dirty=set())
    base = mod.state_digest(req)
    assert mod.state_digest(req) == base
    req.x[0, 3] = 9
    assert mod.state_digest(req) != base
    req.x[0, 3] = 0
    req.dirty = {1}
    assert mod.state_digest(req) != base

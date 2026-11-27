import pytest

from tools.audit_ibs_spare import aligned, inspect


def trace(repay):
    # First, an ordinary MASK establishes its input ID. Next, token 5 at
    # position 1 is revoked to MASK but the historical writer records no debt.
    rows = [[0], [1], [1 if repay else 2]]
    inputs = [[9], [5], [9]]
    events = []
    for i, rr in enumerate(rows):
        events.append(dict(phase="optimus", block=0, absolute_positions=rr,
                           selected_local=rr, selected_masked=[i != 1],
                           edited_positions=rr if i == 1 else [], written_absolute=[],
                           q_tokens=1, dirty_before=[], offer_tick=i, physical_cap=8,
                           visible_kv=rr[-1]+1, cached_prefix_before=0))
    calls = [dict(rows=1, physical_rows=1, batch=1, kv_rows=rr[-1]+1) for rr in rows]
    return dict(calls=calls, events=events,forward_inputs=inputs, prompt_ids=[],
                generation_forwards=3, generation_query_rows=3, generation_cap=2048,
                verified_layers=20, output_tokens=2)


@pytest.mark.parametrize("repay,missed", [(True, 0), (False, 1)])
def test_revocation_audit_distinguishes_actual_reprojection(repay, missed):
    count,_=inspect(trace(repay),0)
    assert (count["revoked"],count["revoked_not_paid_next"]) == (1,missed)
    assert (count["free_mask_Q"],count["free_decoded_Q"],count["idle"]) == (2,1,21)


def test_refuse_inconsistent_trace_width():
    with pytest.raises(ValueError,match="length"):
        aligned([1,2],[1])

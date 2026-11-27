from copy import deepcopy

import pytest

from experiments.t2t.serving_slots.low_commit_ceiling import (
    capped_source,
    capped_work,
    eligible_pair,
    fingerprint,
    gaps,
    macro_tpfs,
    projection,
    source_run,
    trace_work,
)


def head(stage, block=2, offer=8):
    start = 32*block
    phase = "commit" if stage == "commit" else "denoise"
    return dict(phase=phase, stage=stage, store_kv=stage == "commit", block=block,
                context_version=1, q_start=start, q_end=start+32, prefix_kv_length=start,
                visible_kv=start+32, absolute_positions=list(range(start, start+32)), F=1,
                q_tokens=32, actual_q=32, physical_q=32, layer_physical_q=[32]*20,
                approximate_same_block_kv=False, q_budget=offer)


def unchanged():
    return dict(head("verify"), masked_before=0, masked_after=0, edited=0, accepted=0)


def test_only_matching_full_unchanged_verify():
    v, c = unchanged(), head("commit")
    assert eligible_pair(v, c)
    assert not eligible_pair(c, v)
    assert not eligible_pair(v, head("commit", block=3))
    assert not eligible_pair(v, dict(c, context_version=2))
    assert not eligible_pair(v, dict(c, prefix_kv_length=32))
    assert not eligible_pair(v, dict(c, store_kv=False))
    assert not eligible_pair(v, dict(c, absolute_positions=list(range(65, 97))))


@pytest.mark.parametrize("bad", [
    {"edited": 1}, {"masked_before": 1}, {"masked_after": 1}, {"accepted": 1},
    {"masked_before": None}, {"stage": "selected"}, {"physical_q": 16},
    {"layer_physical_q": [32]*19 + [31]}, {"approximate_same_block_kv": True},
    {"absolute_positions": list(reversed(range(64, 96)))},
])
def test_reject_ineligible_verify(bad):
    assert not eligible_pair(dict(unchanged(), **bad), head("commit"))


def record(events):
    f = {stage: sum(e["stage"] == stage for e in events)
         for stage in ("bootstrap", "verify", "commit")}
    return dict(events=events, index=0, output_tokens=31, prefill_rows=32,
                generation_forwards=sum(f.values()), generation_query_rows=32*sum(f.values()),
                generation_physical_padded_rows=32*sum(f.values()), max_q=32,
                forwards_by_phase=f, q_by_phase={k: 32*n for k, n in f.items()})


def test_only_adjacent_commit_is_deleted_and_prefill_is_free():
    prefill = dict(head("bootstrap"), stage="prefill", phase="prefill")
    row = record([prefill, head("bootstrap"), unchanged(), head("commit")])
    work = trace_work(row, 8)
    assert (work["N"], work["F"], work["Q"], work["k"]) == (31, 3, 96, 1)
    bound = projection(work, 8)
    assert (bound["projected_F"], bound["projected_physical_Q"]) == (2, 64)
    assert bound["projected_pooled_tpf"] == 31/2
    assert (bound["projected_calls_over_slot"], bound["projected_rows_over_slot"]) == (2, 48)
    row = record([prefill, unchanged(), head("bootstrap"), head("commit")])
    assert trace_work(row, 8)["k"] == 0
    row["generation_forwards"] += 1
    with pytest.raises(ValueError, match="Inconsistent charged work"):
        trace_work(row, 8)


def test_gap_sign_and_missing_are_distinct():
    assert gaps(110, 90, 100) == (10, -10, -10)
    assert gaps(110, 90, 80) == (30, 10, 12.5)
    assert gaps(110, 90, None) == (None, None, None)


def test_macro_tpf_weights_requests_equally_before_and_after():
    requests = [dict(N=10, F=5, k=1), dict(N=12, F=2, k=0)]
    old, projected = macro_tpfs(requests)
    assert (old, projected) == (4, 4.25)
    assert old != 22/7
    assert projected != 22/6
    with pytest.raises(ValueError, match="Undefined per-request TPF"):
        macro_tpfs([dict(N=10, F=1, k=1)])


def test_incomplete_source_is_not_a_baseline(tmp_path):
    folder = tmp_path / "llada21-optimus-hardcap-endpoints"
    folder.mkdir()
    (folder / "00-optcap_ibs_8.json").write_text("{}")
    assert capped_source(tmp_path, 8, "ibs", [], {}) is None
    (folder / "run.json").write_text('{"policies":["optcap_ibs_8"]}')
    (folder / "complete-optcap_ibs_8.json").write_text('{"policy":"optcap_ibs_8","requests":30}')
    with pytest.raises(ValueError, match="Unsealed source"):
        source_run(folder, "optcap_ibs_8", [], {})


def test_failed_capped_request_is_charged_and_cap_is_checked():
    row = dict(phase="formal", policy="optcap_ibs_8", generation_cap=2048,
               status="generation_failed", prefill_rows=32, generation_forwards=1,
               generation_query_rows=2, generation_physical_padded_rows=2,
               events=[dict(phase="prefill"), dict(phase="optimus", offer_tick=0, physical_cap=8,
                       absolute_positions=[32, 33], candidate_local=[0, 1], written_absolute=[],
                       dirty_before=[], accepted=0, edited=0, block=1, cached_prefix_after=2,
                       q_tokens=2)],
               calls=[dict(batch=1, rows=32, physical_rows=32), dict(batch=1, rows=2, physical_rows=2)])
    assert capped_work(row, 8)["Q"] == 2
    bad = deepcopy(row)
    bad["events"][1]["physical_cap"] = 16
    with pytest.raises(ValueError, match="Physical cap"):
        capped_work(bad, 8)


def test_source_fingerprints_bind_file_identity_and_order():
    assert fingerprint([("a", "x")]) != fingerprint([("b", "x")])
    assert fingerprint([("a", "x"), ("b", "y")]) != fingerprint([("b", "y"), ("a", "x")])

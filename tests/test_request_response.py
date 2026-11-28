import pytest

from bdlm_serving.methods.t2t.request_response import curve, read_fork, recommend_adjacent
from tools.request_response_eval import evaluate


def data(req="one", digest="0", phase="roll", multiplier=1):
    return dict(model="llada21", executor_id="synthetic-test", continuation_q=16,
                request_id=req, state_digest=digest*64, pre_delivered_n=10,
                state=dict(head_mask_rows=8, dirty_rows=1, resident_rows=32, tail_open=False,
                           prev_q=8, recent_m2t=1, recent_t2t=0, epoch=0, phase=phase),
                arms=[dict(q=q, final_n=10+n*multiplier, state_digest=digest*64, correct=False,
                           suffix_decoder_f=2, suffix_encoder_f=1, terminal="truncated")
                      for q, n in zip((4, 8, 16, 32, 64), (3, 6, 6, 3, 9))])


def test_paired_downside_keeps_enc_work_and_quality_veto():
    d = data()
    r = read_fork(d)
    c = curve([r], r.family, d["state"], min_requests=1, allowed_q=[8])
    assert [p["expected_suffix_tpf"] for p in c["points"]] == [1, 2, 2, 1, 3]
    assert [p["q"] for p in c["points"] if p["eligible"]] == [8]
    assert [p["downshift_loss"] for p in c["adjacent"]] == [1, 0, -1, 2]
    assert c["adjacent"][0]["paired_se"] is None


def test_independent_evidence_not_repeated_decisions():
    rows = [data("a"), data("a", "1"), data("b", "2"), data("c", "3", phase="verify")]
    r = read_fork(rows[0])
    assert curve(list(map(read_fork, rows)), r.family, rows[0]["state"],
                 min_requests=2, exclude_request="b")["status"] == "unsupported"
    assert evaluate(rows, min_requests=2)[0]["supported_decisions"] == 0
    report = evaluate(rows, min_requests=1)[0]
    assert report["total_requests"] == 3 and report["supported_requests"] == 2
    assert report["supported_decisions"] == 3
    assert report["state_model"]["marginal_mae"] == 0
    assert report["budget_only_on_same_support"]["adjacent_direction_accuracy"] == 1


@pytest.mark.parametrize("change", ["future", "bad_digest", "missing", "miscount", "unbilled", "double"])
def test_reject_incomparable_forks(change):
    d = data()
    if change == "future":
        d["state"]["eos_step"] = 12
    elif change == "bad_digest":
        d["arms"][1]["state_digest"] = "1"*64
    elif change == "missing":
        d["arms"].pop()
    elif change == "miscount":
        d["arms"][2]["final_n"] = 9
    elif change == "unbilled":
        d["arms"][3].update(suffix_decoder_f=0, suffix_encoder_f=0)
    with pytest.raises(ValueError):
        evaluate([d, d] if change == "double" else [d])


def test_versions_and_continuations_never_mixed():
    a, b, c = data("a"), data("b"), data("c")
    b["executor_id"] = "different"
    c["continuation_q"] = 32
    reports = evaluate([a, b, c], min_requests=1)
    assert len(reports) == 3 and all(r["supported_decisions"] == 0 for r in reports)


def test_switch_guard_both_signs_and_imposed_outer_menu():
    a, b = data(), data("two")
    train = list(map(read_fork, (a, b)))
    c = curve(train, train[0].family, a["state"], min_requests=2, allowed_q=[4, 8, 16, 32, 64])
    assert recommend_adjacent(c, 4, [4, 8])["q"] == 8
    assert recommend_adjacent(c, 32, [16, 32])["q"] == 16
    assert recommend_adjacent(c, 4, [4, 16])["q"] == 4  # cannot skip a tier
    assert recommend_adjacent(c, 32, [4, 8])["q"] is None
    c["adjacent"][0]["paired_se"] = 2.
    assert recommend_adjacent(c, 4, [4, 8])["q"] == 4


def test_unobserved_feature_cannot_authorize_a_switch_or_erase_quality_gate():
    a = data()
    r = read_fork(a)
    unknown = dict(a["state"], recent_m2t=None)
    c = curve([r], r.family, unknown, min_requests=1, allowed_q=[4])
    assert c["status"] == "unsupported" and c["reason"] == "incomplete_state"
    assert recommend_adjacent(c, 4, [4, 8])["q"] == 4
    assert recommend_adjacent(c, 8, [4, 8])["q"] is None
    a["state"] = unknown
    report, = evaluate([a, data("other")], min_requests=1)
    assert report["total_decisions"] == 2 and report["supported_decisions"] == 0

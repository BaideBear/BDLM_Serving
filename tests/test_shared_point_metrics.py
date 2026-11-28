"""Compact fixed-budget freeze/summary contracts; no inference."""
from statistics import mean

import pytest

from experiments.t2t.serving_slots.shared_point_metrics import summarize
from tools.frozen90_six_shards import split


def samples():
    rows = []
    for i in range(90):
        rows.append(dict(index=i, task_id=f"task/{i}", seed=1000 + i,
                         prompt_sha256=f"{i:064x}", dataset=("gsm8k", "humaneval", "math500")[i % 3]))
    return rows


def receipts(model="gemma", offer=64, version="v1"):
    out = []
    for board, shard in split(samples()).items():
        for row in shard:
            i = row["index"]
            f = 1 + i % 3
            q = min(50 + 10 * (i % 3), f * offer)
            out.append(dict(row, board=board, model=model, offer=offer, version=version,
                            status="complete", N=10 + i % 7, F=f, Q=q, decoder_Q=q,
                            decoder_F=f, encoder_F=0, max_decoder_Q=offer, correct=True))
    return out


def test_complete_point_and_cohort_ratios():
    rows = receipts()
    result = summarize(samples(), rows, "gemma", 64, "v1")
    assert (result["requests"], result["complete"], result["failures"], result["correct"]) == (90, 90, 0, 90)
    assert result["cohorts"]["all90"]["requests"] == 90
    assert result["cohorts"]["original30"]["requests"] == 30
    assert result["cohorts"]["new60"]["requests"] == 60
    c = result["cohorts"]["all90"]
    assert c["macro_TPF"] == pytest.approx(mean(r["N"] / r["F"] for r in rows))
    assert c["pooled_TE"] == pytest.approx(sum(r["N"] for r in rows) / sum(r["Q"] for r in rows))
    pooled_tpf = sum(r["N"] for r in rows) / sum(r["F"] for r in rows)
    macro_te = mean(r["N"] / r["Q"] for r in rows)
    assert c["macro_TPF"] != pytest.approx(pooled_tpf)
    assert c["pooled_TE"] != pytest.approx(macro_te)


def test_failure_is_retained_as_incorrect_and_in_denominators():
    rows = receipts()
    rows[0].update(status="valid_algorithm_failure", correct=False, N=0)
    result = summarize(samples(), rows, "gemma", 64, "v1")
    c = result["cohorts"]["all90"]
    assert (result["requests"], result["complete"], result["failures"], result["correct"]) == (90, 89, 1, 89)
    assert c["failures"] == 1 and c["correct"] == 89
    assert c["macro_TPF"] == pytest.approx(mean(r["N"] / r["F"] for r in rows))
    assert c["pooled_TE"] == pytest.approx(sum(r["N"] for r in rows) / sum(r["Q"] for r in rows))
    kept = [r for r in rows if r["status"] == "complete"]
    excluded_tpf = mean(r["N"] / r["F"] for r in kept)
    excluded_te = sum(r["N"] for r in kept) / sum(r["Q"] for r in kept)
    assert c["macro_TPF"] < excluded_tpf
    assert c["pooled_TE"] < excluded_te


@pytest.mark.parametrize("bad", [
    dict(N=-1), dict(N=True), dict(F=0), dict(Q=0), dict(Q=65),
    dict(F=99), dict(max_decoder_Q=65), dict(correct=1),
])
def test_invalid_bills_are_rejected(bad):
    rows = receipts()
    rows[0].update(bad)
    with pytest.raises(ValueError):
        summarize(samples(), rows, "gemma", 64, "v1")


def test_failure_cannot_be_correct_and_status_is_checked():
    rows = receipts()
    rows[0].update(status="valid_algorithm_failure", correct=True)
    with pytest.raises(ValueError):
        summarize(samples(), rows, "gemma", 64, "v1")
    rows = receipts()
    rows[0]["status"] = "infrastructure_error"
    with pytest.raises(ValueError):
        summarize(samples(), rows, "gemma", 64, "v1")


def test_q_is_decoder_only_and_encoder_cannot_mask_width():
    rows = receipts("gemma", 32, "v1")
    rows[0].update(F=2, decoder_F=1, encoder_F=1, Q=64, decoder_Q=64, max_decoder_Q=32)
    with pytest.raises(ValueError):
        summarize(samples(), rows, "gemma", 32, "v1")


def test_llada_requires_zero_postprefill_encoder():
    rows = receipts("llada21", 16, "v1")
    summarize(samples(), rows, "llada21", 16, "v1")
    rows[0].update(F=3, decoder_F=2, encoder_F=1)
    with pytest.raises(ValueError):
        summarize(samples(), rows, "llada21", 16, "v1")


def test_generation_cap_and_failure_zero_output():
    rows = receipts()
    rows[0]["N"] = 2048
    summarize(samples(), rows, "gemma", 64, "v1")
    rows[0]["N"] = 2049
    with pytest.raises(ValueError):
        summarize(samples(), rows, "gemma", 64, "v1")
    rows = receipts()
    rows[0].update(status="valid_algorithm_failure", correct=False, N=1)
    with pytest.raises(ValueError):
        summarize(samples(), rows, "gemma", 64, "v1")


def test_peak_and_actual_decoder_q_consistency():
    rows = receipts()
    rows[0]["decoder_Q"] += 1
    with pytest.raises(ValueError):
        summarize(samples(), rows, "gemma", 64, "v1")
    rows = receipts()
    rows[0]["max_decoder_Q"] = 1
    with pytest.raises(ValueError):
        summarize(samples(), rows, "gemma", 64, "v1")


def test_generator_receipts_are_materialized():
    rows = receipts()
    result = summarize(samples(), iter(rows), "gemma", 64, "v1")
    assert result["requests"] == 90


def test_missing_duplicate_and_unregistered_point_are_rejected():
    rows = receipts()
    with pytest.raises(ValueError):
        summarize(samples(), rows[:-1], "gemma", 64, "v1")
    with pytest.raises(ValueError):
        summarize(samples(), rows[:-1] + rows[:1], "gemma", 64, "v1")
    with pytest.raises(ValueError):
        summarize(samples(), rows, "gemma", 12, "v1")
    with pytest.raises(ValueError):
        summarize(samples(), rows, "gemma", 64, "")

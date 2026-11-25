import json
from pathlib import Path

import pytest

from tools.conditional_seat import (
    aggregate,
    build_policy,
    check_manifest,
    main,
    metrics,
    read_csv,
    select_rows,
    verify_base_noedit,
)

RUN_LLADA = Path("/data1/project_data/gx-erqw-20261123/runtime/t2t-native-1125/one-seat-v1-llada21")
CONTROL_LLADA = Path("/data1/project_data/gx-erqw-20261123/runtime/t2t-native-1125/hard-v1-llada21")
RUN_GEMMA = Path("/data1/project_data/gx-erqw-20261123/runtime/t2t-native-1125/one-seat-v2-gemma")
CONTROL_GEMMA = Path("/data1/project_data/gx-erqw-20261123/runtime/t2t-native-1125/hard-v1-gemma")
REAL_RUNS = RUN_LLADA.exists() and CONTROL_LLADA.exists() and RUN_GEMMA.exists() and CONTROL_GEMMA.exists()


def make_row(i, dataset="gsm8k", n=10, f=4, q=40, status="success", truncated=False):
    return dict(index=i, dataset=dataset, task_id=f"{dataset}/{i}", seed=17000 + i,
                output_tokens=n, generation_forwards=f, generation_query_rows=q,
                generation_physical_padded_rows=q, tpf=n / f, token_efficiency=n / q,
                status=status, truncated=truncated, policy="synthetic", events=[])


def test_select_rows_uses_only_the_matching_stratum():
    base = [make_row(0), make_row(1), make_row(2)]
    arm = [make_row(0, n=11), make_row(1, n=12), make_row(2, n=13)]
    strata = ["slack", "mask", "none"]
    anchors = [5, 6, -1]

    backfill = select_rows(base, arm, strata, anchors, "backfill")
    mask = select_rows(base, arm, strata, anchors, "mask")
    every = select_rows(base, arm, strata, anchors, "all")

    assert [x["intervened"] for x in backfill] == [1, 0, 0]
    assert [x["intervened"] for x in mask] == [0, 1, 0]
    assert [x["intervened"] for x in every] == [1, 1, 0]
    assert backfill[0]["row"] is arm[0] and backfill[1]["row"] is base[1]
    assert mask[1]["row"] is arm[1] and mask[2]["row"] is base[2]
    assert every[2]["row"] is base[2]


def test_metrics_preserve_the_published_identities():
    row = make_row(0, n=10, f=4, q=40)
    out = metrics(row, 1)
    assert out["N"] == 10 and out["F"] == 4 and out["Q"] == 40
    assert out["tpf"] == pytest.approx(2.5)
    assert out["token_efficiency"] == pytest.approx(0.25)
    assert out["q_per_f"] == pytest.approx(10.0)
    row["tpf"] = 99.0
    with pytest.raises(ValueError):
        metrics(row, 1)


def test_aggregate_reports_net_and_win_loss():
    selected = [
        dict(index=0, dataset="gsm8k", stratum="slack", intervened=1, correct=1,
             N=10, F=4, Q=40, physical_Q=40,
             row=dict(status="success", truncated=False)),
        dict(index=1, dataset="gsm8k", stratum="mask", intervened=0, correct=0,
             N=20, F=5, Q=100, physical_Q=100,
             row=dict(status="success", truncated=False)),
    ]
    reference = [
        dict(index=0, dataset="gsm8k", stratum="slack", correct=0,
             N=8, F=4, Q=32, physical_Q=32,
             row=dict(status="success", truncated=False)),
        dict(index=1, dataset="gsm8k", stratum="mask", correct=1,
             N=20, F=5, Q=100, physical_Q=100,
             row=dict(status="success", truncated=False)),
    ]
    out = aggregate(selected, reference)
    assert out["requests"] == 2 and out["anchors"] == 2
    assert out["correct"] == 1 and out["noedit_correct"] == 1
    assert out["delta_correct"] == 0
    assert out["wins"] == 1 and out["losses"] == 1
    assert out["N"] == 30 and out["F"] == 9 and out["Q"] == 140
    assert out["tpf"] == pytest.approx((10 / 4 + 20 / 5) / 2)
    assert out["token_efficiency"] == pytest.approx(30 / 140)
    assert out["q_per_f"] == pytest.approx(140 / 9)
    assert aggregate(selected, reference, "math500")["requests"] == 0


def test_check_manifest_rejects_a_changed_question(tmp_path):
    datasets = ["gsm8k"] * 10 + ["math500"] * 10 + ["humaneval"] * 10
    samples = [dict(index=i, dataset=datasets[i], task_id=f"t/{i}", seed=17000 + i)
               for i in range(30)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(samples=samples)))
    rows = [dict(index=s["index"], dataset=s["dataset"], task_id=s["task_id"],
                 seed=s["seed"]) for s in samples]
    assert check_manifest(manifest, [("base", rows)])["requests"] == 30
    rows[0]["task_id"] = "changed"
    with pytest.raises(ValueError):
        check_manifest(manifest, [("base", rows)])


def test_verify_base_noedit_allows_only_diagnostic_event_keys():
    common = dict(index=0, dataset="gsm8k", task_id="gsm8k/0", seed=17000,
                  output_tokens=10, generation_forwards=4, generation_query_rows=40,
                  generation_physical_padded_rows=40, forwards_by_phase={}, q_by_phase={},
                  status="success")
    base = {**common, "policy": "base",
            "events": [dict(stage="selected", x=1, intervention={})]}
    noedit = {**common, "policy": "noedit",
              "events": [dict(stage="selected", x=1)]}
    marks = {"0": 1}
    assert verify_base_noedit([base], [noedit], marks, marks) == 1

    changed = {**noedit, "events": [dict(stage="selected", x=2)]}
    with pytest.raises(ValueError):
        verify_base_noedit([base], [changed], marks, marks)
    with pytest.raises(ValueError):
        verify_base_noedit([base], [noedit], marks, {"0": 0})


def test_build_policy_noedit_and_mask_condition():
    base = [make_row(0), make_row(1), make_row(2)]
    arm = [make_row(0, n=11), make_row(1, n=12), make_row(2, n=13)]
    noedit = [make_row(0), make_row(1), make_row(2)]
    marks = {"base": {"0": 1, "1": 0, "2": 1},
             "rank": {"0": 1, "1": 1, "2": 1},
             "random": {"0": 1, "1": 1, "2": 1},
             "noedit": {"0": 1, "1": 0, "2": 1}}
    strata = ["slack", "mask", "none"]
    anchors = [5, 6, -1]

    neutral = build_policy("noedit", None, None, base, base, noedit, strata, anchors, marks)
    assert all(x["source_arm"] == "noedit" and x["intervened"] == 0 for x in neutral)

    masked = build_policy("cond_mask_rank", "rank", "mask", base, arm, noedit,
                          strata, anchors, marks)
    assert [x["intervened"] for x in masked] == [0, 1, 0]
    assert masked[0]["row"] is base[0] and masked[0]["correct"] == 1
    assert masked[1]["row"] is arm[1] and masked[1]["correct"] == 1
    assert masked[2]["row"] is base[2]


@pytest.mark.skipif(not REAL_RUNS, reason="n164 one-seat/noedit runs are not available")
def test_real_runs_splice_and_checks(tmp_path):
    main(["--out", str(tmp_path)])
    checks = read_csv(tmp_path / "conditional-checks.csv")
    assert checks and all(int(row["passed"]) for row in checks)
    summary = read_csv(tmp_path / "conditional-summary.csv")
    by_key = {(row["model"], row["policy"], row["dataset"]): row for row in summary}

    llada_mask = by_key[("llada21", "cond_mask_rank", "all")]
    llada_noedit = by_key[("llada21", "noedit", "all")]
    assert llada_mask["correct"] == llada_noedit["correct"] == "25"
    assert llada_mask["N"] == llada_noedit["N"]
    assert llada_mask["F"] == llada_noedit["F"]
    assert llada_mask["Q"] == llada_noedit["Q"]

    gemma_mask = by_key[("gemma", "cond_mask_rank", "all")]
    gemma_backfill = by_key[("gemma", "cond_backfill_rank", "all")]
    gemma_uncond = by_key[("gemma", "uncond_rank", "all")]
    assert gemma_mask["correct"] == "29" and gemma_mask["delta_correct"] == "1"
    assert gemma_mask["wins"] == "1" and gemma_mask["losses"] == "0"
    assert gemma_backfill["correct"] == "27" and gemma_backfill["delta_correct"] == "-1"
    assert gemma_backfill["wins"] == "0" and gemma_backfill["losses"] == "1"
    assert gemma_uncond["correct"] == "28" and gemma_uncond["delta_correct"] == "0"
    assert gemma_uncond["wins"] == "1" and gemma_uncond["losses"] == "1"

import csv
import json
from pathlib import Path

import pytest

from experiments.t2t.prefix_window import exposure


DATASETS = ("gsm8k", "math500", "humaneval")
ZERO8 = "t2t_q8_f0_m8_hard_prefix_zero"
SWAP8 = "t2t_q8_f0_m6_hard_prefix_frontier_right"
SLACK8 = "t2t_q8_f0_m6_hard_prefix_slack_random"


def gemma(policy, *, budget, quota, fresh, mask, backfill, eligible,
          added=(), evicted=(), block=48, clock=47):
    fresh, mask, backfill = tuple(fresh), tuple(mask), tuple(backfill)
    eligible, added, evicted = tuple(eligible), tuple(added), tuple(evicted)
    zero = tuple(sorted(set(fresh) | set(mask) | set(backfill)))
    selected = tuple(sorted((set(zero) - set(evicted)) | set(added)))
    end = min(mask + backfill) if mask or backfill else max(fresh) + 1
    plan = dict(block_start=block, clock=clock, prefix_end=end, budget=budget,
                quota=quota, fresh_rows=list(fresh),
                eligible_prefix_rows=list(eligible),
                first_mask=min(mask) if mask else None,
                zero_rows=list(zero), zero_mask_rows=list(mask),
                zero_backfill_rows=list(backfill), added_rows=list(added),
                evicted_rows=list(evicted), selected_rows=list(selected),
                zero_width=len(zero), actual_width=len(selected),
                underfill=max(0, budget - len(selected)),
                overflow=max(0, len(selected) - budget))
    return dict(stage="selected", block=block, clock=clock, q=len(selected),
                positions=list(selected), prefix_plan=plan)


def llada(policy=ZERO8, *, budget=8, quota=0, fresh=(26, 27, 29),
          mask=(28, 30, 31), backfill=(), eligible=(), added=(), evicted=(),
          block=1, clock=1):
    fresh, mask, backfill = tuple(fresh), tuple(mask), tuple(backfill)
    eligible, added, evicted = tuple(eligible), tuple(added), tuple(evicted)
    zero = tuple(sorted(set(fresh) | set(mask) | set(backfill)))
    selected = tuple(sorted((set(zero) - set(evicted)) | set(added)))
    end = min(mask + backfill) if mask or backfill else max(fresh) + 1
    start = block * 32
    plan = dict(block_start=start, clock=clock, prefix_end=end, budget=budget,
                quota=quota, fresh_rows=list(fresh),
                eligible_prefix_rows=list(eligible),
                first_mask=min(mask) if mask else None,
                zero_rows=list(zero), zero_mask_rows=list(mask),
                zero_backfill_rows=list(backfill), added_rows=list(added),
                evicted_rows=list(evicted), selected_rows=list(selected),
                zero_width=len(zero), actual_width=len(selected),
                underfill=max(0, budget - len(selected)),
                overflow=max(0, len(selected) - budget))
    return dict(stage="selected", block=block, clock=clock + 1,
                step_in_block=clock + 1, actual_q=len(selected),
                physical_q=len(selected), q_budget=budget,
                absolute_positions=[start + row for row in selected],
                prefix_plan=plan)


def test_gemma_endogenous_zero_metrics():
    under = gemma(ZERO8, budget=8, quota=0, fresh=(0, 1), mask=(6,),
                  backfill=(), eligible=(2, 3, 4, 5))
    view = exposure.classify("gemma", 256, ZERO8, 0, under)
    assert view["kind"] == "zero"
    assert view["slack"] == 5 and view["pool"] == 4
    assert view["opp1"] == 1 and view["opp4"] == 4
    assert view["fresh_overflow"] == 0

    overflow = gemma("t2t_q4_f0_m4_hard_prefix_zero", budget=4, quota=0,
                     fresh=(0, 1, 2, 3, 4, 5), mask=(), backfill=(),
                     eligible=())
    view = exposure.classify("gemma", 256, "t2t_q4_f0_m4_hard_prefix_zero",
                             0, overflow)
    assert view["kind"] == "zero"
    assert view["fresh_overflow"] == 2 and view["slack"] == 0
    assert view["pool"] == 0 and view["opp1"] == view["opp4"] == 0


def test_gemma_swap_slack_and_noop():
    swap = gemma(SWAP8, budget=8, quota=2, fresh=(0,), mask=(6, 7),
                 backfill=(8,), eligible=(1, 2, 3, 4), added=(1, 2),
                 evicted=(7, 8))
    view = exposure.classify("gemma", 256, SWAP8, 0, swap)
    assert view["kind"] == "swap"
    assert view["added"] == 2
    assert view["mask"] == 1 and view["ordinary"] == 1
    assert view["saturated"] and not view["noop"]

    slack = gemma(SLACK8, budget=8, quota=2, fresh=(0,), mask=(6, 7),
                  backfill=(8,), eligible=(1, 2, 3, 4), added=(1, 2))
    view = exposure.classify("gemma", 256, SLACK8, 0, slack)
    assert view["kind"] == "slack"
    assert view["added"] == 2 and view["mask"] == view["ordinary"] == 0
    assert view["slack"] == 4 and view["opp4"] == 4

    noop = gemma(SWAP8, budget=8, quota=2, fresh=(0,), mask=(6, 7),
                 backfill=(8,), eligible=(1, 2, 3, 4))
    view = exposure.classify("gemma", 256, SWAP8, 0, noop)
    assert view["kind"] == "noop" and view["noop"] and not view["saturated"]


def test_opportunity_is_min_k_slack_pool():
    one_pool = gemma(ZERO8, budget=8, quota=0, fresh=(0,), mask=(6,),
                     backfill=(), eligible=(2,))
    view = exposure.classify("gemma", 256, ZERO8, 0, one_pool)
    assert view["slack"] == 6 and view["pool"] == 1
    assert view["opp1"] == 1 and view["opp4"] == 1
    full = gemma(ZERO8, budget=8, quota=0, fresh=(0, 1), mask=(2, 3, 4, 5, 6, 7),
                 backfill=(), eligible=())
    view = exposure.classify("gemma", 256, ZERO8, 0, full)
    assert view["slack"] == 0 and view["opp1"] == view["opp4"] == 0


def test_llada_schema_and_metrics():
    view = exposure.classify("llada", 32, ZERO8, 0, llada())
    assert view["kind"] == "zero"
    assert view["physical"] == 6 and view["slack"] == 2
    assert view["pool"] == 0 and view["opp1"] == view["opp4"] == 0

    event = llada(fresh=(26,), mask=(28,), eligible=(27,))
    view = exposure.classify("llada", 32, ZERO8, 0, event)
    assert view["slack"] == 6 and view["pool"] == 1
    assert view["opp1"] == 1 and view["opp4"] == 1


def test_fail_closed_on_schema_and_endogenous_mismatch():
    with pytest.raises(ValueError, match="unsupported model"):
        exposure.schema("mystery")
    event = gemma(ZERO8, budget=8, quota=0, fresh=(0,), mask=(6,),
                  backfill=(), eligible=(1, 2))
    event.pop("positions")
    with pytest.raises(ValueError, match="missing Gemma fields"):
        exposure.classify("gemma", 256, ZERO8, 0, event)
    event = gemma(ZERO8, budget=8, quota=0, fresh=(0,), mask=(6,),
                  backfill=(), eligible=(1, 2))
    event["prefix_plan"]["clock"] += 1
    with pytest.raises(ValueError, match="clock mismatch"):
        exposure.classify("gemma", 256, ZERO8, 0, event)
    event = llada()
    event["step_in_block"] += 1
    with pytest.raises(ValueError, match="block/clock mismatch"):
        exposure.classify("llada", 32, ZERO8, 0, event)
    event = gemma("t2t_q8_f0_m6_hard_unsupported", budget=8, quota=2,
                  fresh=(0,), mask=(6, 7), backfill=(), eligible=(1, 2))
    with pytest.raises(ValueError, match="bad policy"):
        exposure.classify("gemma", 256,
                          "t2t_q8_f0_m6_hard_unsupported", 0, event)


def test_load_cases_requires_90_unique_balanced(tmp_path):
    samples = []
    for index in range(90):
        dataset = DATASETS[(index % 30) // 10]
        samples.append(dict(index=index, dataset=dataset,
                            task_id="%s/%d" % (dataset, index),
                            seed=17000 + index))
    path = tmp_path / "samples.json"
    path.write_text(json.dumps({"samples": samples}))
    assert len(exposure.load_cases(path)) == 90

    path = tmp_path / "short.json"
    path.write_text(json.dumps({"samples": samples[:-1]}))
    with pytest.raises(ValueError, match="exactly 90"):
        exposure.load_cases(path)
    bad = json.loads(json.dumps({"samples": samples}))
    bad["samples"][1]["task_id"] = bad["samples"][0]["task_id"]
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="unique task_id"):
        exposure.load_cases(path)
    bad = json.loads(json.dumps({"samples": samples}))
    bad["samples"][0]["dataset"] = "math500"
    path = tmp_path / "unbalanced.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="30 per dataset"):
        exposure.load_cases(path)


def _write_run(tmp_path):
    samples = []
    for index in range(90):
        dataset = DATASETS[(index % 30) // 10]
        samples.append(dict(index=index, dataset=dataset,
                            task_id="%s/%d" % (dataset, index),
                            seed=17000 + index))
    samples_path = tmp_path / "samples.json"
    samples_path.write_text(json.dumps({"samples": samples}))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps(dict(
        model="gemma", block=256, samples_sha256=exposure.sha256(samples_path),
        policies=[ZERO8, SWAP8])))
    for index, case in enumerate(samples):
        zero = gemma(ZERO8, budget=8, quota=0, fresh=(0,), mask=(6, 7),
                     backfill=(8,), eligible=(1, 2, 3, 4))
        swap = gemma(SWAP8, budget=8, quota=2, fresh=(0,), mask=(6, 7),
                     backfill=(8,), eligible=(1, 2, 3, 4), added=(1, 2),
                     evicted=(7, 8))
        for policy, event in ((ZERO8, zero), (SWAP8, swap)):
            payload = dict(index=index, dataset=case["dataset"],
                           task_id=case["task_id"], seed=case["seed"],
                           policy=policy, status="success", model="gemma",
                           events=[event])
            (run_dir / ("%02d-%s.json" % (index, policy))).write_text(
                json.dumps(payload))
    for policy in (ZERO8, SWAP8):
        (run_dir / ("complete-%s.json" % policy)).write_text(
            json.dumps(dict(policy=policy, requests=90)))
    return samples_path, run_dir


def test_build_writes_25_column_aggregate(tmp_path):
    samples_path, run_dir = _write_run(tmp_path)
    out = tmp_path / "exposure.csv"
    rows, run, kind_schema, width, digest = exposure.build(
        run_dir, samples_path, out)
    assert kind_schema == "gemma" and width == 256
    assert digest == exposure.sha256(samples_path)
    assert len(rows) == 6 and len(exposure.FIELDS) == 25
    assert b"\r" not in out.read_bytes()
    with out.open(newline="") as stream:
        table = list(csv.DictReader(stream))
    assert list(table[0]) == list(exposure.FIELDS)
    indexed = {(row["policy"], row["cohort"]): row for row in table}
    zero = indexed[(ZERO8, "all90")]
    assert int(zero["requests"]) == 90 and int(zero["events"]) == 90
    assert float(zero["mean_q_rows"]) == 4.0
    assert int(zero["positive_slack_events"]) == 90
    assert int(zero["positive_slack_requests"]) == 90
    assert int(zero["slack_rows"]) == 360
    assert int(zero["eligible_rows"]) == 360
    assert int(zero["opp_k1_rows"]) == 90
    assert int(zero["opp_k4_rows"]) == 360
    assert int(zero["untouched_requests"]) == 90
    swap = indexed[(SWAP8, "all90")]
    assert int(swap["added_rows"]) == 180
    assert int(swap["mask_displaced"]) == 90
    assert int(swap["ordinary_displaced"]) == 90
    assert int(swap["nominal_k_rows"]) == 180
    assert int(swap["saturated_events"]) == 90
    assert int(swap["noop_events"]) == 0
    assert int(swap["untouched_requests"]) == 0
    assert int(swap["opp_k4_rows"]) == 360
    assert int(indexed[(ZERO8, "original30")]["requests"]) == 30
    assert int(indexed[(ZERO8, "added60")]["requests"]) == 60

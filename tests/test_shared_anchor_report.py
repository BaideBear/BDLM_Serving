import json

import pytest

from experiments.t2t.serving_slots.shared_fixed_shard import identity
from tools.frozen90_six_shards import split
from tools.shared_anchor_report import audit, costs


def fixture(root, *, count=90):
    samples = [dict(index=i, task_id=str(i), seed=100+i, prompt_sha256=f"{i:064x}",
                    dataset=("gsm8k", "math500", "humaneval")[i % 3]) for i in range(90)]
    reg = {"stage": "formal90"}
    for board, shard in split(samples).items():
        folder = root / board.replace("n164:", "").replace("A800:", "A")
        folder.mkdir()
        with (folder/"gemma.ndjson").open("w") as f:
            for row in shard:
                if row["index"] >= count:
                    continue
                r = dict(row, board=board, model="gemma", offer=32, stage="formal90",
                         version=identity(reg), status="complete", correct=True,
                         N=8, F=3, Q=24, decoder_F=2, encoder_F=1,
                         max_decoder_Q=16, decoder_Q=24,
                         roles=dict(mask_Q=12, dirty_Q=2, cold_Q=8, prefix_Q=2, sample_Q=16, sample_F=1,
                                    verify_Q=8, verify_F=1, peak_Q=16))
                f.write(json.dumps(r)+"\n")
    return samples, reg


def test_closed_point_with_disjoint_charge(tmp_path):
    samples, reg = fixture(tmp_path)
    report = audit(tmp_path, samples, reg)
    assert report["seen"] == 90
    point, = [p for p in report["points"] if "cohorts" in p]
    assert point["cohorts"]["all90"]["correct"] == 90
    assert point["physical_Q_by_role"] == dict(dirty=180, cold_mask=720, warm_mask=360, prefix=180,
                                               terminal_verification=720, rebuild=0)


def test_partial_has_no_scores_and_unfinished_write_is_deferred(tmp_path):
    samples, reg = fixture(tmp_path, count=89)
    with (tmp_path/"H4"/"gemma.ndjson").open("ab") as f:
        f.write(b'{"stage":')
    report = audit(tmp_path, samples, reg)
    assert len(report["deferred_incomplete_lines"]) == 1 and report["seen"] == 89
    assert all("cohorts" not in p for p in report["points"])


@pytest.mark.parametrize("bad", ["duplicate", "dataset", "overcharge", "peak", "role", "wrongstage"])
def test_malformed_receipts_fail(tmp_path, bad):
    samples, reg = fixture(tmp_path)
    path = tmp_path/"H0"/"gemma.ndjson"
    lines = path.read_text().splitlines()
    r = json.loads(lines[0])
    if bad == "duplicate":
        lines.append(lines[0])
    elif bad == "dataset":
        r["dataset"] = "wrong"
    elif bad == "overcharge":
        r["max_decoder_Q"] = 33
    elif bad == "peak":
        del r["max_decoder_Q"]
    elif bad == "role":
        r["roles"]["cold_Q"] = 13
    else:
        r["stage"] = "screen18"
    lines[0] = json.dumps(r)
    path.write_text("\n".join(lines)+"\n")
    with pytest.raises(ValueError):
        audit(tmp_path, samples, reg)


def test_llada_role_partition():
    row = dict(model="llada21", decoder_Q=10, roles=dict(dirty_write=2, uninitialized_mask=3,
                                                         resident_mask=4, clean_resolved=1))
    assert costs(row)["warm_mask"] == 4
    row["decoder_Q"] -= 1
    with pytest.raises(ValueError):
        costs(row)

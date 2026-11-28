from fractions import Fraction as Fr
import json

import pytest

from experiments.t2t.serving_slots.shared_fixed_shard import identity, shard_cases
from tools.frozen90_six_shards import BOARDS, OFFERS
from tools.shared_screen_report import audit, choose
from tools.shared_screen_sweep import CHOICES, build


def fixture(root, exclude=None):
    base = dict(stage="formal90", cap=2048,
                params={m: dict(open_at="1/4", prefix_share="1/8") for m in OFFERS})
    panel = build(base, "prefix")
    samples = [dict(index=i, task_id=f"id:{i}", seed=17000+i, prompt_sha256=f"{i:064x}",
                    dataset=("gsm8k", "humaneval", "math500")[i % 3]) for i in range(90)]
    for board in BOARDS:
        folder = root / board.replace("n164:", "").replace("A800:", "A")
        folder.mkdir()
        for model, offers in OFFERS.items():
            with (folder / f"{model}.ndjson").open("w") as f:
                for r in shard_cases(samples, board, "screen18"):
                    for q in offers:
                        for d in CHOICES:
                            if (model, q, r["index"], d) == exclude:
                                continue
                            roles = (dict(mask_Q=q, sample_Q=q, sample_F=1, peak_Q=q) if model=="gemma"
                                     else dict(dirty_write=0, resident_mask=0, uninitialized_mask=q, clean_resolved=0))
                            ok = not (model=="llada21" and d=="1/2" and q==64 and r["index"]==0)
                            row = dict(r, model=model, max_decoder_Q=q, F=1, decoder_F=1, encoder_F=0, encoder_Q=0,
                                       Q=q, decoder_Q=q, N=4+int(d=="1/2"), status="complete", correct=ok,
                                       source=identity(base), version=identity(panel), stage="screen18", board=board,
                                       dose=d, params=panel["arms"][model][d], axis="prefix", offer=q, roles=roles)
                            f.write(json.dumps(row)+"\n")
    return samples, base, panel


def test_release_independently_and_use_paired_quality_gate(tmp_path):
    args = fixture(tmp_path)
    report = audit(tmp_path, *args)
    assert report["seen"] == 720
    ll, ge = (report["models"][m] for m in OFFERS)
    assert ll["decision"]["selected"] == "1/8" and ge["decision"]["selected"] == "1/2"
    assert ll["offers_by_dose"]["1/2"][64]["lost_correct"] == 1
    assert ll["offers_by_dose"]["1/2"][64]["mean_actual_Q"] == 64
    point = ge["offers_by_dose"]["1/2"][32]
    assert (point["correct"], point["macro_TPF"], point["pooled_TE"]) == (18, 5, 5/32)


def test_missing_or_truncated_model_does_not_release(tmp_path):
    args = fixture(tmp_path, exclude=("gemma", 64, 0, "1/4"))
    rep = audit(tmp_path, *args)
    assert rep["seen"] == 719 and "decision" not in rep["models"]["gemma"]
    assert "decision" in rep["models"]["llada21"]
    with (tmp_path/"H0"/"llada21.ndjson").open("ab") as f:
        f.write(b"{")
    assert "decision" not in audit(tmp_path, *args)["models"]["llada21"]


def test_control_replay_checks_each_pair_not_aggregate(tmp_path):
    args = fixture(tmp_path)
    root = tmp_path / "formal"
    folder = root / "H0"
    folder.mkdir(parents=True)
    src = (tmp_path / "H0" / "gemma.ndjson").read_text().splitlines()
    controls = [r for line in src if (r := json.loads(line))["dose"] == "1/8"]
    for r in controls:
        r.update(stage="formal90", version=identity(args[1]))
    controls[0]["N"] += 1
    controls[1]["N"] -= 1
    (folder / "gemma.ndjson").write_text("".join(json.dumps(r)+"\n" for r in controls))
    replay = audit(tmp_path, *args, root)["models"]["gemma"]["anchor_replay"]
    assert replay["expected"] == 90 and replay["matched"] == 15
    assert replay["discrepancies"] == [dict(index=r["index"],offer=r["offer"],fields=["N"])
                                      for r in controls[:2]]


@pytest.mark.parametrize("corrupt", ["duplicate", "seed", "params", "version", "source", "encoder_Q", "prefix"])
def test_reject_misattribution(tmp_path, corrupt):
    args = fixture(tmp_path)
    path = tmp_path/"H0"/"gemma.ndjson"
    lines = path.read_text().splitlines()
    row = json.loads(lines[0])
    if corrupt == "duplicate":
        lines.append(lines[0])
    elif corrupt == "prefix":
        row["roles"].update(prefix_Q=1, mask_Q=31)
    else:
        row[corrupt] = 30
    lines[0] = json.dumps(row)
    path.write_text("\n".join(lines)+"\n")
    with pytest.raises(ValueError, match="gemma.ndjson:"):
        audit(tmp_path, *args)


def test_exact_tpf_ordering_and_tie_break():
    offers = OFFERS["llada21"]
    control = {q:(18,Fr(1)) for q in offers}
    scores = {d:control.copy() for d in CHOICES}
    for d in ("0","1/2"):
        scores[d][4] = (18,Fr(1)+Fr(1,10**20))
    assert choose(scores,"1/8")["selected"] == "0"
    scores["1/2"][4] = (18,Fr(1)+Fr(2,10**20))
    assert choose(scores,"1/8")["selected"] == "1/2"
    # Worst tier precedes the geometric mean: never let a single stellar tier decide.
    scores["1/2"] = {q:(18,Fr(2)) for q in offers}
    scores["0"][4] = (18,Fr(1000))
    assert choose(scores,"1/8")["selected"] == "1/2"

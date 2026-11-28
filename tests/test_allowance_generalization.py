import json
import pytest

from tools.audit_allowance_generalization import digest, load_stage


def stage(root, rows):
    receipts, reg = root/"receipts.ndjson", root/"registration.json"
    reg.write_text("{}")
    receipts.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root/"offline-audit.json").write_text(json.dumps({
        "status": "pass", "sha256": {p.name: digest(p) for p in (receipts, reg)}}))


def note(i, **fields):
    bill = {"correct": True, "N": 9, "F": 4, "Q": 17, "failure": None, **fields}
    return {"index": i, "arms": {a: bill for a in ("stock","apply","capped")}}


def test_recovers_with_billed_failed_case(tmp_path):
    stage(tmp_path, [note(7), note(13, correct=False, N=0, failure="no convergence")])
    rows, _ = load_stage(tmp_path, {7,13})
    assert rows[13]["capped"]["Q"] == 17


@pytest.mark.parametrize("rows", [
    [note(7),note(7)], [note(7)], [note(7),note(13,F=0)],
    [note(7),note(13,correct=False,failure="no convergence")],
])
def test_rejects_mixed_or_invalid_bills(tmp_path, rows):
    stage(tmp_path, rows)
    with pytest.raises(ValueError):
        load_stage(tmp_path, {7,13})


def test_stale_seal_is_not_a_completed_run(tmp_path):
    stage(tmp_path, [note(7)])
    (tmp_path/"receipts.ndjson").write_text("")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_stage(tmp_path, {7})

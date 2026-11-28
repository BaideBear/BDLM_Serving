from hashlib import sha256
import json

import pytest

from experiments.t2t.serving_slots import fixed90_gate as gate
from tools import fixed90_q4_recovery as recovery


def worker_cases():
    samples = [{"seed": 17000 + i} for i in range(90)]
    starts = [dict(kind="case_start", index=i, seed=row["seed"]) for i, row in enumerate(samples)]
    reused = [dict(kind="prior_reused", index=i) for i in range(21)]
    return starts[:21], reused + starts[21:], samples


@pytest.mark.parametrize("bad", [None, "duplicate", "missing", "seed", "rerun"])
def test_worker_partition(bad):
    first, resume, samples = worker_cases()
    if bad == "duplicate":
        first.append(first[0])
    elif bad == "missing":
        resume.pop()
    elif bad == "seed":
        resume[-1]["seed"] += 1
    elif bad == "rerun":
        resume.append(first[20])
    if bad is None:
        assert set(recovery.partition(first, resume, samples)) == set(range(21))
    else:
        with pytest.raises(ValueError, match="coverage|seed drift"):
            recovery.partition(first, resume, samples)


def test_worker_rejects_source_drift():
    receipt = dict(transport="registered", id=recovery.FIRST, offer=4, indices=90, mode="run",
                   runner_sha256=recovery.RUNNER, bundle_sha256=recovery.BUNDLE, pid=1)
    events = [receipt, dict(kind="worker", pid=1, uuid=recovery.DEVICE),
              dict(transport="checkpoint_verified", fingerprint=recovery.CHECKPOINT,
                   scorer=recovery.SCORER)]
    recovery.verify_worker(events, recovery.FIRST, [], finished=False)
    receipt["runner_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source"):
        recovery.verify_worker(events, recovery.FIRST, [], finished=False)


def test_transport_width_sorting_only_for_fresh_seals():
    row = dict(widths={"2": 3, "32": 4, "4": 5}, N=13)
    body = (recovery.packed(row) + "\n").encode()
    original = dict(row, widths={int(k): v for k, v in row["widths"].items()})
    producer = sha256((recovery.packed(original) + "\n").encode()).hexdigest()
    assert recovery.signed_seal(body, row, producer, new_seal=True) == sha256(body).hexdigest()
    with pytest.raises(ValueError, match="transport"):
        recovery.signed_seal(body, row, producer, new_seal=False)
    row["N"] = 14
    with pytest.raises(ValueError, match="producer"):
        recovery.signed_seal((recovery.packed(row) + "\n").encode(), row, producer, new_seal=True)


def test_copied_recovery_keeps_first_attempt_and_scoring_origin():
    row = dict(raw=f"{recovery.FIRST}/case-20.ndjson", recovered_from_retirement_tape=True)
    claim = dict(receipt_sha256=recovery.RECOVERED_SEAL)
    recovery.check_origin(row, 20, recovery.FIRST, claim, recovery.RECOVERED_SEAL)
    with pytest.raises(ValueError, match="generation attempt"):
        recovery.check_origin(row, 20, recovery.RESUME, claim, recovery.RECOVERED_SEAL)
    with pytest.raises(ValueError, match="scoring origin"):
        recovery.check_origin(dict(row, recovered_from_retirement_tape=False), 20,
                              recovery.FIRST, claim, recovery.RECOVERED_SEAL)
    with pytest.raises(ValueError, match="reuse claim"):
        recovery.check_origin(row, 20, recovery.FIRST, claim, "0" * 64)


def test_archived_q4_passes_gate_without_claiming_a_hard_cap():
    doc = json.loads((gate.ROOT / "llada-full-admission-q4-frozen90.json").read_bytes())
    q4 = next(cell for cell in gate.evaluate([doc], lock=recovery.LOCK)["tiers"]
              if (cell["model"], cell["q"]) == ("llada21", 4))
    assert q4["status"] == "ready_to_compare"
    assert q4["three_metrics_complete"]
    assert q4["candidate_hard_cap_observed"] is False
    full = q4["cohorts"]["frozen90"]
    assert (full["n"], full["correct"], full["N"], full["F"], full["history_Q"], full["peak"]) == (
        90, 68, 46182, 27307, 144097, 32)
    assert q4["cohorts"]["original30"]["correct"] == 20
    assert q4["cohorts"]["new60"]["correct"] == 48
    cases = doc["cases"]
    assert all(cases[str(i)]["attempt"] == recovery.FIRST for i in range(21))
    assert all(cases[str(i)]["attempt"] == recovery.RESUME for i in range(21, 90))
    assert cases["20"]["source_seal"] == "recovered-20.json"
    assert cases["20"]["ids_sha256"] == recovery.RECOVERED_IDS
    assert cases["20"]["correct"] is True
    assert doc["provenance"]["recovered_20"]["scoring_host"] == "n164"

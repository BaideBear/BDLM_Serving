"""Classify and compact A0 prefix-seat receipts without inference.

A failed instrumentation attempt is deliberately reported as invalid instead of
discarded silently.  ``--require-valid`` is the launch/reporting gate: it exits
nonzero unless every registered case has complete stock/observer/seat metrics.
No output token IDs are copied into this compact view.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Optional

SIX = (7, 13, 25, 33, 55, 71)
ARMS = ("stock", "observer", "seat")
_METRIC_KEYS = ("N", "correct", "scoring", "rng_after", "output_sha256")
_BILL_KEYS = ("F", "Q", "encoder_Q", "peak_decoder_Q", "admissions", "paid_rows")


def _digest(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(arm):
    if not isinstance(arm, dict):
        return {}
    bill = arm.get("bill") if isinstance(arm.get("bill"), dict) else {}
    out = {key: arm.get(key) for key in _METRIC_KEYS}
    out.update({key: bill.get(key) for key in _BILL_KEYS})
    return out


def _complete(metrics):
    return (isinstance(metrics.get("N"), int)
            and all(isinstance(metrics.get(key), int) for key in _BILL_KEYS)
            and isinstance(metrics.get("correct"), bool)
            and isinstance(metrics.get("scoring"), str)
            and isinstance(metrics.get("rng_after"), str)
            and isinstance(metrics.get("output_sha256"), str))


def _case_valid(record, index):
    reasons = []
    if record.get("index") != index:
        reasons.append("index mismatch")
    if record.get("error") is not None:
        reasons.append(f"case error: {record.get('error')}")
    if record.get("stock_observer_identical") is not True:
        reasons.append("stock/observer identity flag false")
    arms = record.get("arms") if isinstance(record.get("arms"), dict) else {}
    if set(arms) != set(ARMS):
        reasons.append("incomplete arm set")
        return False, reasons, {}, None
    metrics = {arm: _metrics(arms.get(arm)) for arm in ARMS}
    bad = [arm for arm in ARMS if not _complete(metrics[arm])]
    if bad:
        reasons.append(f"incomplete arm metrics: {','.join(bad)}")
    if metrics["stock"] != metrics["observer"]:
        reasons.append("stock/observer reported metrics differ")
    seat_record = arms["seat"].get("seat")
    if not isinstance(seat_record, dict):
        seat_record = {"exchanges": [], "blocks_used": []}
        reasons.append("missing seat record")
    exchanges = seat_record.get("exchanges")
    blocks_used = seat_record.get("blocks_used")
    if not isinstance(exchanges, list) or not isinstance(blocks_used, list):
        reasons.append("invalid seat record fields")
        exchanges = exchanges if isinstance(exchanges, list) else []
    fork = record.get("first_fork")
    if exchanges:
        first = exchanges[0]
        if fork != {"kind": "trace", "tick": first.get("tick"),
                    "reference": first.get("before"), "treatment": first.get("after")}:
            # The exact trace objects can differ in serializer details; require
            # only the audited tick, which the worker already checked.
            if not (isinstance(fork, dict) and fork.get("kind") == "trace"
                    and fork.get("tick") == first.get("tick")):
                reasons.append("seat exchange/fork tick mismatch")
    elif fork is not None:
        reasons.append("unexposed case has a fork")
    if not reasons:
        return True, reasons, metrics, seat_record
    return False, reasons, metrics, seat_record


def summarize(root: Path):
    root = Path(root)
    digests = {}
    summary = None
    summary_error = None
    summary_path = root / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_bytes())
        except Exception as exc:
            summary_error = repr(exc)
    else:
        summary_error = "summary.json missing"
    if summary is not None:
        digests[summary_path.name] = _digest(summary_path)
    for index in SIX:
        path = root / f"case-{index:02}.json"
        if path.is_file():
            digests[path.name] = _digest(path)
    if (root / "raw.log").is_file():
        digests["raw.log"] = _digest(root / "raw.log")

    valid_cases, invalid_cases = [], []
    if isinstance(summary, dict):
        expected_cases = summary.get("cases")
        expected_arms = summary.get("arms")
    else:
        expected_cases = expected_arms = None
    header_ok = (expected_cases == list(SIX) and expected_arms == list(ARMS)
                 and summary is not None and summary.get("gate") == "passed"
                 and summary.get("failed") == [])
    for index in SIX:
        path = root / f"case-{index:02}.json"
        try:
            record = json.loads(path.read_bytes())
        except Exception as exc:
            invalid_cases.append(dict(index=index, valid=False,
                                      reasons=[f"case load error: {exc}"]))
            continue
        valid, reasons, metrics, seat_record = _case_valid(record, index)
        row = dict(index=index, task_id=record.get("task_id"),
                   dataset=record.get("dataset"), seed=record.get("seed"),
                   valid=valid, reasons=reasons, first_fork=record.get("first_fork"),
                   exchanges=(seat_record or {}).get("exchanges", []),
                   blocks_used=(seat_record or {}).get("blocks_used", []),
                   **metrics)
        if valid:
            row["seat_delta"] = {key: row["seat"][key] - row["stock"][key]
                                 for key in ("N", "F", "Q")}
            row["correct_delta"] = int(row["seat"]["correct"]) - int(row["stock"]["correct"])
            valid_cases.append(row)
        else:
            invalid_cases.append(row)

    if summary_error is not None:
        classification = "invalid_instrumentation"
        invalid_cases.append(dict(index=None, valid=False,
                                  reasons=[f"summary error: {summary_error}"]))
    elif not header_ok:
        classification = "invalid_instrumentation"
        invalid_cases.append(dict(index=None, valid=False,
                                  reasons=["summary gate/case/arm header invalid"]))
    elif len(valid_cases) == len(SIX):
        classification = "valid"
    elif valid_cases:
        classification = "partial_valid"
    else:
        classification = "invalid_instrumentation"
    return dict(source=root.name, classification=classification,
                gate_launch_ok=classification == "valid", header_ok=header_ok,
                valid_case_count=len(valid_cases), registered_case_count=len(SIX),
                valid_cases=valid_cases, invalid_cases=invalid_cases,
                summary_gate=None if summary is None else summary.get("gate"),
                sha256=digests)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--require-valid", action="store_true",
                        help="exit 2 unless all six case rows are valid")
    args = parser.parse_args()
    report = summarize(args.root)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.require_valid and not report["gate_launch_ok"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

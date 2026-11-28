"""Build the compact prefix-seat aggregate without touching raw receipts."""
import argparse
import json
from pathlib import Path

ARMS = ("stock", "observer", "seat")
SUM_KEYS = ("N", "F", "Q", "encoder_Q", "admissions", "paid_rows", "correct")
MAX_KEYS = ("peak_decoder_Q",)


def build(root: Path):
    reduced = json.loads((root / "reducer-report.json").read_bytes())
    if reduced.get("classification") != "valid" or reduced.get("valid_case_count") != 6:
        raise ValueError("Refusing to aggregate an incomplete or invalid run")
    cases = reduced["valid_cases"]
    aggregate = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for key in SUM_KEYS:
            aggregate[arm][key] = sum(int(case[arm][key]) for case in cases)
        for key in MAX_KEYS:
            aggregate[arm][key] = max(int(case[arm][key]) for case in cases)
        aggregate[arm]["macro_TPF"] = sum(case[arm]["N"] / case[arm]["F"]
                                            for case in cases) / len(cases)
        aggregate[arm]["pooled_TE"] = aggregate[arm]["N"] / aggregate[arm]["Q"]
    seat_cases = []
    for case in cases:
        seat_cases.append({
            "index": case["index"], "exchanges": case["exchanges"],
            "delta": {key: case["seat"][key] - case["stock"][key]
                      for key in (*SUM_KEYS, *MAX_KEYS)},
            "correct_delta": int(case["seat"]["correct"]) - int(case["stock"]["correct"]),
            "stock_output_sha256": case["stock"]["output_sha256"],
            "seat_output_sha256": case["seat"]["output_sha256"],
        })
    return {
        "source": str(root), "classification": reduced["classification"],
        "gate": reduced.get("summary_gate"), "reducer_gate": reduced["gate_launch_ok"],
        "registered_case_count": reduced["registered_case_count"], "arms": list(ARMS),
        "aggregate": aggregate,
        "stock_observer_equal": all(case["stock"]["output_sha256"] == case["observer"]["output_sha256"]
                                    and all(case["stock"][key] == case["observer"][key]
                                            for key in (*SUM_KEYS, *MAX_KEYS))
                                    for case in cases),
        "seat_delta_vs_stock": {key: aggregate["seat"][key] - aggregate["stock"][key]
                                for key in (*SUM_KEYS, *MAX_KEYS)},
        "seat_output_equal_stock": sum(x["stock_output_sha256"] == x["seat_output_sha256"]
                                        for x in seat_cases),
        "rescues": [x["index"] for x in seat_cases if x["correct_delta"] > 0],
        "losses": [x["index"] for x in seat_cases if x["correct_delta"] < 0],
        "cases": seat_cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = build(args.root)
    destination = args.root / "aggregate.json"
    temporary = args.root / "aggregate.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    print(json.dumps(report, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

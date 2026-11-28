#!/usr/bin/env python3
"""Audit whether sealed Gemma-B receipts support a native-argmax endpoint.

This is an offline compact-receipt audit.  It loads no model, opens no device,
and never substitutes sparse or cached logits for a measured native sample.
The question is narrower than whole-request parity: after a complete native
single-head denoising step and a real native stop, may the same turn's native
argmax be the model-defined block output?
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DEFAULT_PREDICATE = Path(
    "experiments/t2t/serving_slots/controls/gemma-factors/"
    "predicate-only-a1/pairs.csv"
)
DEFAULT_CONTINUOUS = Path(
    "experiments/t2t/serving_slots/controls/gemma-factors/"
    "continuous-rng-a1/pairs.csv"
)
DEFAULT_VISIBILITY = Path(
    "experiments/t2t/serving_slots/controls/gemma-factors/"
    "native-visibility-a1/pairs.csv"
)


def read_rows(path: Path, *, require_inputs: bool = True) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 90:
        raise ValueError(f"{path}: expected 90 compact rows, got {len(rows)}")
    keys = [
        (int(row["index"]), int(row["seed"]), row.get("input_sha"))
        for row in rows
    ]
    if require_inputs and any(key[2] is None for key in keys):
        raise ValueError(f"{path}: missing input_sha")
    if len(set(keys)) != 90:
        raise ValueError(f"{path}: request identity is not unique")
    return rows


def integer(row: dict[str, str], arm: str, field: str) -> int:
    return int(row[f"{arm}_{field}"])


def arm_summary(rows: list[dict[str, str]], arm: str) -> dict[str, int | bool]:
    block_count = sum(integer(row, arm, "verify_F") for row in rows)
    native_hits = sum(integer(row, arm, "native_hits") for row in rows)
    limit_only = sum(integer(row, arm, "limit_only") for row in rows)
    all_terminal_native = all(
        integer(row, arm, "verify_F") == integer(row, arm, "native_hits")
        and integer(row, arm, "limit_only") == 0
        for row in rows
    )
    return {
        "requests": len(rows),
        "N": sum(integer(row, arm, "N") for row in rows),
        "F": sum(integer(row, arm, "F") for row in rows),
        "sample_F": sum(integer(row, arm, "sample_F") for row in rows),
        "verify_F": block_count,
        "encoder_F": sum(integer(row, arm, "encoder_F") for row in rows),
        "decoder_Q": sum(integer(row, arm, "decoder_Q") for row in rows),
        "terminal_blocks": block_count,
        "terminal_native_hits": native_hits,
        "terminal_all_accepted_before_native_hit": max(
            0, block_count - native_hits
        ),
        "step_limit_only_terminal_blocks": limit_only,
        "all_terminal_blocks_real_native_stop": all_terminal_native,
        "native_exact_requests": sum(
            integer(row, arm, "native_exact") for row in rows
        ),
        "draft_vs_verified_diff_blocks": sum(
            integer(row, arm, "draft_verified_diff") for row in rows
        ),
        "nonterminal_encoder_input_diff_blocks": sum(
            integer(row, arm, "draft_verified_encoder_diff") for row in rows
        ),
        "first_native_mismatch_is_at_verify_requests": sum(
            integer(row, arm, "first_mismatch_at_verify") for row in rows
        ),
    }


def exceptions(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for row in rows:
        if (
            integer(row, "on", "draft_verified_diff")
            or integer(row, "on", "draft_verified_encoder_diff")
            or integer(row, "on", "encoder_F") != integer(row, "native", "encoder_F")
        ):
            result.append(
                {
                    "index": int(row["index"]),
                    "dataset": row["dataset"],
                    "on_terminal_blocks": integer(row, "on", "verify_F"),
                    "on_nonterminal_encoder_blocks": integer(row, "on", "encoder_F"),
                    "on_terminal_native_hits": integer(row, "on", "native_hits"),
                    "on_draft_vs_verified_diff_blocks": integer(
                        row, "on", "draft_verified_diff"
                    ),
                    "on_nonterminal_encoder_input_diff_blocks": integer(
                        row, "on", "draft_verified_encoder_diff"
                    ),
                    "native_encoder_blocks": integer(row, "native", "encoder_F"),
                    "on_native_exact": bool(integer(row, "on", "native_exact")),
                    "first_mismatch_at_verify": bool(
                        integer(row, "on", "first_mismatch_at_verify")
                    ),
                }
            )
    return result


def visibility_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    def boolean(arm: str, field: str) -> int:
        return sum(row[f"{arm}_{field}"] == "True" for row in rows)

    return {
        "requests": len(rows),
        "first_block_equal_vs_same_stack_native": {
            "strict_off": boolean("off", "native_first_block_equal"),
            "visibility_on": boolean("on", "native_first_block_equal"),
        },
        "same_stack_native_totals": {
            "N": sum(integer(row, "native", "N") for row in rows),
            "F": sum(integer(row, "native", "F") for row in rows),
            "encoder_F": sum(integer(row, "native", "encoder_F") for row in rows),
            "decoder_Q": sum(integer(row, "native", "decoder_Q") for row in rows),
        },
    }


def continuous_summary(rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        "requests": len(rows),
        "exposed": sum(integer(row, "on", "F") >= 0 and int(row["exposed"]) for row in rows),
        "later_block_openings": sum(integer(row, "on", "next_inits") for row in rows),
        "later_openings_carrying_request_rng": sum(integer(row, "on", "carried") for row in rows),
        "on_native_first_equal": sum(integer(row, "on", "native_first_equal") for row in rows),
        "on_native_exact": sum(integer(row, "on", "native_exact") for row in rows),
    }


def audit(predicate_path: Path, continuous_path: Path, visibility_path: Path) -> dict[str, object]:
    predicate = read_rows(predicate_path)
    continuous = read_rows(continuous_path)
    visibility = read_rows(visibility_path, require_inputs=False)
    by_index = {int(row["index"]): row for row in continuous}
    for row in predicate:
        old = by_index[int(row["index"])]
        if (
            int(row["seed"]) != int(old["seed"])
            or row["dataset"] != old["dataset"]
            or row["input_sha"] != old["input_sha"]
            or row["reference"] != old["row_sha"]
            or row["native_reference"] != old["reference_sha"]
        ):
            raise ValueError(
                f"index {row['index']}: predicate/continuous provenance mismatch"
            )

    for row in predicate:
        old = by_index[int(row["index"])]
        vis = next(item for item in visibility if int(item["index"]) == int(row["index"]))
        for field in ("N", "F", "correct", "decoder_Q", "encoder_F"):
            if integer(row, "native", field) != integer(vis, "native", field):
                raise ValueError(
                    f"index {row['index']}: predicate/visibility native {field} mismatch"
                )

    on = arm_summary(predicate, "on")
    off = arm_summary(predicate, "off")
    native_encoder_f = sum(integer(row, "native", "encoder_F") for row in predicate)
    endpoint_supported = bool(
        on["all_terminal_blocks_real_native_stop"]
        and on["step_limit_only_terminal_blocks"] == 0
        and on["sample_F"] > 0
    )
    return {
        "scope": "offline compact receipts only; no model, GPU, logits, or RNG call",
        "predicate_pairs": str(predicate_path),
        "continuous_pairs": str(continuous_path),
        "visibility_pairs": str(visibility_path),
        "endpoint_supported_for_same_turn_argmax": endpoint_supported,
        "whole_request_native_parity_supported": False,
        "reason": (
            "The on arm closes every terminal block with a real native stop and "
            "measured same-turn denoising logits, but ordered strict verification "
            "changes the final argmax draft on some blocks, including five "
            "nonterminal blocks that would feed an encoder."
        ),
        "strict_contract": off,
        "native_stop_candidate": on,
        "native_reference_totals": {
            "N": sum(integer(row, "native", "N") for row in predicate),
            "F": sum(integer(row, "native", "F") for row in predicate),
            "encoder_F": native_encoder_f,
            "decoder_Q": sum(integer(row, "native", "decoder_Q") for row in predicate),
        },
        "actual_not_virtual_next_encode": {
            "candidate_nonterminal_encoder_calls": on["encoder_F"],
            "native_nonterminal_encoder_calls": native_encoder_f,
            "candidate_requests_with_native_encoder_block_count": sum(
                integer(row, "on", "encoder_F")
                == integer(row, "native", "encoder_F")
                for row in predicate
            ),
            "candidate_nonterminal_encoder_input_changes_vs_strict": on[
                "nonterminal_encoder_input_diff_blocks"
            ],
        },
        "continuous_rng_summary": continuous_summary(continuous),
        "visibility_summary": visibility_summary(visibility),
        "exceptions": exceptions(predicate),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicate", type=Path, default=DEFAULT_PREDICATE)
    parser.add_argument("--continuous", type=Path, default=DEFAULT_CONTINUOUS)
    parser.add_argument("--visibility", type=Path, default=DEFAULT_VISIBILITY)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit(args.predicate, args.continuous, args.visibility)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()

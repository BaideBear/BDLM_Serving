"""CPU-only placebo placement report from ``tools/t2t_results.py`` scores.

The placebo arms ``hard_slack_placebo`` and ``hard_trigger_placebo`` reserve the
same number of extra old-row seats as their adaptive counterparts in the same
state, but draw those positions uniformly from the non-fresh resolved pool.
This tool reads the per-request CSV already written by ``tools/t2t_results.py``.
It does not score outputs, load a dataset, run a model, or rewrite score tables.

Inputs: ``--scores-dir`` (directory with ``{model}-online-requests.csv``),
optional ``--reference-scores-dir`` for the published adaptive and hard/noedit
score tables, and optional ``--run`` / ``--reference-run`` raw request
directories.  ``fullwidth`` and ``native256`` raw policies alias to ``native``.

Outputs:
``{model}-placement-comparison.csv``
    Per placebo/reference pair: correct counts, mean per-request TPF,
    ``sum(N)``, ``sum(F)``, ``sum(Q)``, token efficiency, q-per-forward,
    wins/losses, strict success, output/all-Q agreement, first divergence,
    prefix reserved seats, and state evidence.
``{model}-placement-alignment.csv``
    Per request: first differing decoder forward, output identity, prefix
    reserved-seat agreement, and whether per-step state evidence exists.

Caveats: the report is descriptive and one-sided, not a causal estimate.
Missing per-step state yields ``state_evidence=missing_state`` and
``same_state_claim=0``.  ``prefix_*_reserved_seats`` sum only the common Q
prefix (steps before the first divergence), not the complete request.

    python tools/placement_comparison.py --model gemma \
        --scores-dir experiments/t2t/placement \
        --run /path/to/placebo-run --reference-run /path/to/adaptive-run \
        --out experiments/t2t/placement
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
PLACEBO_BASE = {
    "hard_slack_placebo": "hard_slack",
    "hard_trigger_placebo": "hard_trigger",
}
POLICY = re.compile(r"^t2t_q(\d+)_f(\d+)_m(\d+)_([a-z0-9_]+)$")
NATIVE_ALIASES = {"native", "fullwidth", "native256"}

COMPARISON_COLUMNS = (
    "model", "placebo_policy", "placebo_nominal_q", "reference_policy",
    "reference_source", "requests", "correct", "reference_correct",
    "delta_correct", "tpf", "reference_tpf", "delta_tpf", "N", "F", "Q",
    "reference_N", "reference_F", "reference_Q", "token_efficiency",
    "reference_token_efficiency", "q_per_f", "reference_q_per_f", "wins",
    "losses", "strict_success", "reference_strict_success", "output_compared",
    "output_identical", "all_q_compared", "all_q_identical",
    "first_different_requests", "first_different_first_step",
    "first_different_median", "prefix_reserved_compared_requests",
    "prefix_reserved_compared_steps", "prefix_reserved_identical_requests",
    "prefix_reserved_mismatch_steps",
    "prefix_placebo_reserved_seats", "prefix_reference_reserved_seats",
    "state_evidence", "state_verified_requests", "state_mismatch_requests",
    "state_missing_requests", "same_state_claim", "causal_claim",
    "interpretation",
)
ALIGNMENT_COLUMNS = (
    "model", "placebo_policy", "reference_policy", "index", "dataset",
    "task_id", "placebo_steps", "reference_steps", "compared_steps",
    "positions_all_identical", "first_different_step", "first_different_block",
    "first_different_stage", "q_symmetric_difference", "output_identical",
    "output_evidence", "prefix_reserved_compared_steps",
    "prefix_placebo_reserved_seats", "prefix_reference_reserved_seats",
    "prefix_reserved_mismatch_steps",
    "prefix_reserved_counts_identical", "state_evidence", "state_known_steps",
    "state_unknown_steps", "state_same",
)


def _json(path):
    return json.loads(Path(path).read_text())


def parse_policy(text):
    match = POLICY.match(text)
    if match is None:
        raise ValueError("Invalid T2T policy: %s" % text)
    budget, fresh, mask, selection = match.groups()
    return dict(budget=int(budget), fresh=int(fresh), mask=int(mask),
                selection=selection)


def _identity(row):
    return (int(row["index"]), row["dataset"], row["task_id"])


def _int(value):
    return 0 if value in (None, "") else int(value)


def _strict(row):
    return (int(row["correct"]) == 1 and _int(row.get("failed", 0)) == 0
            and _int(row.get("truncated", 0)) == 0)


def _write_table(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_scores(directory, model):
    """Read {model}-online-requests.csv from tools/t2t_results.py."""
    path = Path(directory) / ("%s-online-requests.csv" % model)
    if not path.is_file():
        raise ValueError("Missing score table: %s" % path)
    by_policy = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("model") == model:
                by_policy[row["policy"]].append(row)
    return dict(by_policy)


def _source(policy):
    if policy == "native":
        return "native"
    if policy.endswith("_hard_noedit"):
        return "hard_noedit"
    if policy.endswith("_hard_slack") or policy.endswith("_hard_trigger"):
        return "adaptive"
    return "reference"


def load_reference_scores(directories, model):
    merged, sources = {}, {}
    for directory in directories:
        for policy, rows in read_scores(directory, model).items():
            if policy in merged:
                continue
            merged[policy] = rows
            sources[policy] = _source(policy)
    return merged, sources


def load_raw_runs(folders, model):
    """Map (policy, index) to a raw request; fullwidth aliases to native."""
    records = {}
    for folder in folders:
        for path in sorted(Path(folder).glob("??-*.json")):
            row = _json(path)
            policy = row.get("policy")
            if policy in NATIVE_ALIASES:
                policy = "native"
            elif policy is None or POLICY.match(policy) is None:
                continue
            if row.get("model") != model:
                if not (model == "gemma" and row.get("model") == "diffusiongemma"):
                    continue
            key = (policy, int(row["index"]))
            records.setdefault(key, row)
    return records


def _pair_metrics(placebo, reference):
    a = {_identity(row): row for row in placebo}
    b = {_identity(row): row for row in reference}
    if set(a) != set(b):
        raise ValueError("Placebo and reference cover different requests")
    keys = sorted(a, key=lambda key: (key[0], key[1], key[2]))
    count = len(keys)
    correct = reference_correct = wins = losses = 0
    strict = reference_strict = 0
    sums_a, sums_b = defaultdict(int), defaultdict(int)
    delta_tpf = []
    for key in keys:
        pa, pb = a[key], b[key]
        ca, cb = int(pa["correct"]), int(pb["correct"])
        correct += ca
        reference_correct += cb
        wins += int(ca and not cb)
        losses += int(cb and not ca)
        strict += int(_strict(pa))
        reference_strict += int(_strict(pb))
        for row, target in ((pa, sums_a), (pb, sums_b)):
            for field in ("N", "F", "Q"):
                target[field] += _int(row[field])
        delta_tpf.append(_int(pa["N"]) / _int(pa["F"])
                         - _int(pb["N"]) / _int(pb["F"]))
    return dict(
        requests=count, correct=correct, reference_correct=reference_correct,
        delta_correct=correct - reference_correct,
        tpf=mean(_int(row["N"]) / _int(row["F"]) for row in placebo),
        reference_tpf=mean(_int(row["N"]) / _int(row["F"]) for row in reference),
        delta_tpf=mean(delta_tpf),
        N=sums_a["N"], F=sums_a["F"], Q=sums_a["Q"],
        reference_N=sums_b["N"], reference_F=sums_b["F"], reference_Q=sums_b["Q"],
        token_efficiency=sums_a["N"] / sums_a["Q"] if sums_a["Q"] else 0.0,
        reference_token_efficiency=(sums_b["N"] / sums_b["Q"]
                                    if sums_b["Q"] else 0.0),
        q_per_f=sums_a["Q"] / sums_a["F"] if sums_a["F"] else 0.0,
        reference_q_per_f=sums_b["Q"] / sums_b["F"] if sums_b["F"] else 0.0,
        wins=wins, losses=losses, strict_success=strict,
        reference_strict_success=reference_strict,
    )


def _normalize(model, positions, block):
    return [int(p) for p in positions] if model == "llada21" else [block + int(p) for p in positions]


def _fingerprint(model, event, block):
    if event.get("state_hash") is not None:
        return ("hash", str(event["state_hash"]))
    fresh = event.get("fresh_positions")
    resolved = event.get("resolved_positions")
    if resolved is None and isinstance(event.get("resolved"), list):
        if all(isinstance(item, bool) for item in event["resolved"]):
            resolved = [i for i, item in enumerate(event["resolved"]) if item]
    if fresh is None or resolved is None:
        return None
    return ("sets", tuple(sorted(_normalize(model, fresh, block))),
            tuple(sorted(_normalize(model, resolved, block))))


def _steps(model, record):
    steps = []
    for event in record.get("events", []):
        if event.get("phase") == "prefill" or event.get("stage") == "prefill":
            continue
        if event.get("stage") == "encoder":
            continue
        block = int(event.get("block", 0))
        if model == "llada21":
            positions = event.get("absolute_positions")
            if positions is None:
                positions = range(int(event["q_start"]), int(event["q_end"]))
        else:
            positions = event.get("positions")
            if positions is None:
                continue
        reserved = event.get("reserved_extra_edit_positions")
        steps.append(dict(
            block=block, stage=event.get("stage") or event.get("phase"),
            positions=tuple(sorted(_normalize(model, positions, block))),
            reserved=len(reserved) if isinstance(reserved, list) else None,
            fingerprint=_fingerprint(model, event, block),
        ))
    return steps


def _output(record):
    if record.get("output_ids") is not None:
        return ("ids", tuple(record["output_ids"]))
    if record.get("output_text") is not None:
        return ("text", str(record["output_text"]))
    return None


def compare_request(model, placebo, reference):
    """Compare one raw request pair; reserved totals cover the common Q prefix."""
    a, b = _steps(model, placebo), _steps(model, reference)
    first_diff = None
    same, known, unknown = True, 0, 0
    for index in range(min(len(a), len(b))):
        x, y = a[index], b[index]
        if x["block"] != y["block"] or x["stage"] != y["stage"]:
            unknown += 1
            first_diff = dict(ordinal=index + 1, block=x["block"], stage=x["stage"],
                              symmetric=len(set(x["positions"]) ^ set(y["positions"])))
            break
        if x["fingerprint"] is None or y["fingerprint"] is None:
            unknown += 1
            same = False
        else:
            known += 1
            same = same and x["fingerprint"] == y["fingerprint"]
        if x["positions"] != y["positions"]:
            first_diff = dict(ordinal=index + 1, block=x["block"], stage=x["stage"],
                              symmetric=len(set(x["positions"]) ^ set(y["positions"])))
            break
    if first_diff is None and len(a) != len(b):
        unknown += 1
        longer = a if len(a) > len(b) else b
        first_diff = dict(ordinal=min(len(a), len(b)) + 1,
                          block=longer[min(len(a), len(b))]["block"],
                          stage=longer[min(len(a), len(b))]["stage"], symmetric=0)

    # Reserved-seat totals cover the common Q prefix only: steps strictly
    # before the first divergence, never the complete request.
    prefix = (first_diff["ordinal"] - 1) if first_diff else min(len(a), len(b))
    reserved_compared = reserved_mismatch = 0
    reserved_a = reserved_b = 0
    for x, y in zip(a[:prefix], b[:prefix]):
        if x["reserved"] is None or y["reserved"] is None:
            continue
        reserved_compared += 1
        reserved_a += x["reserved"]
        reserved_b += y["reserved"]
        reserved_mismatch += int(x["reserved"] != y["reserved"])

    out_a, out_b = _output(placebo), _output(reference)
    if out_a is None or out_b is None:
        output_identical, output_evidence = None, "missing"
    elif out_a[0] == out_b[0]:
        output_identical, output_evidence = out_a[1] == out_b[1], out_a[0]
    else:
        output_identical = False if "ids" in (out_a[0], out_b[0]) else out_a[1] == out_b[1]
        output_evidence = "mixed"
    if not a or not b:
        state_evidence = "missing_raw"
    elif unknown:
        state_evidence = "missing_state"
    else:
        state_evidence = "verified"
    return dict(
        placebo_steps=len(a), reference_steps=len(b),
        compared_steps=min(len(a), len(b)),
        positions_all_identical=(first_diff is None and len(a) == len(b)),
        first_different_step=first_diff["ordinal"] if first_diff else "",
        first_different_block=first_diff["block"] if first_diff else "",
        first_different_stage=first_diff["stage"] if first_diff else "",
        q_symmetric_difference=first_diff["symmetric"] if first_diff else 0,
        output_identical=output_identical, output_evidence=output_evidence,
        prefix_reserved_compared_steps=reserved_compared,
        prefix_placebo_reserved_seats=reserved_a,
        prefix_reference_reserved_seats=reserved_b,
        prefix_reserved_mismatch_steps=reserved_mismatch,
        prefix_reserved_counts_identical=(reserved_compared > 0
                                          and reserved_mismatch == 0),
        state_evidence=state_evidence, state_known_steps=known,
        state_unknown_steps=unknown,
        state_same=(True if state_evidence == "verified" and same and known else
                    False if state_evidence == "verified" else ""),
        same_state_claim=bool(state_evidence == "verified" and known and same),
    )


def build_comparison(model, placebo_scores, reference_scores, reference_sources,
                     raw_records, *, include_native=True):
    comparison, alignment = [], []
    for policy in sorted(p for p in placebo_scores if p.endswith("_placebo")):
        parsed = parse_policy(policy)
        if parsed["selection"] not in PLACEBO_BASE:
            continue
        base = PLACEBO_BASE[parsed["selection"]]
        references = []
        adaptive = "t2t_q%d_f0_m%d_%s" % (parsed["budget"], parsed["mask"], base)
        if adaptive in reference_scores:
            references.append((adaptive, reference_sources[adaptive]))
        noedit = "t2t_q%d_f0_m%d_hard_noedit" % (parsed["budget"], parsed["budget"])
        if noedit in reference_scores:
            references.append((noedit, reference_sources[noedit]))
        if include_native and "native" in reference_scores:
            references.append(("native", reference_sources["native"]))
        if not references:
            raise ValueError("No reference score for %s" % policy)
        for reference_policy, source in references:
            reference_rows = reference_scores[reference_policy]
            if source in ("adaptive", "hard_noedit"):
                expected_mask = parsed["mask"] if source == "adaptive" else parsed["budget"]
                for row in reference_rows:
                    if (int(row["nominal_q"]) != parsed["budget"]
                            or int(row["mask"]) != expected_mask):
                        raise ValueError("Reference %s has the wrong Q/mask" % reference_policy)
            metrics = _pair_metrics(placebo_scores[policy], reference_rows)

            ref_by_key = {_identity(row): row for row in reference_rows}
            output_compared = output_identical = 0
            all_q_compared = all_q_identical = 0
            first_steps = []
            reserved_requests = reserved_steps = 0
            reserved_identical = reserved_mismatch = 0
            prefix_a = prefix_b = 0
            verified = mismatch = missing = 0
            for row in placebo_scores[policy]:
                key = _identity(row)
                if key not in ref_by_key:
                    continue
                raw_a = raw_records.get((policy, key[0]))
                raw_b = raw_records.get((reference_policy, key[0]))
                if raw_a is None or raw_b is None:
                    continue
                result = compare_request(model, raw_a, raw_b)
                alignment.append(dict(
                    model=model, placebo_policy=policy, reference_policy=reference_policy,
                    index=key[0], dataset=key[1], task_id=key[2],
                    **{field: result[field] for field in (
                        "placebo_steps", "reference_steps", "compared_steps",
                        "positions_all_identical", "first_different_step",
                        "first_different_block", "first_different_stage",
                        "q_symmetric_difference", "output_identical",
                        "output_evidence", "prefix_reserved_compared_steps",
                        "prefix_placebo_reserved_seats",
                        "prefix_reference_reserved_seats",
                        "prefix_reserved_mismatch_steps",
                        "prefix_reserved_counts_identical", "state_evidence",
                        "state_known_steps", "state_unknown_steps", "state_same")},
                ))
                if result["output_identical"] is not None:
                    output_compared += 1
                    output_identical += int(result["output_identical"])
                if result["compared_steps"]:
                    all_q_compared += 1
                    all_q_identical += int(result["positions_all_identical"])
                if result["prefix_reserved_compared_steps"]:
                    reserved_requests += 1
                    reserved_steps += result["prefix_reserved_compared_steps"]
                    reserved_identical += int(result["prefix_reserved_counts_identical"])
                    reserved_mismatch += result["prefix_reserved_mismatch_steps"]
                    prefix_a += result["prefix_placebo_reserved_seats"]
                    prefix_b += result["prefix_reference_reserved_seats"]
                if result["first_different_step"] != "":
                    first_steps.append(result["first_different_step"])
                if result["state_evidence"] == "verified":
                    if result["same_state_claim"]:
                        verified += 1
                    else:
                        mismatch += 1
                else:
                    missing += 1
            if not all_q_compared:
                state_evidence = "missing_raw"
            elif missing:
                state_evidence = "missing_state"
            elif mismatch:
                state_evidence = "state_mismatch"
            else:
                state_evidence = "verified"
            interpretation = ("descriptive one-sided paired report; not a causal estimate")
            if state_evidence == "missing_state":
                interpretation += "; same-state alignment not established (missing per-step state)"
            elif state_evidence == "state_mismatch":
                interpretation += "; compared steps do not all share the same state"
            comparison.append(dict(
                model=model, placebo_policy=policy,
                placebo_nominal_q=parsed["budget"], reference_policy=reference_policy,
                reference_source=source, **metrics,
                output_compared=output_compared, output_identical=output_identical,
                all_q_compared=all_q_compared, all_q_identical=all_q_identical,
                first_different_requests=len(first_steps),
                first_different_first_step=sum(step == 1 for step in first_steps),
                first_different_median=(sorted(first_steps)[len(first_steps) // 2]
                                        if first_steps else ""),
                prefix_reserved_compared_requests=reserved_requests,
                prefix_reserved_compared_steps=reserved_steps,
                prefix_reserved_identical_requests=reserved_identical,
                prefix_reserved_mismatch_steps=reserved_mismatch,
                prefix_placebo_reserved_seats=prefix_a,
                prefix_reference_reserved_seats=prefix_b,
                state_evidence=state_evidence, state_verified_requests=verified,
                state_mismatch_requests=mismatch, state_missing_requests=missing,
                same_state_claim=int(all_q_compared > 0 and missing == 0 and mismatch == 0),
                causal_claim=0, interpretation=interpretation,
            ))
    return comparison, alignment


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="CPU-only placebo placement report from t2t_results scores.",
        epilog="Descriptive and one-sided; not a causal estimate. "
               "Reserved-seat totals cover only the common Q prefix.")
    parser.add_argument("--model", required=True, choices=("llada21", "gemma"))
    parser.add_argument("--scores-dir", type=Path, required=True,
                        help="Directory containing {model}-online-requests.csv")
    parser.add_argument("--reference-scores-dir", type=Path, action="append",
                        help="Extra score directory; repeatable. Defaults to the published adaptive and hard directories")
    parser.add_argument("--run", type=Path, action="append", default=[],
                        help="Raw placebo run directory for alignment; repeatable")
    parser.add_argument("--reference-run", type=Path, action="append", default=[],
                        help="Raw adaptive/noedit run directory for alignment; repeatable")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--no-native", action="store_true",
                        help="Do not add the native reference")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    placebo_scores = {policy: rows for policy, rows in
                      read_scores(args.scores_dir, args.model).items()
                      if policy.endswith("_placebo")}
    if not placebo_scores:
        raise ValueError("No * _placebo policies in %s" % args.scores_dir)
    reference_dirs = args.reference_scores_dir or [
        ROOT / "experiments/t2t/adaptive", ROOT / "experiments/t2t/hard"]
    reference_scores, reference_sources = load_reference_scores(reference_dirs, args.model)
    raw_records = load_raw_runs(list(args.run) + list(args.reference_run), args.model)
    comparison, alignment = build_comparison(
        args.model, placebo_scores, reference_scores, reference_sources,
        raw_records, include_native=not args.no_native)
    _write_table(args.out / ("%s-placement-comparison.csv" % args.model),
                 comparison, COMPARISON_COLUMNS)
    _write_table(args.out / ("%s-placement-alignment.csv" % args.model),
                 alignment, ALIGNMENT_COLUMNS)
    print("placement report: %d placebo arms, %d pairs -> %s"
          % (len(placebo_scores), len(comparison), args.out))
    print("Descriptive one-sided report; not a causal estimate.")
    if alignment:
        missing = sum(row["state_evidence"] != "verified" for row in alignment)
        if missing:
            print("Same-state alignment NOT established for %d/%d requests: raw "
                  "traces lack per-step fresh/resolved state." % (missing, len(alignment)))
    else:
        print("No raw placebo/reference pairs supplied; raw alignment columns are empty.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

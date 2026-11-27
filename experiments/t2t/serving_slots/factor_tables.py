"""Render Gemma rank/close aggregate reports; never requires model outputs."""

import argparse
import csv
import json
from pathlib import Path
import sys

from .cross_race_report import COHORTS, MANIFEST


def tables(report, kind, manifest):
    if (report["schema"] != "cross-race-report-v1" or not report["manifest_verified"]
            or report["samples_sha256"] != manifest["samples_sha256"]
            or kind not in ("rank", "close")):
        raise ValueError("unregistered schema or manifest")
    coverage, results = [], []
    for pair in report["comparisons"]:
        if pair["worker"] != kind:
            raise ValueError("wrong suite")
        for name, indices in COHORTS.items():
            cohort = pair["cohorts"][name]
            base = dict(cohort=name, budget=pair["budget"], reference=pair["reference"],
                        trial=pair["trial"])
            diag = cohort.get("parity", cohort.get("output_comparison", {}))
            coverage.append(dict(**base, expected=len(indices), paired=cohort["paired_work"],
                                 scored=cohort["paired_scored"], status=cohort["status"],
                                 reference_only=cohort["reference_only"], trial_only=cohort["trial_only"],
                                 **diag))
            if cohort["status"] != "complete":
                continue
            if cohort["paired_scored"] != len(indices) or cohort["paired_work"] != len(indices):
                raise ValueError("invalid complete cohort")
            q, work = cohort["quality"], cohort["work"]
            for side in ("reference", "trial"):
                w = work[side]
                results.append(dict(**base, side=side, arm=pair[side], n=len(indices),
                                    correct=q[side+"_correct"], rescues=q["rescue"], losses=q["loss"],
                                    macro_TPF=w["macro_TPF"], token_efficiency=w["N_per_history_Q"],
                                    **{k:w[k] for k in ("N", "F", "decoder_F", "encoder_F", "history_Q",
                                        "decoder_Q", "encoder_Q", "physical_Q", "peak_decoder_Q", "truncated")},
                                    **{f"Q_{k}":v for k,v in w["categories"].items()},
                                    **{k:w[k] for k in ("prefix_seats", "differing_prefix_seats",
                                        "prefix_distance_sum", "prefix_edits", "prefix_revoked",
                                        "resample_paid_F", "fused_F", "fused_dirty_Q") if k in w},
                                    **diag, **{k+"_ci95":v for k,v in cohort["confidence95"].items()
                                        if k in ("accuracy_delta", "macro_TPF_delta", "N_per_history_Q_delta")}))
    return coverage, results


def write_csv(path, rows):
    keys = dict.fromkeys(key for r in rows for key in r)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, lineterminator="\n", fieldnames=list(keys) or
                                ["cohort", "budget", "reference", "trial", "side", "arm", "n"])
        writer.writeheader()
        writer.writerows({k:json.dumps(v, separators=(",", ":")) if isinstance(v, list) else v
                          for k,v in row.items()} for row in rows)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kind", choices=["rank","close"])
    ap.add_argument("destination", type=Path)
    args = ap.parse_args()
    report = json.load(sys.stdin)
    coverage, results = tables(report, args.kind, json.loads(MANIFEST.read_text()))
    dest = args.destination
    dest.mkdir(parents=True, exist_ok=True)
    write_csv(dest/"coverage.csv", coverage)
    write_csv(dest/"paired.csv", results)
    meta = {k:report[k] for k in ("schema", "status", "samples_sha256", "sources", "scorers", "bootstrap")}
    meta["kind"] = args.kind
    tmp = dest/"provenance.tmp"
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True)+"\n")
    tmp.replace(dest/"provenance.json")


if __name__ == "__main__":
    main()

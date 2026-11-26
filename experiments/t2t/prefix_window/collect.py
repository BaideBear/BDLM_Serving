"""Score and audit completed prefix arms against an explicit same-Q zero run.

Point --out at private cluster storage (scoring also emits per-request files).
Only the cohort comparison CSVs and the exposure CSV are for publication.
"""
import argparse
from pathlib import Path

from tools.prefix_cohorts import score_files
from bdlm_serving.methods.t2t.physical_prefix import DONOR_MODES
from . import exposure
from .score_run import snapshot


LABELS = {"hard_prefix_frontier_right": ("", "right"), "hard_prefix_random": ("", "random"),
          "hard_prefix_slack_right": ("slack-", "right"), "hard_prefix_slack_random": ("slack-", "random"),
          "hard_prefix_slack_oldest": ("slack-", "oldest")}
LABELS.update({p: (donor.replace("_", "-") + "-", selector)
               for p, (donor, selector) in DONOR_MODES.items()})


def collect(run, baseline, cases, out):
    manifest, kind, width, digest = exposure.load_run(run, cases)
    base_meta, base_kind, _, base_digest = exposure.load_run(baseline, cases)
    if (base_kind, base_digest, base_meta["model"]) != (kind, digest, manifest["model"]):
        raise ValueError("Baseline model or cases differ")
    model = manifest["model"]
    out.mkdir(parents=True, exist_ok=True)
    cache = out / ".scoring_inputs"
    cases_by_id = exposure.load_cases(cases)
    rows = []
    for p in manifest["policies"]:
        if not (run / f"complete-{p}.json").is_file():
            continue
        parsed = exposure.parse_policy(p)
        if not parsed or parsed["mode"] not in exposure.MODES:
            raise ValueError(f"Unexpected prefix policy: {p}")
        q, mode = parsed["budget"], parsed["mode"]
        zero = f"t2t_q{q}_f0_m{q}_hard_prefix_zero"
        if zero not in base_meta["policies"]:
            raise ValueError(f"No same-Q zero control: {zero}")
        control = snapshot(baseline, zero, digest, cache)
        arm = snapshot(run, p, digest, cache)
        rows.extend(exposure.audit(run, p, cases_by_id, model, kind, width))
        if p != zero:
            k = q - parsed["mask"]
            family, pos = LABELS[mode]
            path = out / f"{model}-q{q}-{family}k{k}-{pos}.csv"
            summary, _, _ = score_files(control, arm, cases, path, baseline_policy=zero, arm_policy=p)
            for item in summary:
                if item["scope"] == "all" and item["window"] == 90:
                    print(path.name, item["arm_correct"], item["arm_macro_tpf"], flush=True)
    if not rows:
        raise ValueError("No complete policies")
    exposure.write_csv(rows, out / f"exposure-{run.name}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("run", "baseline", "cases", "out"):
        parser.add_argument("--" + flag, required=True, type=Path)
    collect(**vars(parser.parse_args()))

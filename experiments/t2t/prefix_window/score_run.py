"""Score only completed frozen90 arms; other arms may still be generating."""
import argparse
import json
import os
from pathlib import Path

from bdlm_serving.evaluation.frozen import sha
from tools.prefix_cohorts import score_files


def snapshot(run, policy, cases_sha, cache):
    meta = json.loads((run / "run.json").read_text())
    if meta["samples_sha256"] != cases_sha or meta["suite"] != "frozen90":
        raise ValueError(f"Unexpected suite or sample digest: {run}")
    if policy not in meta["policies"]:
        raise ValueError(f"Not registered in {run}: {policy}")
    qual = json.loads((run / "qualified.json").read_text())
    complete = json.loads((run / f"complete-{policy}.json").read_text())
    if (qual != dict(model=meta["model"], full_width=meta["block"], matched=90)
            or complete != dict(policy=policy, requests=90)):
        raise ValueError("Missing full-width qualification or complete arm")
    return link_requests(run, policy, cache)


def link_requests(run, policy, cache):
    # Run files are atomically finalized by durable(). Do not hash or load
    # unrelated policies from a live multi-arm directory in the scorer.
    target = cache / run.name / policy
    target.mkdir(parents=True, exist_ok=True)
    for i in range(90):
        src = run / f"{i:02d}-{policy}.json"
        dest = target / src.name
        if not dest.exists():
            os.link(src, dest)
        elif not dest.samefile(src):
            raise ValueError(f"Different input file already linked: {dest}")
    if len(list(target.iterdir())) != 90:
        raise ValueError(f"Unexpected extra input in {target}")
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("cases", "baseline-run", "arm-run", "out"):
        p.add_argument("--" + arg, required=True, type=Path)
    for arg in ("baseline-policy", "arm-policy"):
        p.add_argument("--" + arg, required=True)
    a = p.parse_args()
    digest, cache = sha(a.cases), a.out.parent / ".scoring_inputs"
    base = snapshot(a.baseline_run, a.baseline_policy, digest, cache)
    arm = snapshot(a.arm_run, a.arm_policy, digest, cache)
    summary, _, _ = score_files(base, arm, a.cases, a.out,
                                baseline_policy=a.baseline_policy, arm_policy=a.arm_policy)
    for s in summary:
        if s["scope"] == "all":
            print(s["cohort"], s["arm_correct"], s["arm_macro_tpf"],
                  s["arm_pooled_token_efficiency"], s["rescued"], s["lost"])


if __name__ == "__main__":
    main()

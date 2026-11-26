"""Compare complete Gemma Optimus runs to same nominal-Q K0 and slack arms.

Keep all raw outputs and paired request data in the private scoring directory.
"""
import argparse
import csv
import json
from pathlib import Path

from bdlm_serving.evaluation.frozen import ROOT, sha
from tools.prefix_cohorts import score_files
from .score_run import link_requests, snapshot


def read(path):
    return json.loads(path.read_text())


def baseline(run, policy, cases, cache, reference):
    meta = read(run / "run.json")
    mode, _ = policy.split("_")
    if not (meta["suite"] == "frozen90" and meta["model"] == "gemma" and meta["block"] == 256
            and meta["samples_sha256"] == sha(cases) and meta["seed_offset"] == 0
            and meta["acceptance"] == "optimus_dynamic_sparse_argmax_0.9"
            and meta["hard_q_cap"] is False and policy in meta["policies"]):
        raise ValueError("Optimus reference metadata mismatch")
    if read(run / f"complete-{policy}.json") != dict(policy=policy, requests=90):
        raise ValueError("Incomplete Optimus arm")
    qual = read(run / "qualified-logits.json")
    if qual != dict(modes=list(dict.fromkeys(p.split("_")[0] for p in meta["policies"])),
                    chunk=256, first_step_logits="bit_identical", index=0):
        raise ValueError("Missing mode-specific first-step logits check")
    old = [reference / mode / f"{i:02d}-{policy}.json" for i in range(30)]
    if read(run / f"reproduced30-{policy}.json") != dict(
            policy=policy, matched=30, reference_sha256=[sha(path) for path in old]):
        raise ValueError("Original30 references or reproduction marker differ")
    for i, path in enumerate(old):
        if read(run / f"{i:02d}-{policy}.json") != read(path):
            raise ValueError(f"Original30 reproduction mismatch: {policy}/{i}")
    return link_requests(run, policy, cache)


def compare(data, references, mode, q, out):
    policy = f"{mode}_{q}"
    cases = data / "samples.json"
    digest, cache = sha(cases), out / ".inputs"
    base = baseline(data / "gemma-optimus-reference90", policy, cases, cache, references)
    with (ROOT / "archive/diffusiongemma-optimus/requests.csv").open() as stream:
        public = {int(r["index"]): r for r in csv.DictReader(stream) if r["policy"] == policy}
    if set(public) != set(range(30)):
        raise ValueError("Incomplete public archive")
    zero = f"t2t_q{q}_f0_m{q}_hard_prefix_zero"
    slack = f"t2t_q{q}_f0_m{q-q//4}_hard_prefix_slack_{'right' if q == 32 else 'random'}"
    for label, p, run in (("zero", zero, "gemma-q32" if q == 32 else "gemma-grid-k1"),
                          ("slack", slack, "gemma-prefix-action-lock")):
        arm = snapshot(data / run, p, digest, cache)
        path = out / f"gemma-{policy}-{label}.csv"
        summary, paired, _ = score_files(base, arm, cases, path, nominal_q=q,
                                         baseline_policy=policy, arm_policy=p)
        for r in paired:
            if r["window"] != 30:
                continue
            h = public[r["index"]]
            if (r["task_id"] != h["task_id"] or
                    any(r[f"baseline_{a}"] != int(h[b]) for a,b in (
                        ("correct", "correct"), ("N", "output_tokens"), ("F", "generation_forwards"),
                        ("Q", "generation_query_rows"), ("Q", "generation_physical_padded_rows")))):
                raise ValueError(f"{policy}: public archive mismatch at index {r['index']}")
        for r in summary:
            if r["scope"] == "all":
                print(policy, label, r["window"], r["baseline_correct"], r["arm_correct"],
                      r["baseline_macro_tpf"], r["arm_macro_tpf"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "references", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--q", type=int, choices=(32, 128), required=True)
    parser.add_argument("--mode", choices=("ibs", "obs"), required=True)
    compare(**vars(parser.parse_args()))

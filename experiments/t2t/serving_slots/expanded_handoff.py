"""GPU0 fixed-offer allocation sweep; only change allocation within a service tier."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PARENT = "prefix90-gemma-high-prefix"
SHA = "ba80a293773326950738fd0d4ca0a8a04e6df4b86c6e9a7265bd2ba3a92e6e59"
SAMPLES = "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed"
STAGES = (
    ("gemma", 64, 16, ("backfill_right", "backfill_random")),
    ("llada21", 16, 4, ("slack_right", "slack_random")),
    ("llada21", 4, 2, ("slack_right", "slack_random")),
    ("llada21", 8, 4, ("slack_right", "slack_random")),
    ("llada21", 32, 8, ("slack_right", "slack_random")),
)


def read(path):
    return json.loads(path.read_text())


def checks(folder, model, digest, policies):
    meta = read(folder / "run.json")
    if (meta["model"], meta["suite"], meta["samples_sha256"], meta["policies"]) != (
            model, "frozen90", digest, policies):
        raise ValueError(f"Different experiment: {folder}")
    for p in policies:
        if read(folder / f"complete-{p}.json") != dict(policy=p, requests=90):
            raise ValueError(f"Incomplete: {folder}/{p}")


def predecessor(data):
    while not subprocess.run(["tmux", "has-session", "-t", "=" + PARENT],
                             capture_output=True, check=False).returncode:
        time.sleep(3)
    lo, hi = "128u32", "256n"
    pats = [(lo,), (hi,), (hi, hi, lo, lo), (lo, lo, hi, hi), (hi, lo, lo, lo), (lo, hi, lo, lo)]
    checks(data / "gemma-native-high-prefix128", "gemma", SAMPLES,
           ["slottrace_" + "_".join(t) for t in pats])


def matrix(root):
    data = root / "prefix90"
    if hashlib.sha256((data / "samples.json").read_bytes()).hexdigest() != SAMPLES:
        raise ValueError("Wrong frozen suite")
    launcher = root / "launch_prefix90.py"
    if hashlib.sha256(launcher.read_bytes()).hexdigest() != SHA:
        raise ValueError("Unreviewed launcher")
    predecessor(data)
    for model, q, k, modes in STAGES:
        out = data / f"{model}-registered-q{q}-k{k}"
        arms = [f"{q}:0:{q-k}:hard_prefix_{m}" for m in modes]
        policies = [f"t2t_q{q}_f0_m{q-k}_hard_prefix_{m}" for m in modes]
        env = dict(os.environ, FROZEN90_SAMPLES=str(data/"samples.json"), TRACE_OUT=str(out),
                   PREFIX_MODE="online", T2T_ARMS=" ".join(arms))
        with out.with_suffix(".log").open("x") as log:
            print(f"Launching {model} fixed Q{q}, prefix K{k}: {modes}", flush=True)
            subprocess.run([sys.executable, "-u", launcher, model, "0"], env=env, cwd=root,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        checks(out, model, SAMPLES, policies)
        if read(out/"qualified.json") != dict(model=model,full_width=32 if model=="llada21" else 256,matched=90):
            raise ValueError(f"Unqualified: {out}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["chain"]:
        p = argparse.ArgumentParser()
        p.add_argument("chain")
        p.add_argument("runtime", type=Path)
        matrix(p.parse_args().runtime)
    else:
        raise ValueError("Expected chain <runtime>")

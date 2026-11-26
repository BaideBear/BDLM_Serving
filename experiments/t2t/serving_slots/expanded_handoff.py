"""Single GPU0 successor for both models' frozen90 midpoint and gate experiments."""
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
STAGES = (("llada21", "fixed", (24, 12, 6)), ("gemma", "fixed", (192, 96, 48)),
          ("gemma", "gate", (64, 192, 32)), ("llada21", "gate", (24, 8, 16)))


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
    for model, mode, qs in STAGES:
        # Releasing between invocations gives the holder back before loading the next model.
        groups = [qs] if mode == "fixed" else [(q,) for q in qs]
        for group in groups:
            suffix = "-".join(map(str, group))
            out = data / f"{model}-expanded-{mode}-{suffix}"
            if mode == "fixed":
                arms = [f"{q}:0:{q}:hard_prefix_zero" for q in group]
                policies = [f"t2t_q{q}_f0_m{q}_hard_prefix_zero" for q in group]
            else:
                policies = ["wide_" + s for s in ("fixed_low", "fixed_high", "all_high",
                                                    "tail_high", "all_low", "tail_low")]
            env = dict(os.environ, FROZEN90_SAMPLES=str(data/"samples.json"), TRACE_OUT=str(out),
                       PREFIX_MODE="online" if mode == "fixed" else "fifo")
            if mode == "fixed":
                env["T2T_ARMS"] = " ".join(arms)
                script = launcher
            else:
                env.update(WIDE_LOW=str(group[0]), FIFO_CONTROLS=str(data))
                script = Path(__file__).resolve()
            with out.with_suffix(".log").open("x") as log:
                print(f"Launching {model} {mode} {group}", flush=True)
                subprocess.run([sys.executable, "-u", script, model, "0"], env=env, cwd=root,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            checks(out, model, SAMPLES, policies)
            marker = read(out / ("qualified.json" if mode == "fixed" else "replay-qualified.json"))
            if mode == "fixed":
                expected = dict(model=model, full_width=32 if model == "llada21" else 256, matched=90)
            else:
                expected = dict(model=model, low=group[0], matched_per_endpoint=90, samples=SAMPLES)
            if marker != expected:
                raise ValueError(f"Unqualified: {out}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["chain"]:
        p = argparse.ArgumentParser()
        p.add_argument("chain")
        p.add_argument("runtime", type=Path)
        matrix(p.parse_args().runtime)
    elif (sys.argv[1:] in ([m, "0"] for m in ("gemma","llada21")) or
          sys.argv[1:] in ([m, "0", "worker"] for m in ("gemma","llada21"))):
        launcher = Path(os.environ["FROZEN90_SAMPLES"]).resolve().parents[1] / "launch_prefix90.py"
        src = launcher.read_bytes()
        target = b"from experiments.t2t.serving_slots.fifo import run"
        if (hashlib.sha256(src).hexdigest() != SHA or src.count(target) != 1
                or os.environ["PREFIX_MODE"] != "fifo"):
            raise RuntimeError("Unreviewed protected launcher")
        exec(compile(src.replace(target, b"from experiments.t2t.serving_slots.wide_gate import run"),
                     str(launcher), "exec"), globals())
    else:
        raise ValueError("Expected chain <runtime> or protected GPU0 worker")

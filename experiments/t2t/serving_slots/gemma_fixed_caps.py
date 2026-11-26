"""Fixed Gemma Q32/Q64 hard-cap IBS/OBS controls on the frozen90 suite.

The reviewed Optimus runtime already enforces labelled decoder caps.  This
module only registers the two missing fixed budgets, reuses the protected GPU0
launcher, and audits the completed run against the uncapped Q32/Q64 arms.  No
upstream model or launcher file is edited.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from bdlm_serving.evaluation.frozen import sha

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
ROOT = REPO.parent / "t2t-native-1125"
DATA = ROOT / "prefix90"
LAUNCHER = ROOT / "launch_prefix90.py"
SAMPLES = DATA / "samples.json"
RUNTIME = REPO / "bdlm_serving/runtime/diffusiongemma_optimus.py"

SAMPLES_SHA = "bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed"
LAUNCHER_SHA = "ba80a293773326950738fd0d4ca0a8a04e6df4b86c6e9a7265bd2ba3a92e6e59"
RUNTIME_SHA = "a09d04477361cb2a260dac85ce2195c0bfee9218306b7cfb96d7ddf4f1dba9af"
OUTPUT = "gemma-fixed-caps-q32q64-v1"
MODE = "optimus"
PLANS = ("32", "64")
MODES = ("ibs", "obs")
POLICIES = tuple(f"{mode}_cap_{plan}" for plan in PLANS for mode in MODES)
UNCAPPED = {
    "ibs_cap_32": ("gemma-optimus-reference90", "ibs_32"),
    "obs_cap_32": ("gemma-optimus-reference90", "obs_32"),
    "ibs_cap_64": ("gemma-optimus-primary90", "ibs_64"),
    "obs_cap_64": ("gemma-optimus-primary90", "obs_64"),
}
PREDECESSOR = "prefix90-mask-pair-gemma"
IMPORT = b"from bdlm_serving.runtime.diffusiongemma_optimus import run"
REPLACEMENT = b"from experiments.t2t.serving_slots.gemma_fixed_caps import run"
CALL = b"run(args, before_cuda=prepare)"


def read(path):
    return json.loads(Path(path).read_text())


def load(path, message):
    try:
        return read(path)
    except FileNotFoundError as error:
        raise ValueError(message) from error


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cap_of(policy):
    mode, tag, value = policy.split("_")
    require(tag == "cap" and mode in MODES, f"Unknown fixed-cap policy: {policy}")
    return int(value)


def protected_source(launcher):
    """Compile the reviewed launcher with only the Optimus run import replaced."""
    source = Path(launcher).read_bytes()
    require(hashlib.sha256(source).hexdigest() == LAUNCHER_SHA,
            "Not the reviewed GPU0 protected launcher")
    require(source.count(IMPORT) == 1 and source.count(CALL) == 1,
            "Unexpected Optimus launcher call site")
    require(b'MODE == "optimus"' in source, "Launcher lost the Optimus mode")
    return compile(source.replace(IMPORT, REPLACEMENT), str(launcher), "exec")


def verify_sources():
    require(LAUNCHER.is_file(), f"Missing protected launcher: {LAUNCHER}")
    protected_source(LAUNCHER)
    require(RUNTIME.is_file(), f"Missing reviewed Optimus runtime: {RUNTIME}")
    require(hashlib.sha256(RUNTIME.read_bytes()).hexdigest() == RUNTIME_SHA,
            "Reviewed Optimus runtime changed")
    return LAUNCHER


def verify_references(data, tasks):
    """The uncapped Q32/Q64 arms are reused, not rerun."""
    require(sha(data / "samples.json") == SAMPLES_SHA, "Wrong frozen90 samples")
    require([task["index"] for task in tasks] == list(range(90)),
            "Wrong frozen90 ordering")
    for policy, (folder, arm) in UNCAPPED.items():
        path = data / folder
        meta = read(path / "run.json")
        require((meta["model"], meta["suite"], meta["samples_sha256"]) ==
                ("gemma", "frozen90", SAMPLES_SHA) and meta.get("hard_q_cap") is False,
                f"Wrong uncapped reference: {path}")
        require(arm in meta["policies"], f"Missing uncapped arm {arm} in {path}")
        require(read(path / f"complete-{arm}.json") == dict(policy=arm, requests=90),
                f"Incomplete uncapped arm: {path}/{arm}")
        for task in tasks:
            row = read(path / f"{task['index']:02d}-{arm}.json")
            require((row["model"], row["policy"]) == ("gemma", arm) and
                    all(row[k] == task[k] for k in ("index", "task_id", "seed", "dataset")),
                    f"Wrong uncapped row: {arm}/{task['index']}")


def inspect_uncapped(data, tasks):
    """Where the nominal Q32/Q64 arms actually exceed their physical offer."""
    rows = []
    for policy, (folder, arm) in UNCAPPED.items():
        cap = cap_of(policy)
        stat = dict(policy=policy, cap=cap, requests=0, binding_requests=0,
                    excess_rows=0, max_decoder_q=0, first_binding_min=None,
                    first_binding_max=None, first_binding_histogram={})
        first = []
        for task in tasks:
            row = read(data / folder / f"{task['index']:02d}-{arm}.json")
            excess = [step["q"] - cap for step in row["trace"] if step["q"] > cap]
            stat["requests"] += 1
            stat["max_decoder_q"] = max(stat["max_decoder_q"], row["max_decoder_q"])
            if excess:
                stat["binding_requests"] += 1
                stat["excess_rows"] += sum(excess)
                first.append(next(i for i, step in enumerate(row["trace"])
                                  if step["q"] > cap))
        if first:
            stat["first_binding_min"] = min(first)
            stat["first_binding_max"] = max(first)
            stat["first_binding_histogram"] = {
                str(key): first.count(key) for key in sorted(set(first))}
        rows.append(stat)
    return rows


def verify_predecessor(data, after):
    """Do not start until the previous Gemma GPU0 job is actually complete."""
    if after != PREDECESSOR:
        return
    folder = data / "gemma-fixed512-mask-pair"
    for policy in ("active_w2_mask_on", "active_w2_mask_off"):
        require(load(folder / f"complete-{policy}.json",
                     f"Incomplete predecessor arm: {policy}") ==
                dict(policy=policy, requests=90),
                f"Incomplete predecessor arm: {policy}")
    qualified = load(folder / "qualified.json", "Missing predecessor qualification")
    require(qualified["requests"] == 90 and qualified["historical_replayed"] == 90,
            "Predecessor did not finish the paired replay")
    audit = load(folder / "pair-audit.json", "Missing predecessor pair audit")
    require(len(audit["cases"]) == 90, "Predecessor pair audit is incomplete")


def audit_row(row, original, cap):
    """Physical cap, initialization, admission and billing audit for one request."""
    from bdlm_serving.runtime.optimus_limits import validate_trace
    from experiments.t2t.serving_slots.external_report import check_first_cut

    validate_trace(row)
    require(row["policy"] in POLICIES and cap_of(row["policy"]) == cap,
            "Wrong fixed-cap policy")
    require(row["max_decoder_q"] <= cap, "Decoder exceeded the hard cap")
    require(row["generation_query_rows"] == row["generation_physical_padded_rows"],
            "Padded and query rows differ")
    require(row["generation_query_rows"] == sum(q for _, q in row["events"][1:]),
            "Billed Q differs from observed work")
    require(row["generation_forwards"] == len(row["events"]) - 1,
            "Forward ledger mismatch")
    require(row["events"][0][0] == "encoder" and row["events"][0][1] > 0,
            "Missing prompt prefill")
    require(all(tag == "decoder" or (tag == "encoder" and q == 256)
                for tag, q in row["events"][1:]),
            "Unexpected encoder or decoder work")
    dec = [q for tag, q in row["events"][1:] if tag == "decoder"]
    require(max(dec) == row["max_decoder_q"], "Decoder peak disagrees")
    require(max(q for _, q in row["events"][1:]) == row["max_q"], "Peak Q disagrees")
    for tick, step in enumerate(row["trace"]):
        require(step["tick"] == tick and step["offer"] == cap and 0 < step["q"] <= cap,
                "Wrong request-local decoder cap clock")
        require(step["q"] == len(step["positions"]) and step["proposed_q"] >= step["q"],
                "Selected rows disagree with the upstream plan")
        require(len(step["admitted"]) == step["accepted"] <= step["q"],
                "Admission ledger mismatch")
        require(set(step["admitted"]) <= set(step["positions"]),
                "Admitted an unselected row")
        require(step["cross"] == sum(i >= step["block"] * 256 + 256
                                     for i in step["positions"]),
                "OBS cross-block count mismatch")
        require(step["visible_end"] == max(step["positions"]) + 1,
                "Resident visible span mismatch")
    check_first_cut(original, row, cap)
    if not any(step["proposed_q"] > step["offer"] for step in row["trace"]):
        for key in ("output_ids", "events", "generation_forwards",
                    "generation_physical_padded_rows"):
            require(row[key] == original[key],
                    f"Nonbinding fixed control changed {key}")


def verify_run(folder, data, tasks):
    meta = read(folder / "run.json")
    require((meta["model"], meta["suite"], meta["samples_sha256"]) ==
            ("gemma", "frozen90", SAMPLES_SHA), "Wrong fixed-cap run")
    require(meta["hard_q_cap"] is True and
            meta["cap_clock"] == "all_decoder_forwards_request_local" and
            meta["cap_excludes"] == "encoder" and
            meta["obs_extent"] == "B+max(schedule)" and
            meta["actual_token_edits"] == "M2T only",
            "Wrong cap semantics")
    require(meta["capped_plans"] == [[32], [64]], "Wrong fixed-cap matrix")
    require(meta["policies"] == list(POLICIES), "Wrong fixed-cap policies")
    require(read(folder / "qualified-logits.json") == dict(
        modes=["ibs", "obs"], chunk=256, first_step_logits="bit_identical", index=0),
        "Missing first-step logits qualification")
    for policy in POLICIES:
        require(load(folder / f"complete-{policy}.json",
                     f"Incomplete fixed-cap arm: {policy}") ==
                dict(policy=policy, requests=len(tasks)),
                f"Incomplete fixed-cap arm: {policy}")
        cap = cap_of(policy)
        uncapped, arm = UNCAPPED[policy]
        for task in tasks:
            index = task["index"]
            row = load(folder / f"{index:02d}-{policy}.json",
                       f"Missing fixed-cap row: {policy}/{index}")
            require(all(row[k] == task[k] for k in ("index", "task_id", "seed", "dataset")),
                    f"Wrong case identity: {policy}/{index}")
            original = load(data / uncapped / f"{index:02d}-{arm}.json",
                            f"Missing uncapped row: {arm}/{index}")
            audit_row(row, original, cap)
    return meta


def run(args, before_cuda=None):
    """Adapt the protected launcher's Optimus call to the two fixed budgets."""
    from bdlm_serving.runtime import diffusiongemma_optimus as engine

    require(args.suite == "frozen90", "Fixed caps require frozen90")
    require(sha(args.samples) == SAMPLES_SHA, "Wrong frozen90 samples")
    require(list(args.modes) == ["ibs", "obs"], "Only IBS and OBS are registered")
    args.capped_plans = list(PLANS)
    args.uncapped90 = DATA
    args.chunks = [32, 64]
    engine.run(args, before_cuda)
    verify_run(args.output, DATA, read(args.samples)["samples"])


def mark_protected_complete(folder):
    payload = dict(status="complete", model="gemma", policies=list(POLICIES),
                   requests=90, gpu=0, holder_restored=True,
                   samples_sha256=SAMPLES_SHA, run=folder.name)
    (folder / "protected-complete.json").write_text(json.dumps(payload, indent=2))


def protected_entry():
    argv = sys.argv[1:]
    require(argv in (["gemma", "0"], ["gemma", "0", "worker"]),
            "Use the protected Gemma GPU0 entry: gemma 0")
    parent = argv == ["gemma", "0"]
    code = protected_source(LAUNCHER)
    if parent:
        sys.path.insert(0, str(REPO))
        verify_predecessor(DATA, PREDECESSOR)
    else:
        signal_dir = Path(os.environ["DG_SIGNAL"]).resolve(strict=True)
        require(signal_dir.parent == ROOT and signal_dir.name.startswith("control-0-"),
                "Worker requires a GPU0 staging directory from the protected launcher")
    os.environ.update(PREFIX_MODE=MODE, FROZEN90_SAMPLES=str(SAMPLES),
                      TRACE_OUT=str(DATA / OUTPUT))
    try:
        exec(code, {"__file__": str(HERE), "__name__": "__main__"})
    except SystemExit as error:
        if error.code not in (0, None):
            raise
    if parent:
        verify_run(DATA / OUTPUT, DATA, read(SAMPLES)["samples"])
        mark_protected_complete(DATA / OUTPUT)


def tmux_command(root):
    log = root / "prefix90" / (OUTPUT + "-chain.log")
    return ("tmux new-session -d -s prefix90-gemma-fixed-caps "
            f"\"cd {REPO} && PYTHONPATH=. ../venv/bin/python -u -m "
            f"experiments.t2t.serving_slots.gemma_fixed_caps queue gemma {root} "
            f"> {log} 2>&1\"")


def queue(root, after=PREDECESSOR):
    root = Path(root).resolve()
    data = root / "prefix90"
    require(sha(data / "samples.json") == SAMPLES_SHA, "Wrong frozen90 samples")
    tasks = read(data / "samples.json")["samples"]
    verify_sources()
    verify_references(data, tasks)
    out = data / OUTPUT
    require(not out.exists(), f"Output already exists: {out}")
    while True:
        check = subprocess.run(["tmux", "has-session", "-t", "=" + after],
                               capture_output=True)
        if check.returncode == 1:
            break
        check.check_returncode()
        print(f"Waiting on {after}", flush=True)
        time.sleep(20)
    verify_references(data, tasks)
    verify_predecessor(data, after)
    log = data / (OUTPUT + ".log")
    env = dict(os.environ, FROZEN90_SAMPLES=str(data / "samples.json"),
               TRACE_OUT=str(out))
    with log.open("x") as handle:
        print(f"Launching protected GPU0 fixed-cap handoff; worker log: {log}",
              flush=True)
        subprocess.run([sys.executable, "-u", str(HERE), "gemma", "0"], cwd=REPO,
                       env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    verify_run(out, data, tasks)
    print(f"Verified {len(POLICIES)} fixed-cap policies, 90 cases each: {out}",
          flush=True)


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(
            key for row in rows for key in row)), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def score(data, out):
    """Paired capped-vs-uncapped tables for the four fixed-cap arms."""
    deps = REPO.parent / "fixed-baseline-deps"
    if deps.is_dir() and str(deps) not in sys.path:
        sys.path.append(str(deps))
    from tools.prefix_cohorts import (_window_row, load_cases, load_request_set,
                                      select_window_cases)

    cases = load_cases(data / "samples.json")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for policy in POLICIES:
        cap = cap_of(policy)
        folder, arm = UNCAPPED[policy]
        baseline = {row["index"]: row for row in
                    load_request_set(data / folder, cases, "Gemma", arm)[0]}
        capped = {row["index"]: row for row in
                  load_request_set(data / OUTPUT, cases, "Gemma", policy)[0]}
        for window in (30, 60, 90):
            subset = select_window_cases(cases, window)
            for scope in ("all", "gsm8k", "humaneval", "math500"):
                row = _window_row(subset, baseline, capped, window, scope,
                                  "gemma", arm, policy, cap)
                row["source"] = "capped-vs-uncapped"
                rows.append(row)
    write_csv(out / "gemma-fixed-caps-paired.csv", rows)
    write_csv(out / "gemma-fixed-caps-neutrality.csv",
              neutrality(data, data / OUTPUT, cases))


def neutrality(data, folder, tasks):
    """Is the hard cap behaviourally neutral on requests where it binds?"""
    rows = []
    for policy in POLICIES:
        cap = cap_of(policy)
        uncapped, arm = UNCAPPED[policy]
        stat = dict(policy=policy, cap=cap, binding_requests=0, same_output=0,
                    same_events=0, same_work=0)
        for task in tasks:
            row = read(folder / f"{task['index']:02d}-{policy}.json")
            original = read(data / uncapped / f"{task['index']:02d}-{arm}.json")
            if any(step["proposed_q"] > step["offer"] for step in row["trace"]):
                stat["binding_requests"] += 1
                stat["same_output"] += row["output_ids"] == original["output_ids"]
                stat["same_events"] += row["events"] == original["events"]
                stat["same_work"] += (
                    row["generation_forwards"] == original["generation_forwards"] and
                    row["generation_physical_padded_rows"] ==
                    original["generation_physical_padded_rows"])
        rows.append(stat)
    return rows


def main():
    if sys.argv[1:2] == ["gemma"]:
        protected_entry()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "queue", "score"))
    parser.add_argument("model", choices=("gemma",))
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--after", default=PREDECESSOR,
                        help="tmux session that must finish first")
    args = parser.parse_args()
    root = args.runtime.resolve()
    if args.action == "check":
        data = root / "prefix90"
        tasks = read(data / "samples.json")["samples"]
        verify_sources()
        verify_references(data, tasks)
        for row in inspect_uncapped(data, tasks):
            print(json.dumps(row, sort_keys=True))
        print(tmux_command(root))
        print("CPU-only preflight; no launcher or CUDA command was invoked")
    elif args.action == "queue":
        queue(root, args.after)
    else:
        score(root / "prefix90", root / "prefix90" / "controls")


if __name__ == "__main__":
    main()

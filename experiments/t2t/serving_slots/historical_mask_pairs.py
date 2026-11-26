"""Historical LLaDA2.1 Optimus OBS 30-question mask on/off paired check.

Read-only w.r.t. archived artifacts and offline (CPU only).  It consumes the
aggregate table ``experiments/optimus-masked-obs/requests.csv`` only: no
request JSON, trace, KV state, checkpoint or model output is read, and no
model is loaded.

Case identity comes from the committed ``benchmarks/frozen30/manifest.json``
``samples`` array (index, dataset, task_id, seed); no external cluster sample
file is required.  All 30 identities must be present in both arms before any
number is emitted, and every archived row is checked against that manifest.

Arms, per fixed Optimus window W::

    baseline  obs_W   cross-block visibility left open
    arm       bobs_W  native B32 block-causal mask overlaid on OBS

This is the historical ``关 -> 开`` direction used by EXPERIMENT/EXPANDED.
Quality is the archived ``correct`` flag.  Failed requests are counted as wrong
and keep their work; a failed row with ``N=0`` is accepted (TPF 0) rather than
rejected or silently dropped.

Metrics, matching ``tools.prefix_cohorts`` except where noted:

* macro TPF = per-request ``N/F`` mean;
* token efficiency = pooled ``sum(N) / sum(physical_Q)``.  The denominator is
  the archived physical padded/projected rows, NOT the request-row ``Q``;
  ``*_request_row_efficiency`` columns keep the ``sum(N)/sum(Q)`` values and
  ``efficiency_denominator`` states ``physical_Q`` on every row.

The archived table has no intervention/trigger field.  Exposure is NOT
inferred, estimated or imputed here; every row carries ``exposure_recorded=no``.
Any exposure-conditioned claim must come from a run that recorded it.

Uncertainty reused from ``tools.prefix_cohorts``:
  * quality: Newcombe hybrid-score paired interval (Wilson square-and-add
    fallback at zero discordance), stratified by dataset;
  * macro TPF: stratified paired-normal 95% interval.
Added here:
  * exact two-sided McNemar p from rescued/lost;
  * leave-one-case-out range of the quality delta and the count of removals
    that flip its sign;
  * a per-window sign pattern over the three dataset scopes.

Usage (from the repo root, no GPU)::

    CUDA_VISIBLE_DEVICES= python experiments/t2t/serving_slots/historical_mask_pairs.py
    CUDA_VISIBLE_DEVICES= python experiments/t2t/serving_slots/historical_mask_pairs.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                      # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.prefix_cohorts import _window_row
from tools.t2t_results import table

ARCHIVE = ROOT / "experiments/optimus-masked-obs/requests.csv"
CASES = ROOT / "benchmarks/frozen30/manifest.json"
DEFAULT_OUT = HERE / "controls/llada21-obs30-mask-paired.csv"
MODEL = "llada21"
DATASETS = ("gsm8k", "math500", "humaneval")
WINDOWS = (4, 8, 16, 32)
SCOPES = ("all",) + DATASETS
BOOLS = {"0": False, "1": True, "true": True, "false": False}


def load_cases(path: Path = CASES):
    """Load and validate the 30 identities from the committed manifest."""
    payload = json.loads(Path(path).read_text())
    rows = payload.get("samples")
    if not isinstance(rows, list) or len(rows) != 30:
        raise ValueError(f"{path}: expected a 30-entry samples array")
    seen, per_dataset = set(), Counter()
    for raw in rows:
        index = raw.get("index")
        if type(index) is not int or not 0 <= index < 30:
            raise ValueError(f"{path}: bad sample index {index!r}")
        if index in seen:
            raise ValueError(f"{path}: duplicate sample index {index}")
        for key in ("dataset", "task_id"):
            if not isinstance(raw.get(key), str) or not raw[key]:
                raise ValueError(f"{path}: sample {index}: missing {key}")
        if type(raw.get("seed")) is not int:
            raise ValueError(f"{path}: sample {index}: seed must be an integer")
        seen.add(index)
        per_dataset[raw["dataset"]] += 1
    if seen != set(range(30)):
        raise ValueError(f"{path}: sample indices are not exactly 0..29")
    if set(per_dataset) != set(DATASETS) or any(n != 10 for n in per_dataset.values()):
        raise ValueError(f"{path}: expected 10 per dataset, got {dict(per_dataset)}")
    return [dict(raw) for raw in sorted(rows, key=lambda r: r["index"])]


def _flag(raw, name, key):
    value = raw[name]
    if value not in BOOLS:
        raise ValueError(f"{key}: non-boolean {name}={value!r}")
    return int(BOOLS[value])


def _positive(raw, name, key):
    value = raw[name]
    if not value.isdigit() or int(value) <= 0:
        raise ValueError(f"{key}: {name} must be a positive integer, got {value!r}")
    return int(value)


def load_archive(path: Path, cases):
    """Return {(policy, index): normalized row} for the archived LLaDA30 table.

    Rows carry every field ``tools.t2t_results.summarize`` reads.  ``max_q`` is
    not archived and is not used by any reported column, so it stays 0.
    """
    case_by_index = {case["index"]: case for case in cases}
    rows = {}
    with Path(path).open(newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw["model"] != MODEL or int(raw["index"]) not in case_by_index:
                continue
            index = int(raw["index"])
            case = case_by_index[index]
            key = f"{raw['policy']}#{index}"
            for name in ("dataset", "task_id"):
                if raw[name] != case[name]:
                    raise ValueError(f"{key}: {name} {raw[name]!r} != {case[name]!r}")
            if key in rows:
                raise ValueError(f"{key}: duplicate archived request")
            n = int(raw["N"])
            f = _positive(raw, "F", key)
            q = _positive(raw, "Q", key)
            physical = _positive(raw, "physical_Q", key)
            if not 0 <= n <= q:
                raise ValueError(f"{key}: N={n} outside [0, Q={q}]")
            rows[key] = dict(
                index=index, dataset=raw["dataset"], task_id=raw["task_id"],
                seed=case["seed"], model=MODEL, policy=raw["policy"],
                max_q=0, N=n, F=f, Q=q, physical_Q=physical,
                tpf=n / f, token_efficiency=n / q,
                correct=_flag(raw, "correct", key),
                failed=_flag(raw, "failed", key),
                truncated=_flag(raw, "truncated", key))
    return rows


def require_all_30(archive, policy):
    missing = [i for i in range(30) if f"{policy}#{i}" not in archive]
    if missing:
        raise ValueError(f"{policy}: missing case indices {missing}")
    return {i: archive[f"{policy}#{i}"] for i in range(30)}


def mcnemar_exact(rescued: int, lost: int) -> float:
    """Two-sided exact binomial McNemar p (no continuity correction)."""
    n = rescued + lost
    if n == 0:
        return 1.0
    k = min(rescued, lost)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def loo_quality(cases, baseline, arm):
    """Leave-one-case-out range of the paired quality delta and sign flips."""
    deltas = []
    for drop in range(len(cases)):
        keep = [c for i, c in enumerate(cases) if i != drop]
        deltas.append(
            sum(arm[c["index"]]["correct"] - baseline[c["index"]]["correct"] for c in keep)
            / len(keep))
    full = (sum(arm[c["index"]]["correct"] - baseline[c["index"]]["correct"] for c in cases)
            / len(cases))
    flips = sum(1 for value in deltas
                if (value > 0) != (full > 0) or (value == 0 and full != 0))
    return min(deltas), max(deltas), flips


def build_rows(cases, archive, windows):
    if [c["index"] for c in cases] != list(range(30)):
        raise ValueError("Cases must be exactly indices 0..29")
    rows = []
    for window in windows:
        baseline_policy, arm_policy = f"obs_{window}", f"bobs_{window}"
        baseline = require_all_30(archive, baseline_policy)
        arm = require_all_30(archive, arm_policy)
        per_window = []
        for scope in SCOPES:
            row = _window_row(cases, baseline, arm, 30, scope, MODEL,
                              baseline_policy, arm_policy, None)
            subset = [c for c in cases if scope == "all" or c["dataset"] == scope]
            lo, hi, flips = loo_quality(subset, baseline, arm)
            row = dict(model=row.pop("model"), optimus_window=window, **row)
            # Physical-Q token efficiency is the reported one; keep the
            # request-row ratio under an explicit name.
            row["baseline_request_row_efficiency"] = row["baseline_pooled_token_efficiency"]
            row["arm_request_row_efficiency"] = row["arm_pooled_token_efficiency"]
            row["baseline_pooled_token_efficiency"] = (
                row["baseline_N"] / row["baseline_physical_Q"])
            row["arm_pooled_token_efficiency"] = (
                row["arm_N"] / row["arm_physical_Q"])
            row["pooled_token_efficiency_delta"] = (
                row["arm_pooled_token_efficiency"]
                - row["baseline_pooled_token_efficiency"])
            row.update(
                efficiency_denominator="physical_Q",
                exposure_recorded="no",
                mcnemar_exact_p=mcnemar_exact(row["rescued"], row["lost"]),
                quality_delta_loo_min=lo,
                quality_delta_loo_max=hi,
                quality_delta_loo_sign_flips=flips,
            )
            per_window.append(row)
        pattern = "".join(
            "+" if r["quality_delta"] > 0 else "-" if r["quality_delta"] < 0 else "0"
            for r in per_window[1:])
        for row in per_window:
            row["dataset_sign_pattern"] = pattern
        rows.extend(per_window)
    return rows


def selftest():
    """Offline contract check: manifest identity, failed N=0, paired math."""
    cases = load_cases()
    assert len(cases) == 30 and [c["index"] for c in cases] == list(range(30))
    assert Counter(c["dataset"] for c in cases) == Counter(
        {"gsm8k": 10, "math500": 10, "humaneval": 10})
    assert len({c["seed"] for c in cases}) == 30
    archive = load_archive(ARCHIVE, cases)
    failed = [r for r in archive.values() if r["failed"]]
    assert failed and all(r["N"] == 0 and r["tpf"] == 0 for r in failed), failed
    assert archive["obs_32#5"]["failed"] == 1 and archive["obs_32#5"]["N"] == 0
    assert abs(mcnemar_exact(5, 1) - 0.21875) < 1e-12
    assert mcnemar_exact(0, 0) == 1.0
    try:
        build_rows(cases, {k: v for k, v in archive.items() if k != "bobs_16#7"}, [16])
    except ValueError as error:
        assert "missing case indices [7]" in str(error), error
    else:
        raise AssertionError("missing identity was accepted")

    mini = cases[:6]
    base = {c["index"]: dict(index=c["index"], dataset=c["dataset"],
                             task_id=c["task_id"], seed=c["seed"],
                             model=MODEL, policy="obs", max_q=0,
                             N=10, F=2, Q=5, physical_Q=5, tpf=5.0,
                             token_efficiency=2.0, correct=1, failed=0,
                             truncated=0) for c in mini}
    arm = {k: dict(v, correct=(0 if k == 1 else 1),
                   N=(8 if k == 0 else 10), tpf=(4.0 if k == 0 else 5.0))
           for k, v in base.items()}
    row = _window_row(mini, base, arm, 30, "all", MODEL, "obs", "bobs", None)
    assert row["rescued"] == 0 and row["lost"] == 1, row
    assert abs(row["quality_delta"] + 1 / 6) < 1e-12
    lo, hi, flips = loo_quality(mini, base, arm)
    assert (lo, hi) == (-1 / 5, 0.0) and flips == 1, (lo, hi, flips)
    print("selftest OK")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--cases", "--samples", dest="cases", type=Path, default=CASES,
                        help="frozen30 manifest; its samples array supplies identity and seed")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--windows", type=int, nargs="+", default=list(WINDOWS))
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    cases = load_cases(args.cases)
    archive = load_archive(args.archive, cases)
    rows = build_rows(cases, archive, args.windows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table(args.out, rows)
    for row in rows:
        if row["scope"] == "all":
            print(f"W{row['optimus_window']:>2} {row['baseline_policy']:>7}->"
                  f"{row['arm_policy']:<8} {row['baseline_correct']:>2}->"
                  f"{row['arm_correct']:<2} r/l={row['rescued']}/{row['lost']} "
                  f"dQ={row['quality_delta']:+.3f} "
                  f"[{row['quality_delta_ci_low']:+.3f},{row['quality_delta_ci_high']:+.3f}] "
                  f"p={row['mcnemar_exact_p']:.4f} "
                  f"dTPF={row['macro_tpf_delta']:+.3f} "
                  f"eff={row['baseline_pooled_token_efficiency']:.4f}->"
                  f"{row['arm_pooled_token_efficiency']:.4f} (physical_Q)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

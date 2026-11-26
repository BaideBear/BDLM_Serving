"""Paired on-minus-off table for Gemma ``unified90`` mask-only runs.

Reuses ``unified90.paired`` (mask-only pre-intervention check) and the
committed scorer; the aggregate matches ``unified90.score`` without writing
into the run folder.  Intervals reuse ``tools.prefix_cohorts.paired_ci``/
``quality_ci`` and ``mask_pair_report.efficiency_interval``.  CPU
``cpu-pilot`` runs are exact ``pilot3`` counts only (no intervals); GPU runs
keep original30/new60/full90.  Only aggregate tables are written; no per-case
exports.

Usage: ``python unified_mask_score.py --group RUN_FOLDER [--out OUT_FOLDER]``
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from bdlm_serving.evaluation.scoring import score
from experiments.t2t.serving_slots import unified90
from experiments.t2t.serving_slots.mask_pair_report import efficiency_interval
from tools.prefix_cohorts import paired_ci, quality_ci, TPF_CI_METHOD, QUALITY_CI_METHOD
from tools.t2t_results import table

PILOT = (0, 10, 20)
COHORTS = {"original30": (0, 30), "new60": (30, 90), "full90": (0, 90)}
SCOPES = ("all", "exposed", "unexposed")
EFFICIENCY_CI_METHOD = "stratified paired percentile bootstrap (mask_pair_report)"


def read(path):
    return json.loads(Path(path).read_text())


def _spec(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("Malformed arm spec")
    series, causal, prefix = value
    if (not isinstance(series, (list, tuple)) or type(causal) is not bool
            or not isinstance(prefix, dict)):
        raise ValueError("Malformed arm spec")
    return tuple(series), causal, dict(prefix)


def load_pair(folder, *, tasks=None):
    meta = read(Path(folder) / "run.json")
    suite = meta.get("suite")
    if suite not in ("cpu-pilot", "frozen90"):
        raise ValueError(f"Unsupported suite: {suite!r}")
    if meta.get("model") != "diffusiongemma":
        raise ValueError(f"Not a diffusiongemma run: {meta.get('model')!r}")
    if meta.get("samples_sha256") != unified90.caps.SAMPLES_SHA:
        raise ValueError("Run samples_sha256 is not the frozen90 cohort")
    pilot = suite == "cpu-pilot"
    if meta.get("device") != ("cpu" if pilot else "cuda:0"):
        raise ValueError("Suite/device mismatch")
    if not pilot and read(Path(folder) / "returned.json") != {"gpu": 0, "restored": True}:
        raise ValueError("Missing protected GPU return")
    indices = list(PILOT if pilot else range(90))
    if meta.get("indices") != indices:
        raise ValueError(f"{suite} must have indices {indices}; got {meta.get('indices')}")
    arms = meta.get("arms")
    if not isinstance(arms, dict) or len(arms) != 2:
        raise ValueError(f"Expected exactly one on/off pair; got arms {arms!r}")
    specs = {}
    for policy, value in arms.items():
        if policy not in unified90.ARMS:
            raise ValueError(f"Unregistered policy: {policy!r}")
        spec = _spec(value)
        if spec != _spec(unified90.ARMS[policy]):
            raise ValueError(f"Run arm {policy} disagrees with unified90.ARMS")
        specs[policy] = spec
    on = [policy for policy, spec in specs.items() if spec[1]]
    off = [policy for policy, spec in specs.items() if not spec[1]]
    if len(on) != 1 or len(off) != 1:
        raise ValueError(
            f"Need exactly one causal-on and one causal-off arm; got on={on} off={off}")
    on, off = on[0], off[0]
    if specs[on][0] != specs[off][0] or specs[on][2] != specs[off][2]:
        raise ValueError("Not a mask-only pair: offers or prefix quotas differ")
    for policy in (on, off):
        path = Path(folder) / f"complete-{policy}.json"
        if not path.is_file() or read(path) != dict(
                policy=policy, requests=len(indices)):
            raise ValueError(f"Incomplete arm {policy}")
    if tasks is None:
        if unified90.sha(unified90.caps.SAMPLES) != meta["samples_sha256"]:
            raise ValueError("Frozen samples file changed")
        tasks = unified90.load_tasks(unified90.caps.SAMPLES)
    by_index = {task["index"]: task for task in tasks}
    return meta, pilot, on, off, indices, by_index


def pair_cases(folder, on, off, indices, tasks, scorer=score):
    rows = []
    for index in indices:
        a = read(Path(folder) / f"{index:02d}-{on}.json")
        b = read(Path(folder) / f"{index:02d}-{off}.json")
        for arm, r in ((on, a), (off, b)):
            if r["policy"] != arm or r["status"] != "success" or any(
                    r[k] != tasks[index][k] for k in unified90.KEYS):
                raise ValueError(f"Wrong frozen input or incomplete row at {index}:{arm}")
            n, f, q = r["output_tokens"], r["generation_forwards"], r["generation_physical_padded_rows"]
            if not (0 <= n and 0 < f <= q and q == r["encoder_q"] + r["decoder_q"]
                    and r["tpf"] == n/f and r["token_efficiency"] == n/q
                    and sum(e["q"] for e in r["trace"]) == r["decoder_q"]
                    and r["encoder_q"] % 256 == 0 and f == len(r["trace"]) + r["encoder_q"]//256):
                raise ValueError(f"Invalid physical work at {index}:{arm}")
        unified90.paired(a, b)
        rows.append(dict(
            index=index, dataset=tasks[index]["dataset"],
            on_truncated=a["truncated"], off_truncated=b["truncated"],
            on_correct=bool(scorer(a["output_text"], tasks[index])["correct"]),
            off_correct=bool(scorer(b["output_text"], tasks[index])["correct"]),
            on_N=a["output_tokens"], off_N=b["output_tokens"],
            on_F=a["generation_forwards"], off_F=b["generation_forwards"],
            on_Q=a["generation_physical_padded_rows"],
            off_Q=b["generation_physical_padded_rows"],
            on_tpf=a["tpf"], off_tpf=b["tpf"],
            exposed=a["first_opportunity"] is not None))
    return rows


def aggregate(rows):
    on_n, off_n = sum(r["on_N"] for r in rows), sum(r["off_N"] for r in rows)
    on_q, off_q = sum(r["on_Q"] for r in rows), sum(r["off_Q"] for r in rows)
    on_correct, off_correct = (sum(r["on_correct"] for r in rows),
                               sum(r["off_correct"] for r in rows))
    return dict(requests=len(rows), exposed=sum(r["exposed"] for r in rows),
                on_correct=on_correct, off_correct=off_correct,
                on_truncated=sum(r["on_truncated"] for r in rows),
                off_truncated=sum(r["off_truncated"] for r in rows),
                rescued=sum(r["on_correct"] and not r["off_correct"] for r in rows),
                lost=sum(r["off_correct"] and not r["on_correct"] for r in rows),
                on_macro_tpf=mean(r["on_tpf"] for r in rows),
                off_macro_tpf=mean(r["off_tpf"] for r in rows),
                on_eff=on_n / on_q, off_eff=off_n / off_q,
                on_N=on_n, off_N=off_n, on_F=sum(r["on_F"] for r in rows),
                off_F=sum(r["off_F"] for r in rows), on_Q=on_q, off_Q=off_q)


def intervals(rows, pilot):
    if pilot:
        return dict(tpf=None, quality=None, efficiency=None,
                    method="pilot3 exact counts only")
    tpf_groups = defaultdict(list)
    quality_counts = defaultdict(lambda: [0, 0, 0, 0])
    for row in rows:
        tpf_groups[row["dataset"]].append(row["on_tpf"] - row["off_tpf"])
        cell = quality_counts[row["dataset"]]
        cell[0 if row["on_correct"] and row["off_correct"]
             else 1 if row["off_correct"] else 2 if row["on_correct"] else 3] += 1
    cases = [dict(index=r["index"], dataset=r["dataset"]) for r in rows]
    off = {r["index"]: dict(N=r["off_N"], physical_Q=r["off_Q"]) for r in rows}
    on = {r["index"]: dict(N=r["on_N"], physical_Q=r["on_Q"]) for r in rows}

    def safe(fn, *args):
        try:
            return fn(*args)
        except ValueError:
            return None

    return dict(tpf=safe(paired_ci, tpf_groups),
                quality=safe(quality_ci, quality_counts),
                efficiency=safe(efficiency_interval, cases, off, on),
                method=f"TPF: {TPF_CI_METHOD}; quality: {QUALITY_CI_METHOD}; "
                       f"efficiency: {EFFICIENCY_CI_METHOD}")


def _ci(ci, key, index):
    return None if ci[key] is None else ci[key][index]


def _row(cohort, scope, agg, ci):
    return dict(
        cohort=cohort, scope=scope, requests=agg["requests"],
        exposed=agg["exposed"], unexposed=agg["requests"] - agg["exposed"],
        on_correct=agg["on_correct"], off_correct=agg["off_correct"],
        on_truncated=agg["on_truncated"], off_truncated=agg["off_truncated"],
        rescued=agg["rescued"], lost=agg["lost"],
        on_macro_tpf=agg["on_macro_tpf"], off_macro_tpf=agg["off_macro_tpf"],
        macro_tpf_delta=agg["on_macro_tpf"] - agg["off_macro_tpf"],
        macro_tpf_ci_low=_ci(ci, "tpf", 2), macro_tpf_ci_high=_ci(ci, "tpf", 3),
        on_token_efficiency=agg["on_eff"], off_token_efficiency=agg["off_eff"],
        token_efficiency_delta=agg["on_eff"] - agg["off_eff"],
        token_efficiency_ci_low=_ci(ci, "efficiency", 0),
        token_efficiency_ci_high=_ci(ci, "efficiency", 1),
        on_N=agg["on_N"], off_N=agg["off_N"], on_F=agg["on_F"], off_F=agg["off_F"],
        on_physical_Q=agg["on_Q"], off_physical_Q=agg["off_Q"],
        delta_N=agg["on_N"] - agg["off_N"], delta_F=agg["on_F"] - agg["off_F"],
        delta_physical_Q=agg["on_Q"] - agg["off_Q"],
        quality_delta=agg["on_correct"] / agg["requests"]
        - agg["off_correct"] / agg["requests"],
        quality_ci_low=_ci(ci, "quality", 2), quality_ci_high=_ci(ci, "quality", 3),
        ci_method=ci["method"])


def table_rows(rows, pilot):
    out = []
    for cohort in (["pilot3"] if pilot else list(COHORTS)):
        low, high = (0, 90) if pilot else COHORTS[cohort]
        subset = [r for r in rows if low <= r["index"] < high]
        for scope in SCOPES:
            scoped = [r for r in subset
                      if scope == "all" or r["exposed"] == (scope == "exposed")]
            if scoped:
                out.append(_row(cohort, scope, aggregate(scoped),
                                intervals(scoped, pilot)))
    return out


def by_dataset(rows):
    """Descriptive paired effects, keeping the original30 nested in full90."""
    if sorted(r["index"] for r in rows) != list(range(90)):
        raise ValueError("Expected one pair per frozen90 request")
    out = []
    for cohort, (lo, hi) in COHORTS.items():
        for dataset in ("gsm8k", "math500", "humaneval"):
            matched = [r for r in rows if lo <= r["index"] < hi and r["dataset"] == dataset]
            if len(matched) != (hi-lo)//3:
                raise ValueError("Unbalanced frozen cohort")
            out.append(dict(cohort=cohort, dataset=dataset, **aggregate(matched)))
    return out


def offer_work(folder, policy, indices):
    """Attribute paid work to each decoder offer, including encoder calls it triggered."""
    series = unified90.ARMS[policy][0]
    totals = {}
    for index in indices:
        row = read(Path(folder) / f"{index:02d}-{policy}.json")
        f = q = 0
        for tick, e in enumerate(row["trace"]):
            b = e["offer"]
            if (e["tick"] != tick or b != series[tick % len(series)]
                    or not 0 <= e["parked"] <= e["q"] or type(e["opportunity"]) is not bool):
                raise ValueError("Wrong trace clock, parked rows or opportunity")
            reenc = len(e["encoded"])
            f += 1 + reenc
            q += e["q"] + 256 * reenc
            for cohort in ("original30" if index < 30 else "new60", "full90"):
                t = totals.setdefault((cohort,b), defaultdict(int))
                t["decoder_f"] += 1
                t["decoder_q"] += e["q"]
                t["encoder_f"] += reenc
                t["encoder_q"] += 256 * reenc
                t["exposed_ticks"] += e["opportunity"]
                # Below 512 the tail is never sampled: zero refresh rows
                # means zero tail projections, even when tail KV is visible.
                t["exposed_low_cached_only"] += e["opportunity"] and b < 512 and e["parked"] == 0
                t["exposed_low_with_refresh"] += e["opportunity"] and b < 512 and e["parked"] > 0
        if (f,q) != (row["generation_forwards"],row["generation_physical_padded_rows"]):
            raise ValueError("Offer work does not reconstruct the request")
    return [dict(policy=policy, cohort=c, offer=b, **t) for (c,b),t in sorted(totals.items())]


def write(out, meta, rows):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [dict(on_policy=meta["on"], off_policy=meta["off"], **r) for r in rows]
    table(out / "paired.csv", rows)
    (out / "audit.json").write_text(json.dumps(dict(
        group=str(meta["folder"]), suite=meta["suite"], on=meta["on"], off=meta["off"],
        indices=meta["indices"], samples_sha256=meta["samples_sha256"],
        ci_methods=dict(tpf=TPF_CI_METHOD, quality=QUALITY_CI_METHOD,
                        efficiency=EFFICIENCY_CI_METHOD),
        table=rows), indent=2, sort_keys=True) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    meta, pilot, on, off, indices, tasks = load_pair(args.group)
    rows = pair_cases(args.group, on, off, indices, tasks)
    out = args.out or args.group.with_name(args.group.name + "-paired")
    write(out, dict(folder=args.group, suite="cpu-pilot" if pilot else "frozen90",
                    on=on, off=off, indices=indices,
                    samples_sha256=meta["samples_sha256"]),
          table_rows(rows, pilot))
    if not pilot:
        table(out / "by-dataset.csv", by_dataset(rows))
        table(out / "offer-work.csv", offer_work(args.group, off, indices)
              + offer_work(args.group, on, indices))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

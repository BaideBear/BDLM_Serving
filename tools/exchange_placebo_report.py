"""Matched-victim candidate-randomization report for the LLaDA exchange study.

This report consumes published ``tools.t2t_results`` request and summary tables.
It does not rescore outputs and does not read raw generation events.  Raw-event
checks (full-width parity, gate/victim match, first divergence, swap type and
strict edit of the added row) belong to the separate raw-event audit.

Inputs
------
``--measured``
    Directory written by ``tools.t2t_results`` for this round.  It must contain
    the two placebo arms ``hard_exchange_random_a`` and
    ``hard_exchange_random_b`` at every requested Q.
``--ranked``
    Published exchange directory (default ``experiments/t2t/exchange``).  It
    provides the historical ``hard_exchange_edit4`` arm and, when present, the
    zero-reserve control.
``--out``
    Output directory.

The zero-reserve control is resolved in this order: ``--ranked`` requests,
``--measured`` requests, then the sibling ``hard`` study requests
(``<ranked>/../hard/<model>-online-requests.csv``), which is where the published
exchange study keeps its historical no-reserve control.  The resolved source is
recorded in every output table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from tools.t2t_results import summarize, table


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RANKED = ROOT / "experiments/t2t/exchange"
DEFAULT_BASELINES = ROOT / "experiments/baselines/summary.csv"
DEFAULT_MANIFEST = ROOT / "benchmarks/frozen30/manifest.json"

MODEL_LABEL = {"llada21": "LLaDA 2.1", "gemma": "DiffusionGemma"}
DATASETS = ("gsm8k", "math500", "humaneval")
ARM_ORDER = ("zero", "ranked", "placebo_a", "placebo_b")
ARM_LABEL = {"zero": "zero reserve", "ranked": "ranked edit4",
             "placebo_a": "placebo a", "placebo_b": "placebo b"}
ARM_SUFFIX = {"zero": "hard_noedit", "ranked": "hard_exchange_edit4",
              "placebo_a": "hard_exchange_random_a",
              "placebo_b": "hard_exchange_random_b"}
FIG_LABEL = {"zero": "zero", "ranked": "ranked edit4",
             "placebo_a": "placebo a", "placebo_b": "placebo b"}
REFERENCES = ("zero", "placebo_a", "placebo_b")


@dataclass
class Arm:
    name: str
    q: int
    policy: str
    source: str
    requests: list
    summary: dict


@dataclass
class Report:
    model: str
    qs: list
    arms: dict
    paired: dict
    paired_requests: list
    context: dict
    checks: list
    manifest: Path
    measured: Path
    ranked: Path
    baselines: Path | None


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def display_path(path, root=ROOT):
    """Return a repo-relative path when possible, otherwise only the basename."""
    try:
        return str(Path(path).resolve().relative_to(root))
    except ValueError:
        return Path(path).name


def identity(row):
    return int(row["index"]), row["dataset"], row["task_id"]


def load_manifest(path):
    data = json.loads(path.read_text())
    samples = data["samples"]
    keys = [(int(s["index"]), s["dataset"], s["task_id"]) for s in samples]
    if len(keys) != 30 or len(set(keys)) != 30:
        raise ValueError(f"{path}: expected 30 distinct frozen requests")
    if sorted(k[0] for k in keys) != list(range(30)):
        raise ValueError(f"{path}: frozen request indices are not 0..29")
    counts = {name: sum(1 for k in keys if k[1] == name)
              for name in ("gsm8k", "math500", "humaneval")}
    if counts != {"gsm8k": 10, "math500": 10, "humaneval": 10}:
        raise ValueError(f"{path}: expected 10 requests per dataset, got {counts}")
    return keys


def request_tpf(row):
    return row["N"] / row["F"]


def summary_of(rows):
    summary = summarize(rows)
    if summary["physical_Q"] <= 0:
        raise ValueError("Physical Q must be positive")
    summary["physical_efficiency"] = summary["N"] / summary["physical_Q"]
    summary["physical_q_per_f"] = summary["physical_Q"] / summary["F"]
    return summary


def coerce_request(row, model, policy, q, mask, path):
    try:
        index = int(row["index"])
        nominal_q = int(row["nominal_q"])
        fresh = int(row["fresh"])
        mask_value = int(row["mask"])
        edit = int(row["edit"])
        max_q = int(row["max_q"])
        n, f, query, physical = (int(row[k]) for k in ("N", "F", "Q", "physical_Q"))
        correct, failed, truncated = (int(row[k]) for k in ("correct", "failed", "truncated"))
        dataset, task_id = row["dataset"], row["task_id"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: malformed request row {row!r}: {exc}") from exc
    if row.get("model") != model or row.get("policy") != policy:
        raise ValueError(f"{path}: request belongs to {row.get('model')}/{row.get('policy')}, "
                         f"expected {model}/{policy}")
    if nominal_q != q or fresh != 0 or mask_value != mask:
        raise ValueError(f"{path}: {policy} has Q={nominal_q} fresh={fresh} mask={mask_value}; "
                         f"expected Q={q} fresh=0 mask={mask}")
    if edit != q - fresh - mask_value:
        raise ValueError(f"{path}: {policy} has edit={edit}, expected {q - fresh - mask_value}")
    if min(n, f, query, physical, max_q) < 0:
        raise ValueError(f"{path}: {policy} has a negative work count")
    if f <= 0 or query <= 0:
        raise ValueError(f"{path}: {policy} request {index} has F={f}, Q={query}")
    if physical < query:
        raise ValueError(f"{path}: {policy} request {index} has physical Q {physical} < Q {query}")
    if any(value not in (0, 1) for value in (correct, failed, truncated)):
        raise ValueError(f"{path}: {policy} request {index} has a non-binary outcome flag")
    if correct and failed:
        raise ValueError(f"{path}: {policy} request {index} is both correct and failed")
    return dict(model=model, policy=policy, nominal_q=q, fresh=fresh, mask=mask_value,
                edit=edit, index=index, dataset=dataset, task_id=task_id, max_q=max_q,
                N=n, F=f, Q=query, physical_Q=physical, correct=correct,
                failed=failed, truncated=truncated)


def check_summary(directory, model, policy, summary, requests_path):
    path = directory / f"{model}-online-summary.csv"
    if not path.is_file():
        raise ValueError(f"Missing summary table: {path}")
    rows = [r for r in read_csv(path)
            if r.get("model") == model and r.get("policy") == policy
            and r.get("dataset") == "all"]
    if len(rows) != 1:
        raise ValueError(f"{path}: expected one dataset=all row for {policy}, found {len(rows)}")
    row = rows[0]
    try:
        for key in ("requests", "correct", "failed", "truncated", "N", "F", "Q",
                    "physical_Q", "max_q"):
            if int(row[key]) != summary[key]:
                raise ValueError(f"{path}: {policy} {key}={row[key]} disagrees with "
                                 f"requests ({summary[key]})")
        for key in ("tpf", "token_efficiency", "q_per_f"):
            if not math.isclose(float(row[key]), float(summary[key]),
                                rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"{path}: {policy} {key}={row[key]} disagrees with "
                                 f"requests ({summary[key]})")
    except KeyError as exc:
        raise ValueError(f"{path}: {policy} summary is missing {exc.args[0]!r}") from exc


def find_arm(directory, model, q, arm, expected, source):
    suffix = ARM_SUFFIX[arm]
    mask = q if arm == "zero" else q - 1
    path = directory / f"{model}-online-requests.csv"
    if not path.is_file():
        return None
    matches = [r for r in read_csv(path)
               if r.get("model") == model and r.get("policy", "").endswith("_" + suffix)
               and str(r.get("nominal_q")) == str(q)]
    if not matches:
        return None
    policies = sorted({r["policy"] for r in matches})
    if len(policies) != 1:
        raise ValueError(f"{path}: Q{q} has ambiguous {arm} policies: {policies}")
    policy = policies[0]
    if len(matches) != len(expected):
        raise ValueError(f"{path}: {policy} has {len(matches)} requests, expected {len(expected)}")
    typed = [coerce_request(r, model, policy, q, mask, path) for r in matches]
    by_key = {}
    for row in typed:
        key = (row["index"], row["dataset"], row["task_id"])
        if key in by_key:
            raise ValueError(f"{path}: duplicate request {key} in {policy}")
        by_key[key] = row
    expected_set = set(expected)
    missing = [key for key in expected if key not in by_key]
    extra = [key for key in by_key if key not in expected_set]
    if missing or extra:
        raise ValueError(f"{path}: {policy} does not match frozen30 "
                         f"(missing={missing[:3]}, extra={extra[:3]})")
    ordered = [by_key[key] for key in expected]
    summary = summary_of(ordered)
    check_summary(directory, model, policy, summary, path)
    return Arm(name=arm, q=q, policy=policy, source=source, requests=ordered, summary=summary)


def resolve_zero(measured, ranked, model, q, expected):
    for source, directory in (("ranked", ranked), ("measured", measured),
                              ("hard", ranked.parent / "hard")):
        arm = find_arm(directory, model, q, "zero", expected, source)
        if arm is not None:
            return arm
    raise ValueError(f"Missing zero-reserve control at Q{q}: looked in "
                     f"{ranked}, {measured}, {ranked.parent / 'hard'}")


def resolve_ranked(measured, ranked, model, q, expected):
    for source, directory in (("ranked", ranked), ("measured", measured)):
        arm = find_arm(directory, model, q, "ranked", expected, source)
        if arm is not None:
            return arm
    raise ValueError(f"Missing ranked hard_exchange_edit4 at Q{q}: looked in {ranked}, {measured}")


def resolve_placebo(measured, model, q, arm, expected):
    found = find_arm(measured, model, q, arm, expected, "measured")
    if found is None:
        raise ValueError(f"Missing {ARM_SUFFIX[arm]} at Q{q} in {measured}")
    return found


def pair(ranked, reference, reference_name, q):
    per = []
    for a, b in zip(ranked.requests, reference.requests, strict=True):
        if (a["index"], a["dataset"], a["task_id"]) != (b["index"], b["dataset"], b["task_id"]):
            raise ValueError(f"Cannot pair different tasks at Q{q}")
        atpf, btpf = request_tpf(a), request_tpf(b)
        per.append(dict(
            model=a["model"], nominal_q=q, reference=reference_name,
            index=a["index"], dataset=a["dataset"], task_id=a["task_id"],
            ranked_correct=a["correct"], reference_correct=b["correct"],
            win=int(a["correct"] and not b["correct"]),
            loss=int(b["correct"] and not a["correct"]),
            delta_correct=a["correct"] - b["correct"],
            ranked_N=a["N"], reference_N=b["N"], delta_N=a["N"] - b["N"],
            ranked_F=a["F"], reference_F=b["F"], delta_F=a["F"] - b["F"],
            ranked_Q=a["Q"], reference_Q=b["Q"], delta_Q=a["Q"] - b["Q"],
            ranked_physical_Q=a["physical_Q"], reference_physical_Q=b["physical_Q"],
            delta_physical_Q=a["physical_Q"] - b["physical_Q"],
            ranked_tpf=atpf, reference_tpf=btpf, delta_tpf=atpf - btpf))
    wins = sum(r["win"] for r in per)
    losses = sum(r["loss"] for r in per)
    both = sum(r["ranked_correct"] and r["reference_correct"] for r in per)
    deltas = [r["delta_tpf"] for r in per]
    summary = dict(
        model=ranked.requests[0]["model"], nominal_q=q, dataset="all",
        reference=reference_name, reference_policy=reference.policy,
        reference_source=reference.source,
        requests=len(per), only_ranked_correct=wins, only_reference_correct=losses,
        both_correct=both, neither_correct=len(per) - wins - losses - both,
        delta_correct=wins - losses, delta_tpf=mean(deltas),
        delta_tpf_positive=sum(d > 0 for d in deltas),
        delta_tpf_negative=sum(d < 0 for d in deltas),
        delta_tpf_zero=sum(d == 0 for d in deltas),
        delta_N=sum(r["delta_N"] for r in per),
        delta_F=sum(r["delta_F"] for r in per),
        delta_Q=sum(r["delta_Q"] for r in per),
        delta_physical_Q=sum(r["delta_physical_Q"] for r in per),
        delta_token_efficiency=(ranked.summary["token_efficiency"]
                                - reference.summary["token_efficiency"]),
        delta_physical_efficiency=(ranked.summary["physical_efficiency"]
                                   - reference.summary["physical_efficiency"]),
        delta_q_per_f=ranked.summary["q_per_f"] - reference.summary["q_per_f"])
    return summary, per


def load_context(path, model, qs):
    context = {q: {} for q in qs}
    if path is None or not path.is_file():
        return context, False
    rows = read_csv(path)
    for q in qs:
        for mode in ("ibs", "obs"):
            policy = f"{mode}_{q}"
            matches = [r for r in rows if r.get("model") == model
                       and r.get("role") == "baseline" and r.get("dataset") == "all"
                       and r.get("policy") == policy]
            if len(matches) > 1:
                raise ValueError(f"{path}: duplicate baseline row for {policy}")
            if not matches:
                continue
            row = matches[0]
            try:
                item = dict(requests=int(row["requests"]), correct=int(row["correct"]),
                            failed=int(row["failed"]), truncated=int(row["truncated"]),
                            N=int(row["N"]), F=int(row["F"]), Q=int(row["Q"]),
                            physical_Q=int(row["physical_Q"]), tpf=float(row["tpf"]),
                            token_efficiency=float(row["efficiency"]),
                            physical_efficiency=float(row["physical_efficiency"]),
                            q_per_f=float(row["q_per_f"]))
            except (KeyError, TypeError, ValueError):
                continue
            if item["requests"] != 30 or item["F"] <= 0 or item["physical_Q"] <= 0:
                continue
            context[q][mode] = item
    return context, True


def validate_totals(report):
    request_keys = set()
    per_dataset = {}
    for q in report.qs:
        for arm in ARM_ORDER:
            a = report.arms[q][arm]
            if len(a.requests) != 30:
                raise ValueError(f"Q{q} {arm} has {len(a.requests)} requests, expected 30")
            for row in a.requests:
                key = (q, arm, row["index"])
                if key in request_keys:
                    raise ValueError(f"Duplicated request total: {key}")
                request_keys.add(key)
                per_dataset[q, arm, row["dataset"]] = per_dataset.get((q, arm, row["dataset"]), 0) + 1
            for dataset in DATASETS:
                if per_dataset.get((q, arm, dataset), 0) != 10:
                    raise ValueError(f"Q{q} {arm} has {per_dataset.get((q, arm, dataset), 0)} "
                                     f"{dataset} requests, expected 10")
    if len(request_keys) != 4 * 30 * len(report.qs):
        raise ValueError(f"Expected {4 * 30 * len(report.qs)} request totals, got {len(request_keys)}")
    summary_keys = [(r["nominal_q"], r["arm"]) for r in summary_rows(report)]
    if len(summary_keys) != len(set(summary_keys)) or len(summary_keys) != 4 * len(report.qs):
        raise ValueError("Duplicated or missing summary totals")
    return len(request_keys), len(summary_keys)


def build_checks(arms, context, context_available, baselines, qs):
    checks = [dict(check="frozen30_manifest", status="pass",
                   detail="30 requests, 10 per dataset")]
    for q in qs:
        for arm in ARM_ORDER:
            a = arms[q][arm]
            checks.append(dict(check=f"arm:Q{q}:{arm}", status="pass",
                               detail=f"{a.policy} source={a.source} "
                                      f"requests={a.summary['requests']}"))
            checks.append(dict(check=f"requests_summary:Q{q}:{arm}", status="pass",
                               detail=f"{a.policy}: counts, N/F/Q/physical Q, "
                                      f"TPF and efficiency agree"))
    checks.append(dict(check="placebo_streams", status="pass",
                       detail="placebo_a and placebo_b are separate 30-request "
                              "executions; never pooled"))
    checks.append(dict(check="context_pairing", status="not_paired",
                       detail="IBS/OBS use the original fixed-window gate and original "
                              "seeds; context only"))
    checks.append(dict(check="request_totals", status="pass",
                       detail=f"{4 * 30 * len(qs)} unique (Q, arm, index) rows; "
                              f"10 requests per dataset per arm; no all+dataset duplication"))
    checks.append(dict(check="raw_event_audit", status="not_run_here",
                       detail="full-width parity, gate/victim match, first divergence, "
                              "swap type and strict edit of the added row are audited "
                              "separately from raw events"))
    if not context_available:
        checks.append(dict(check="context_available", status="missing",
                           detail=f"{display_path(baselines)} not found"))
    else:
        for q in qs:
            for mode in ("ibs", "obs"):
                if mode not in context[q]:
                    checks.append(dict(check=f"context:Q{q}:{mode}", status="missing",
                                       detail=f"no valid {mode}_{q} row"))
    return checks


def build_report(args):
    manifest = args.manifest.resolve()
    expected = load_manifest(manifest)
    qs = sorted(set(args.qs or [4, 8]))
    if not qs or any(q < 2 for q in qs):
        raise ValueError("Q values must be >= 2")
    measured, ranked = args.measured.resolve(), args.ranked.resolve()
    arms = {}
    for q in qs:
        arms[q] = dict(
            zero=resolve_zero(measured, ranked, args.model, q, expected),
            ranked=resolve_ranked(measured, ranked, args.model, q, expected),
            placebo_a=resolve_placebo(measured, args.model, q, "placebo_a", expected),
            placebo_b=resolve_placebo(measured, args.model, q, "placebo_b", expected))
    paired = {q: {} for q in qs}
    paired_requests = []
    for q in qs:
        for reference in REFERENCES:
            summary, per = pair(arms[q]["ranked"], arms[q][reference], reference, q)
            paired[q][reference] = summary
            paired_requests.extend(per)
    context, context_available = load_context(args.baselines, args.model, qs)
    checks = build_checks(arms, context, context_available, args.baselines, qs)
    report = Report(model=args.model, qs=qs, arms=arms, paired=paired,
                    paired_requests=paired_requests, context=context, checks=checks,
                    manifest=manifest, measured=measured, ranked=ranked,
                    baselines=args.baselines)
    validate_totals(report)
    return report


def request_rows(report):
    rows = []
    for q in report.qs:
        for arm in ARM_ORDER:
            a = report.arms[q][arm]
            for r in a.requests:
                rows.append(dict(
                    model=r["model"], nominal_q=q, arm=arm, policy=a.policy,
                    source=a.source, fresh=r["fresh"], mask=r["mask"], edit=r["edit"],
                    index=r["index"], dataset=r["dataset"], task_id=r["task_id"],
                    max_q=r["max_q"], N=r["N"], F=r["F"], Q=r["Q"],
                    physical_Q=r["physical_Q"], request_tpf=request_tpf(r),
                    request_token_efficiency=r["N"] / r["Q"],
                    request_physical_efficiency=r["N"] / r["physical_Q"],
                    correct=r["correct"], failed=r["failed"], truncated=r["truncated"]))
    return rows


def summary_rows(report):
    rows = []
    for q in report.qs:
        for arm in ARM_ORDER:
            a = report.arms[q][arm]
            s = a.summary
            rows.append(dict(
                model=report.model, nominal_q=q, arm=arm, policy=a.policy,
                source=a.source, dataset="all", requests=s["requests"], correct=s["correct"],
                failed=s["failed"], truncated=s["truncated"], N=s["N"], F=s["F"],
                Q=s["Q"], physical_Q=s["physical_Q"], macro_tpf=s["tpf"],
                token_efficiency=s["token_efficiency"],
                physical_efficiency=s["physical_efficiency"],
                q_per_f=s["q_per_f"], physical_q_per_f=s["physical_q_per_f"],
                max_q=s["max_q"], summary_checked=1))
    return rows


def paired_rows(report):
    rows = []
    for q in report.qs:
        for reference in REFERENCES:
            rows.append(report.paired[q][reference])
    return rows


def context_rows(report):
    rows = []
    for q in report.qs:
        for mode in ("ibs", "obs"):
            if mode not in report.context[q]:
                continue
            c = report.context[q][mode]
            rows.append(dict(
                model=report.model, nominal_q=q, policy=f"{mode}_{q}",
                source="baselines", role="context", dataset="all", requests=c["requests"],
                correct=c["correct"], failed=c["failed"], truncated=c["truncated"],
                N=c["N"], F=c["F"], Q=c["Q"], physical_Q=c["physical_Q"],
                macro_tpf=c["tpf"], token_efficiency=c["token_efficiency"],
                physical_efficiency=c["physical_efficiency"], q_per_f=c["q_per_f"],
                physical_q_per_f=c["Q"] / c["F"], max_q=""))
    return rows


def plot_study(path, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, y, bands = [], 0, []
    for q in report.qs:
        start = y
        for arm in ARM_ORDER:
            rows.append((y, q, FIG_LABEL[arm], arm, report.arms[q][arm].summary))
            y += 1
        for mode, label in (("ibs", "IBS (orig)"), ("obs", "OBS (orig)")):
            if mode in report.context[q]:
                rows.append((y, q, label, "context", report.context[q][mode]))
                y += 1
        bands.append((q, start, y - 1))
        y += 0.7
    height = 2.4 + 0.42 * y
    fig, axes = plt.subplots(1, 5, figsize=(15.8, height),
                             gridspec_kw={"width_ratios": [1.55, 1, 1, 1, 1]})
    fig.subplots_adjust(left=.13, right=.985, top=1 - 1.15 / height,
                        bottom=1.1 / height, wspace=.16)
    ink, muted = "#172b38", "#50616e"
    fig.text(.03, 1 - .22 / height,
             f"{MODEL_LABEL.get(report.model, report.model)}  |  "
             "Matched-victim candidate randomization (B32)",
             fontsize=17, weight="bold", color=ink)
    fig.text(.03, 1 - .50 / height,
             "Frozen 30  |  ranked = hard_exchange_edit4; placebo = uniform old "
             "candidate, same 4x gate/victim", fontsize=10, color=muted)
    fig.text(.03, .55 / height,
             "Both placebo streams are separate 30-request executions; never pooled. "
             "IBS/OBS use the original fixed-window gate and original seeds: context "
             "only, not paired.", fontsize=9, color=muted)
    fig.text(.03, .40 / height,
             "TPF = mean request N/F; physical efficiency = sum(N)/sum(physical Q); "
             "failed/truncated retained.", fontsize=9, color=muted)
    styles = {
        "zero": ("o", "#5b6b78", 55, "white"),
        "ranked": ("o", "#087b74", 62, "white"),
        "placebo_a": ("s", "#246da2", 52, "white"),
        "placebo_b": ("^", "#246da2", 58, "white"),
        "context": ("D", "none", 48, "#8a99a4"),
    }
    names = axes[0]
    names.set_xlim(0, 1)
    names.set_ylim(y, -0.6)
    names.set_xticks([])
    names.set_yticks([r[0] for r in rows])
    names.set_yticklabels([f"Q{r[1]} · {r[2]}" for r in rows],
                          fontsize=9, color=ink)
    names.tick_params(length=0, pad=6)
    for spine in names.spines.values():
        spine.set_visible(False)
    names.set_title("Budget / arm", loc="left", color=ink, weight="bold")
    for _, start, end in bands:
        for ax in axes:
            ax.axhspan(start - .45, end + .45, color="#edf3f6", lw=0, zorder=0)
    metrics = [
        ("Macro TPF", lambda s: s["tpf"], lambda x: f"{x:.3f}"),
        ("Correct / 30", lambda s: s["correct"], lambda x: f"{x:.0f}"),
        ("Physical Q", lambda s: s["physical_Q"], lambda x: f"{x:,.0f}"),
        ("Physical efficiency", lambda s: s["physical_efficiency"],
         lambda x: f"{100 * x:.2f}%"),
    ]
    for ax, (title, getter, formatter) in zip(axes[1:], metrics):
        values = [getter(r[4]) for r in rows]
        ax.set_ylim(y, -0.6)
        ax.set_yticks([])
        ax.tick_params(length=0, pad=4, labelsize=9, colors=muted)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="x", color="#dde6ec", lw=.8)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", color=ink, weight="bold")
        for row, value in zip(rows, values):
            yy, _, _, kind, _ = row
            marker, color, size, edge = styles[kind]
            face = "none" if kind == "context" else color
            ax.scatter([value], [yy], s=size, marker=marker, facecolor=face,
                       edgecolor=edge if kind == "context" else color,
                       linewidth=.9, zorder=3)
            ax.annotate(formatter(value), (value, yy), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=8.5,
                        color="#6b7b86" if kind == "context" else color, zorder=4)
        high = max(values)
        if title == "Correct / 30":
            ax.set_xlim(-.6, 34.0)
        else:
            ax.set_xlim(0, max(high * 1.28, .01))
    path.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"svg.fonttype": "none", "svg.hashsalt": "exchange-placebo"}):
        fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    if path.suffix == ".svg":
        lines = [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()]
        while lines and not lines[-1]:
            lines.pop()
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_delta(path, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, y, bands = [], 0, []
    for q in report.qs:
        start = y
        for reference in REFERENCES:
            rows.append((y, q, FIG_LABEL[reference], report.paired[q][reference]))
            y += 1
        bands.append((q, start, y - 1))
        y += 0.7
    height = 2.2 + 0.55 * y
    fig, axes = plt.subplots(1, 5, figsize=(15.8, height),
                             gridspec_kw={"width_ratios": [1.45, 1, 1, 1, 1]})
    fig.subplots_adjust(left=.12, right=.985, top=1 - 1.05 / height,
                        bottom=.40 / height, wspace=.18)
    ink, muted = "#172b38", "#50616e"
    fig.text(.03, 1 - .20 / height,
             f"{MODEL_LABEL.get(report.model, report.model)}  |  "
             "Ranked edit4 − reference (paired)",
             fontsize=17, weight="bold", color=ink)
    fig.text(.03, 1 - .47 / height,
             "Same 30 frozen tasks. Positive = ranked higher. Both placebo streams are "
             "separate 30-request executions; do not pool them.",
             fontsize=10, color=muted)
    names = axes[0]
    names.set_xlim(0, 1)
    names.set_ylim(y, -0.6)
    names.set_xticks([])
    names.set_yticks([r[0] for r in rows])
    names.set_yticklabels([f"Q{r[1]} · {r[2]}" for r in rows], fontsize=9, color=ink)
    names.tick_params(length=0, pad=6)
    for spine in names.spines.values():
        spine.set_visible(False)
    names.set_title("Budget / reference", loc="left", color=ink, weight="bold")
    for _, start, end in bands:
        for ax in axes:
            ax.axhspan(start - .45, end + .45, color="#edf3f6", lw=0, zorder=0)
    metrics = [
        ("Δ macro TPF", lambda p: p["delta_tpf"], lambda x: f"{x:+.3f}"),
        ("Δ correct / 30", lambda p: p["delta_correct"], lambda x: f"{int(x):+d}"),
        ("Δ physical Q", lambda p: p["delta_physical_Q"], lambda x: f"{int(x):+,}"),
        ("Δ physical efficiency", lambda p: p["delta_physical_efficiency"],
         lambda x: f"{100 * x:+.2f} pp"),
    ]
    for ax, (title, getter, formatter) in zip(axes[1:], metrics):
        values = [getter(r[3]) for r in rows]
        ax.set_ylim(y, -0.6)
        ax.set_yticks([])
        ax.tick_params(length=0, pad=4, labelsize=9, colors=muted)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.axvline(0, color="#8a99a4", lw=1, zorder=1)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", color=ink, weight="bold")
        for row, value in zip(rows, values):
            yy = row[0]
            color = "#087b74" if value > 0 else "#aa465b" if value < 0 else "#8a99a4"
            ax.barh(yy, value, height=.55, color=color, alpha=.85, zorder=3)
            ax.annotate(formatter(value), (value, yy),
                        xytext=(6 if value >= 0 else -6, 0), textcoords="offset points",
                        va="center", ha="left" if value >= 0 else "right",
                        fontsize=8.5, color=color, zorder=4)
        limit = max(abs(min(values)), abs(max(values)), 1e-9)
        ax.set_xlim(-limit * 2.0, limit * 2.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"svg.fonttype": "none", "svg.hashsalt": "exchange-placebo-delta"}):
        fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    if path.suffix == ".svg":
        lines = [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()]
        while lines and not lines[-1]:
            lines.pop()
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_markdown(report, out):
    def num(value):
        return f"{value:,.0f}"

    def pct(value):
        return f"{100 * value:.2f}%"

    def signed(value, digits=3):
        return f"{value:+.{digits}f}"

    lines = [
        f"# {MODEL_LABEL.get(report.model, report.model)}：交换是否需要预测出的候选？",
        "",
        "本轮在 Q4/1（主预算）和 Q8/1（预定负对照）上比较历史 4x 交换 "
        "`hard_exchange_edit4`（ranked）与两个新的候选随机化 placebo。两个 placebo "
        "保持 4x gate/victim、换位数和必刷 fresh 不变，只把换入候选改为从合格旧位中"
        "均匀随机抽取；它们是两次独立的 30 题执行，不能合并成 60 题。零预留为同预算"
        "对照，来源逐臂列在下表。所有臂使用同一 frozen30（GSM8K、MATH-500、HumanEval "
        "各 10 题）。",
        "",
        "## 状态",
        "",
        f"完整：{len(report.qs)} 个预算 × 4 个臂 × 30 题 = "
        f"{4 * 30 * len(report.qs)} 条请求明细；每个臂的 30 个身份与 frozen30 一致，"
        "requests 与 summary 表交叉核对通过。以下原始事件检查不在本报告计算，由"
        "独立原始事件审计负责：全宽 parity、gate/victim 匹配、首次物理 Q 分叉、"
        "换位类型、新增行的严格改写。",
        "",
        "## 实测臂",
        "",
        "| Q | 臂 | policy | 来源 | 答对/30 | 失败 | 截断 | macro TPF | 物理 Q | "
        "物理 efficiency | token efficiency |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for q in report.qs:
        for arm in ARM_ORDER:
            a = report.arms[q][arm]
            s = a.summary
            lines.append(
                f"| {q} | {ARM_LABEL[arm]} | `{a.policy}` | {a.source} | "
                f"{s['correct']} | {s['failed']} | {s['truncated']} | "
                f"{s['tpf']:.3f} | {num(s['physical_Q'])} | "
                f"{pct(s['physical_efficiency'])} | {pct(s['token_efficiency'])} |")
    lines += [
        "",
        "macro TPF 是逐请求 `N/F` 的均值；物理 efficiency 是 `ΣN/Σ物理 Q`；"
        "token efficiency 是 `ΣN/ΣQ`。失败与截断保留在分母中。",
        "",
        "## 配对：ranked − 每个参照",
        "",
        "| Q | 参照 | 只 ranked 对 | 只参照对 | 都对 | 都错 | Δ答对 | ΔTPF | "
        "Δ物理 Q | Δ物理 efficiency |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for q in report.qs:
        for reference in REFERENCES:
            p = report.paired[q][reference]
            lines.append(
                f"| {q} | {ARM_LABEL[reference]} | {p['only_ranked_correct']} | "
                f"{p['only_reference_correct']} | {p['both_correct']} | "
                f"{p['neither_correct']} | {p['delta_correct']:+d} | "
                f"{signed(p['delta_tpf'])} | {p['delta_physical_Q']:+,} | "
                f"{100 * p['delta_physical_efficiency']:+.2f} pp |")
    lines += [
        "",
        "ΔTPF 是逐请求 `N/F` 差值的均值；Δ物理 Q 是 30 题 `physical Q` 总和的差。"
        "两个 placebo 是独立执行，分别列出，不做合并。",
        "",
        "## 上下文：原固定窗口 IBS/OBS",
        "",
    ]
    if any(report.context[q] for q in report.qs):
        lines += [
            "| Q | policy | 答对/30 | macro TPF | 物理 Q | 物理 efficiency |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for q in report.qs:
            for mode in ("ibs", "obs"):
                if mode not in report.context[q]:
                    continue
                c = report.context[q][mode]
                lines.append(
                    f"| {q} | `{mode}_{q}` | {c['correct']} | {c['tpf']:.3f} | "
                    f"{num(c['physical_Q'])} | {pct(c['physical_efficiency'])} |")
        lines += [
            "",
            "IBS/OBS 使用原固定窗口 gate 和原 seed，不是本轮匹配的 4x gate/victim；"
            "它们只用于定位差距，不进入任何配对比较。配对只比较同一 frozen30 上、"
            "同一预算的交换臂。",
        ]
    else:
        lines += [
            "未找到可用的 IBS/OBS 基线行；本报告不含该上下文。",
            "",
            "IBS/OBS 使用原固定窗口 gate 和原 seed，不是本轮匹配的 4x gate/victim；"
            "它们只用于定位差距，不进入任何配对比较。",
        ]
    lines += [
        "",
        "## 范围与限制",
        "",
        "- 两个 placebo 是两次独立的 30 题执行，不能当作 60 个独立请求。",
        "- 本报告不重算评分，不读原始生成事件；失败与截断保留在分母中。",
        "- 全宽 parity、gate/victim 匹配、首次分叉、换位类型和新增行严格改写由"
        "原始事件审计负责，本报告不声称这些检查已经完成。",
        "",
        "## 文件",
        "",
        "- `exchange-placebo-requests.csv`：逐请求明细（4 臂 × 2 预算 × 30 题）。",
        "- `exchange-placebo-summary.csv`：逐臂汇总。",
        "- `exchange-placebo-paired.csv`：ranked 对每个参照的配对汇总。",
        "- `exchange-placebo-paired-requests.csv`：逐请求配对差值。",
        "- `exchange-placebo-context.csv`：原固定窗口 IBS/OBS 上下文（不配对）。",
        "- `exchange-placebo-checks.csv`：数据完整性与分离检查。",
        "- `exchange-placebo-study.svg`：四臂与 IBS/OBS 的数值图。",
        "- `exchange-placebo-delta.svg`：ranked − 参照的配对差值图。",
        "- `exchange-placebo-report.md`：本文件。",
        "",
        "## 复现",
        "",
        "```bash",
        "python -m tools.exchange_placebo_report --measured <measured-dir> "
        "--ranked <ranked-dir> --out <out-dir>"
        + (f" --model {report.model}" if report.model != "llada21" else "")
        + ("".join(f" --q {q}" for q in report.qs) if report.qs != [4, 8] else ""),
        "```",
        "",
    ]
    return "\n".join(lines)


def write_table(path, rows, fieldnames=()):
    """Reuse t2t_results.table, but keep a header when a context table is empty."""
    if rows:
        table(path, rows)
        return
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as stream:
        csv.DictWriter(stream, list(fieldnames), lineterminator="\n").writeheader()
    tmp.replace(path)


def write_report(report, out):
    out.mkdir(parents=True, exist_ok=True)
    plot_study(out / "exchange-placebo-study.svg", report)
    plot_delta(out / "exchange-placebo-delta.svg", report)
    table(out / "exchange-placebo-requests.csv", request_rows(report))
    table(out / "exchange-placebo-summary.csv", summary_rows(report))
    table(out / "exchange-placebo-paired.csv", paired_rows(report))
    table(out / "exchange-placebo-paired-requests.csv", report.paired_requests)
    write_table(out / "exchange-placebo-context.csv", context_rows(report), (
        "model", "nominal_q", "policy", "source", "role", "dataset", "requests", "correct",
        "failed", "truncated", "N", "F", "Q", "physical_Q", "macro_tpf",
        "token_efficiency", "physical_efficiency", "q_per_f", "physical_q_per_f",
        "max_q"))
    table(out / "exchange-placebo-checks.csv", report.checks)
    (out / "exchange-placebo-report.md").write_text(
        render_markdown(report, out), encoding="utf-8")
    run = dict(
        model=report.model, qs=report.qs,
        measured=dict(role="measured", name=report.measured.name,
                      path=display_path(report.measured)),
        ranked=dict(role="ranked", name=report.ranked.name,
                    path=display_path(report.ranked)),
        baselines=(dict(role="baselines", name=report.baselines.name,
                        path=display_path(report.baselines))
                   if report.baselines else None),
        manifest=dict(role="manifest", name=report.manifest.name,
                      path=display_path(report.manifest)),
        zero_sources={str(q): report.arms[q]["zero"].source for q in report.qs},
        arms={f"Q{q}": {arm: dict(policy=report.arms[q][arm].policy,
                                   source=report.arms[q][arm].source)
                        for arm in ARM_ORDER} for q in report.qs},
        complete=True,
        scope="published request/summary tables only; raw-event audit is separate",
        files=["exchange-placebo-requests.csv", "exchange-placebo-summary.csv",
               "exchange-placebo-paired.csv", "exchange-placebo-paired-requests.csv",
               "exchange-placebo-context.csv", "exchange-placebo-checks.csv",
               "exchange-placebo-study.svg", "exchange-placebo-delta.svg",
               "exchange-placebo-report.md", "exchange-placebo-run.json"])
    (out / "exchange-placebo-run.json").write_text(
        json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured", type=Path, required=True,
                        help="tools.t2t_results output directory for this round")
    parser.add_argument("--ranked", type=Path, default=DEFAULT_RANKED,
                        help="published exchange directory (default: experiments/t2t/exchange)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="llada21")
    parser.add_argument("--q", dest="qs", type=int, action="append",
                        help="nominal Q; repeatable (default: 4 and 8)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = build_report(args)
        write_report(report, args.out.resolve())
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()

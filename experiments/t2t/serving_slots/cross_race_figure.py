"""Archive the registered Gemma fixed cross race. Gather on the data host; render locally."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import re
import sys

from experiments.t2t.serving_slots import cross_race_report as report


HERE = Path(__file__).resolve().parent
CONTROLS = HERE / "controls"
OUT = CONTROLS / "cross-race-gemma"
PAIRS = [(a.name, b.name) for a, b in report.COMPARISONS if a.kind == "cross"]
COHORTS = report.COHORTS
ARMS = tuple(dict.fromkeys(a for pair in PAIRS for a in pair))
CI = ("accuracy_delta", "macro_TPF_delta", "physical_Q_per_request_delta",
      "N_per_history_Q_delta", "tail_Q_per_request_delta", "tail_match_rate_delta")
SNAPSHOT = "gemma-fixed-cross-compact-v1"
# Fixed runtime snapshot; local report.py can register other experiments.
REMOTE_REPORT_SHA = "3c3830bd08dfdfabb85d74873737da05219b17e3d762bb94a84a9dd6a42dd499"
CONTROL_FILES = (
    "block-history90-matrix.csv", "block-history90-audited.csv",
    "block-history90-gemma-ibs512-h800.csv", "block-history90-gemma-w3-w4-b3.csv",
)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def gather(receipts, logs, samples, draws):
    """Only cohort aggregates and coverage indices leave this process."""
    manifest = json.loads(report.MANIFEST.read_text())
    if digest(samples) != manifest["samples_sha256"]:
        raise ValueError("samples digest disagrees with frozen90 manifest")
    started, done = set(), set()
    tracebacks = 0
    log_paths = sorted(logs.glob("g[0-9]*.log"))
    for path in log_paths:
        for line in path.read_text(errors="replace").splitlines():
            m = re.match(r"^(START|DONE) (C\S+)-(\d+)(?:\s|$)", line)
            if m and m[2] in ARMS:
                (started if m[1] == "START" else done).add((m[2], int(m[3])))
            tracebacks += line.startswith("Traceback (most recent call last):")

    trials, runtime = [], Counter()
    for path in sorted(receipts.glob("*.json")):
        try:
            row = json.loads(path.read_text())
            t = report.normalize(row)
            if t.arm.name not in ARMS or path.stem != f"{t.arm.name}-{t.index:02d}":
                raise ValueError("unexpected arm or filename/receipt mismatch")
            trials.append(t)
            runtime[tuple(row[k] for k in ("host", "gpu", "torch"))] += 1
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"invalid receipt {path.name}: {exc}") from exc
    if len(runtime) > 1:
        raise ValueError("mixed host/GPU type/torch in race receipts")
    result = report.summarize(trials, manifest=manifest, kinds=["cross"],
                              draws=0)
    rows = defaultdict(dict)
    for t in trials:
        rows[t.arm.name][t.index] = t
    coverage = {}
    for cohort, indices in COHORTS.items():
        audit = {}
        for arm in ARMS:
            c = result["arms"][arm]["cohorts"][cohort]
            missing = c["missing_indices"]
            audit[arm] = dict(expected=c["expected"], received=c["received"], scored=c["scored"],
                              missing=len(missing), score_pending=len(c["score_pending_indices"]),
                              missing_indices=missing, pending_indices=c["score_pending_indices"],
                              started_no_receipt=[i for i in missing if (arm, i) in started],
                              done_no_receipt=[i for i in missing if (arm, i) in done])
        coverage[cohort] = audit

    # The shared reporter validates all receipts first. Reuse its estimators only
    # for entire cohorts, so even transient matched subsets cannot be archived.
    cohorts = {}
    for cohort, indices in COHORTS.items():
        if any(r["scored"] != len(indices) for r in coverage[cohort].values()):
            continue
        cohorts[cohort] = dict(
            arms={a: dict(correct=sum(rows[a][i].correct for i in indices),
                          **report._work_summary([rows[a][i] for i in indices])) for a in ARMS},
            pairs=[(a, b, report._paired(rows[a], rows[b], indices, draws, 0)) for a, b in PAIRS],
        )
    return dict(schema=SNAPSHOT, coverage=coverage, cohorts=cohorts,
                provenance=dict(observed_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                receipts=str(receipts), logs=str(logs), log_files=len(log_paths),
                                log_tracebacks=tracebacks, samples_sha256=manifest["samples_sha256"],
                                manifest_sha256=digest(report.MANIFEST),
                                report_sha256=digest(Path(report.__file__)),
                                worker_sha256=digest(Path(report.__file__).with_name("cross_race.py")),
                                draws=draws, seed=0, level=.95,
                                runtime=[dict(host=h, gpu=g, torch=t, receipts=n)
                                         for (h, g, t), n in runtime.items()],
                                scorer_sha256=sorted({t.scorer_sha for t in trials if t.scorer_sha}),
                                receipts_without_scorer_sha256=sum(not t.scorer_sha for t in trials)))


def read_csv(name):
    with (CONTROLS / name).open(newline="") as f:
        return list(csv.DictReader(f))


def controls(cohort, requests):
    aliases = {"original30": ("original-0-29", "original30"),
               "new60": ("added-30-89", "added60"), "frozen90": ("all-0-89", "all90")}
    audited_cohort, matrix_cohort = aliases[cohort]
    matrix_file, audit, h800, wide = CONTROL_FILES
    matrix = {(r["method"], r["setting"]): r for r in read_csv(matrix_file)
              if r["model"] == "gemma" and r["cohort"] == matrix_cohort}
    sources = {name: read_csv(name) for name in (audit, h800, wide)}
    for method, tier, name, source in (
        ("native", "256", "Native C256 / Flow W1", audit),
        ("ibs", "64", "IBS C64", audit), ("ibs", "128", "IBS C128", audit),
        ("ibs", "512", "IBS C512 (H800)", h800), ("ibs", "512", "IBS C512 (A800)", audit),
        ("flow", "2", "Flow W2", audit), ("flow", "3", "Flow W3", wide),
        ("flow", "4", "Flow W4", wide),
    ):
        if source == wide:
            part = "merged90" if cohort == "frozen90" else cohort
            row, = [r for r in sources[source] if r["policy"] == f"flow_w{tier}" and r["cohort"] == part]
            n, f, q, tpf, failed, trunc = (row["sum_output_tokens"], row["sum_generation_forwards"],
                row["sum_generation_physical_padded_rows"], row["mean_tpf"], row["failures"], "")
        else:
            row, = [r for r in sources[source] if r["model"] == "gemma" and r["cohort"] == audited_cohort
                    and r["method"] == method and r["setting"] == tier]
            n, f, q, tpf, failed, trunc = (row[k] for k in ("N", "F", "physical_Q", "macro_tpf",
                                                            "failed", "truncated"))
        n, f, q, correct, failed = map(int, (n, f, q, row["correct"], failed))
        if (int(row["requests"]) != requests or not 0 <= correct <= requests - failed <= requests
                or n < 0 or min(f, q) <= 0):
            raise ValueError(f"incomplete or inconsistent control: {name}/{cohort}")
        prefix = None
        if method != "flow" and (tier != "512" or source == h800):
            m = matrix[method, tier]
            if tuple(int(m[k]) for k in ("N", "F", "physical_Q", "correct")) != (n, f, q, correct):
                raise ValueError(f"control disagrees with matrix: {name}/{cohort}")
            prefix = int(m["prefix_kv_Q"])
            if not 0 < prefix <= q:
                raise ValueError(f"invalid prefix-KV denominator: {name}/{cohort}")
        yield dict(cohort=cohort, arm=name, role="external", requests=requests, correct=correct,
                   failed=failed, truncated=trunc, N=n, F=f, physical_Q=q, prefix_kv_Q=prefix,
                   macro_TPF=float(tpf), N_per_physical_Q=n/q, N_per_prefix_kv_Q=n/prefix if prefix else None,
                   source=f"{source}: {row.get('source', row.get('policy'))}")


def factor(arm):
    meta = report.ARMS[arm]
    _, gate, slack, den, pref = arm.split("-")
    return dict(budget=meta.budget, gate=gate[4:], slack=int(slack[-1]), late_den=int(den),
                prefix=pref[4:].replace("of", "/"))


def label(arm):
    if arm not in report.ARMS:
        return arm
    m = factor(arm)
    c = m["budget"]
    if c == 512:
        return f"C512  gate {m['gate']}"
    if m["slack"]:
        return f"C{c}  borrow at 1/{m['late_den']}"
    return f"C{c}  prefix {m['prefix']}" + (" (base)" if m["prefix"] == "1/8" else "")


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    if not fields:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: f"{v:.9g}" if isinstance(v, float) else v
                          for k, v in row.items()} for row in rows)


def figure(path, rows, cohorts):
    """Aligned dot plots: no wall-clock claim and no interpolation across arms."""
    cols = (("accuracy", "Correct (%)", 100, (60, 70, 80, 90, 100)),
            ("macro_TPF", "Macro tokens / forward", 1, (0, 8, 16, 24, 32)),
            ("N_per_physical_Q", "Tokens / physical Q (%)", 100, (0, 5, 10, 15, 20)))
    marks = {"original30": "#2b6690", "new60": "#bf691c", "frozen90": "#12795e"}
    first = [r for r in rows if r["cohort"] == cohorts[0]]
    y = 112
    positions = {}
    last_group = None
    width = 940 if len(cohorts) == 1 else 1110
    colw = (width - 258) / 3
    ink, muted, rule = "#192a36", "#4e6372", "#d5dfe6"
    body = []

    def text(x, y, value, size=13, fill=ink, extra=""):
        body.append(f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" {extra}>'
                    f'{escape(str(value))}</text>')

    def line(x, x2, y, y2):
        body.append(f'<path d="M{x} {y}L{x2} {y2}" stroke="{rule}"/>')

    text(24, 31, "DiffusionGemma | Fixed cross race", 22, extra='font-weight="bold"')
    text(24, 54, "Frozen request cohorts. Historical controls use separate executors.", 13, muted)
    for i, part in enumerate(cohorts):
        body.append(f'<circle cx="{30 + 144*i}" cy="72" r="3.3" fill="{marks[part]}"/>')
        text(40 + 144*i, 76, part, 12, marks[part])
    for r in first:
        a = r["arm"]
        group = r.get("budget") if r["role"] == "race" else "external"
        if group != last_group:
            y += 8
            line(24, width-24, y-12, y-12)
        if group == "external" and last_group != group:
            text(24, y, "HISTORICAL CONTROLS", 10, muted, 'font-weight="bold"')
            y += 16
        positions[a] = y
        text(24, y+4, label(a), fill=ink if group != "external" else muted)
        y += 18
        last_group = group
    for j, (key, title, scale, ticks) in enumerate(cols):
        left = 226 + j*colw
        lo = min(ticks[0], min(r[key]*scale for r in rows))
        hi = max(ticks[-1], max(r[key]*scale for r in rows))
        if key == "accuracy" and lo == ticks[0]:
            lo -= 1
        def x(v):
            return round(left + (v-lo)/(hi-lo)*(colw-27), 2)
        text(left, 102, title, 13, extra='font-weight="bold"')
        for tick in ticks:
            line(x(tick), x(tick), 108, y-7)
            text(x(tick), y+8, tick, 11, muted, 'text-anchor="middle"')
        for r in rows:
            part = r["cohort"]
            cy = positions[r["arm"]] + (cohorts.index(part) - (len(cohorts)-1)/2)*4
            cx = x(r[key]*scale)
            color = marks[part]
            fill = color if r["role"] == "race" else "#fff"
            hint = f"{part} {label(r['arm'])}: {r[key]*scale:.3f}"
            body.append(f'<circle cx="{cx}" cy="{cy}" r="3.3" fill="{fill}" stroke="{color}">'
                        f'<title>{escape(hint)}</title></circle>')
    text(24, y+32, "Filled: race   Open: external   |   Physical Q includes generation encoder; excludes prefill.",
         12, muted)
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{y+47}" '
                    f'viewBox="0 0 {width} {y+47}" role="img">\n<rect width="100%" height="100%" fill="white"/>'
                    '\n<g font-family="system-ui, sans-serif">' + "\n".join(body) + "\n</g></svg>\n",
                    encoding="utf-8")


def render(bundle):
    p = bundle["provenance"]
    if (bundle["schema"] != SNAPSHOT or p["manifest_sha256"] != digest(report.MANIFEST)
            or p["report_sha256"] != REMOTE_REPORT_SHA):
        raise ValueError("unexpected gather schema, manifest or fixed runtime reporter")
    cov, complete = [], {}
    for cohort, indices in COHORTS.items():
        audit = bundle["coverage"][cohort]
        if set(audit) != set(ARMS):
            raise ValueError(f"unexpected coverage arm registry: {cohort}")
        for a in ARMS:
            row = audit[a]
            if not (0 <= row["scored"] <= row["received"] <= row["expected"] == len(indices)
                    and row["missing"] == row["expected"] - row["received"]
                    and row["score_pending"] == row["received"] - row["scored"]):
                raise ValueError(f"inconsistent coverage: {cohort}, {a}")
            cov.append(dict(cohort=cohort, arm=a,
                            status="complete" if row["scored"] == len(indices) else "incomplete",
                            **{k: " ".join(map(str, v)) if isinstance(v, list) else v for k, v in row.items()}))
        complete[cohort] = all(audit[a]["scored"] == len(indices) for a in ARMS)
    eligible = [c for c in COHORTS if complete[c]]
    if set(eligible) != set(bundle["cohorts"]):
        raise ValueError("coverage and result cohorts disagree")
    for c in ("original30", "frozen90"):
        if c not in eligible and (OUT / f"{c}.svg").exists():
            raise ValueError(f"refusing to replace an existing complete {c} archive with incomplete coverage")

    arms, pairs = [], []
    for c in eligible:
        n = len(COHORTS[c])
        data = bundle["cohorts"][c]
        if set(data["arms"]) != set(ARMS) or [(a, b) for a, b, _ in data["pairs"]] != PAIRS:
            raise ValueError(f"unexpected results arm/pair registry: {c}")
        for a in ARMS:
            w = dict(data["arms"][a])
            w.update(prefix_kv_Q=w.pop("history_Q"), N_per_prefix_kv_Q=w.pop("N_per_history_Q"))
            categories = w.pop("categories")
            arms.append(dict(cohort=c, arm=a, role="race", **factor(a), requests=n,
                             failed=0, source=p["receipts"], **w,
                             **{f"Q_{k.replace('-', '_')}": v for k, v in categories.items()}))
        arms.extend(controls(c, n))
        for a, b, q in data["pairs"]:
            if q["status"] != "complete" or q["paired_work"] != n or q["paired_scored"] != n:
                raise ValueError(f"incomplete pair: {c}, {a} vs {b}")
            m = factor(b)
            work, quality = q["work"]["delta"], q["quality"]
            pair = dict(cohort=c, factor="gate" if m["budget"] == 512 else "borrow" if m["slack"]
                        else "prefix", budget=m["budget"], reference=a, trial=b,
                        paired_work=n, paired_scored=n, **quality,
                        ties=n-quality["rescue"]-quality["loss"],
                        macro_TPF_delta=work["macro_TPF"], physical_Q_per_request_delta=work["physical_Q"]/n,
                        decoder_Q_delta=work["decoder_Q"], N_per_history_Q_delta=work["N_per_history_Q"],
                        tail_Q_per_request_delta=work["tail_Q"]/n, tail_match_rate_delta=work["tail_match_rate"])
            for k in CI:
                pair[f"{k}_lo95"], pair[f"{k}_hi95"] = q["confidence95"].get(k) or (None, None)
            pairs.append(pair)
    for row in arms:
        row["accuracy"] = row["correct"]/row["requests"]
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "coverage.csv", cov)
    write_csv(OUT / "arms.csv", arms)
    write_csv(OUT / "paired.csv", pairs)
    hashes = {name: digest(CONTROLS / name) for name in CONTROL_FILES}
    (OUT / "provenance.json").write_text(
        json.dumps(dict(p, complete_cohorts=eligible, control_sha256=hashes,
                        render_report_sha256=digest(Path(report.__file__))), indent=2) + "\n")
    if complete["original30"]:
        figure(OUT / "original30.svg", [r for r in arms if r["cohort"] == "original30"], ["original30"])
    if complete["frozen90"]:
        figure(OUT / "frozen90.svg", arms, list(COHORTS))
    for c, indices in COHORTS.items():
        counts = [r["scored"] for r in cov if r["cohort"] == c]
        print(f"{c}: {min(counts)}..{max(counts)}/{len(indices)} scored per arm; "
              f"{sum(i == len(indices) for i in counts)}/{len(ARMS)} arms complete", file=sys.stderr)
    return complete["frozen90"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    g = sub.add_parser("gather", help="emit aggregate JSON to stdout on the receipt host")
    g.add_argument("receipts", type=Path)
    g.add_argument("--logs", type=Path, required=True)
    g.add_argument("--samples", type=Path, required=True)
    g.add_argument("--draws", type=int, default=4000)
    r = sub.add_parser("render", help="read aggregate JSON from stdin and archive reports")
    r.add_argument("--require-frozen90", action="store_true", help="exit 1 unless the full matrix is complete")
    args = p.parse_args()
    try:
        if args.mode == "gather":
            if args.draws < 0 or not args.receipts.is_dir() or not args.logs.is_dir():
                raise ValueError("expected nonnegative draws and existing receipt/log directories")
            print(json.dumps(gather(args.receipts, args.logs, args.samples, args.draws), allow_nan=False))
            return 0
        complete = render(json.load(sys.stdin))
        return int(args.require_frozen90 and not complete)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

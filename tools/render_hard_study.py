"""Render frozen-30 hard or adaptive results with same-window controls to SVG."""
import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
MODELS = {"llada21": ("LLaDA 2.1", 32), "gemma": ("DiffusionGemma", 256)}
HARD = {"hard_noedit": "Hard no-edit", "hard_frontier": "Hard frontier",
        "hard_random": "Hard random"}
HARD_POLICY = re.compile(r"(?:^|_)(hard_noedit|hard_frontier|hard_random)$")
ADAPTIVE = {"hard_slack": ("Hard slack", "adaptive"),
            "hard_trigger": ("Hard trigger", "adaptive"),
            "hard_oracle1": ("Hard oracle1 (probes charged)", "oracle")}
ADAPTIVE_POLICY = re.compile(r"(?:^|_)(hard_slack|hard_trigger|hard_oracle1)$")
BASE_POLICY = re.compile(r"(ibs|obs)_(\d+)")


@dataclass(frozen=True)
class Point:
    tpf: float
    correct: int
    efficiency: float
    q_per_f: float


def all_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["dataset"] == "all":
                yield row


def point(row, efficiency_column):
    tpf, correct, efficiency = float(row["tpf"]), int(row["correct"]), float(row[efficiency_column])
    q_per_f = float(row["q_per_f"])
    if (int(row["requests"]) != 30 or not 0 <= correct <= 30
            or not math.isfinite(tpf) or tpf < 0 or not 0 <= efficiency <= 1
            or not math.isfinite(q_per_f) or q_per_f <= 0):
        raise ValueError(f"invalid frozen-30 metrics: {row['model']} / {row['policy']}")
    return Point(tpf, correct, efficiency, q_per_f)


def put(table, key, value):
    if key in table and table[key] != value:
        raise ValueError(f"conflicting rows for {key!r}")
    table[key] = value


def read_baselines(path):
    baselines = {}
    for row in all_rows(path):
        match = BASE_POLICY.fullmatch(row["policy"])
        if row["model"] in MODELS and row["role"] == "baseline" and match:
            mode, window = match.groups()
            put(baselines, (row["model"], int(window), mode), point(row, "efficiency"))
    return baselines


def groups(path, model, width, baselines):
    online = defaultdict(dict)
    for row in all_rows(path):
        match = HARD_POLICY.search(row["policy"])
        if row["model"] != model or (not match and row["policy"] != "native"):
            continue
        q = int(row["nominal_q"])
        if q < 1 or (not match and q != width):
            raise ValueError(f"{path}: unexpected nominal Q={q} for {row['policy']}")
        put(online[q], row["policy"], point(row, "token_efficiency"))

    for q, measured in sorted(online.items()):
        if q >= width:
            # The runner executes one dense route at/beyond native block width.
            policy = min(measured, key=lambda p: (p != "native", not p.endswith("hard_noedit"), p))
            if len(set(measured.values())) > 1:
                print(f"{model} Q{q}: full-width rows differ; showing only {policy}", file=sys.stderr)
            label = "Native (full width)" if q == width else "Dense saturation control"
            rows = [(label, "dense", measured[policy])]
        else:
            rows = []
            for mode, label in HARD.items():
                matches = sorted((p, value) for p, value in measured.items() if p.endswith(mode))
                for p, value in matches:
                    name = label if len(matches) == 1 else p
                    rows.append((name, "hard", value))
        yield q, rows + window_controls(model, q, width, baselines)


def window_controls(model, q, width, baselines):
    return [(f"{mode.upper()} (C={q})", mode, baselines[model, q, mode])
            for mode in ("ibs", "obs") if q <= width and (model, q, mode) in baselines]


def adaptive_groups(path, hard, model, width, baselines):
    online = defaultdict(dict)
    for row in all_rows(path):
        match = ADAPTIVE_POLICY.search(row["policy"])
        if row["model"] != model or not match or int(row["requests"]) != 30:
            continue
        q = int(row["nominal_q"])
        if not 1 <= q < width:
            raise ValueError(f"{path}: unexpected adaptive nominal Q={q} for {row['policy']}")
        put(online[q], match[1], point(row, "token_efficiency"))

    noedit = {}
    if online and hard.is_file():
        for row in all_rows(hard):
            match = HARD_POLICY.search(row["policy"])
            if (row["model"] == model and match and match[1] == "hard_noedit"
                    and int(row["requests"]) == 30 and int(row["nominal_q"]) in online):
                put(noedit, int(row["nominal_q"]), point(row, "token_efficiency"))

    for q, measured in sorted(online.items()):
        rows = [("Hard no-edit (v1)", "hard", noedit[q])] if q in noedit else []
        rows += [(name, kind, measured[mode]) for mode, (name, kind) in ADAPTIVE.items()
                 if mode in measured]
        yield q, rows + window_controls(model, q, width, baselines)


def placement_groups(path, hard, adaptive, model, width, baselines):
    points = defaultdict(dict)
    for p, suffixes in [(path, ("hard_slack_placebo", "hard_trigger_placebo")),
                        (adaptive, ("hard_slack", "hard_trigger")),
                        (hard, ("hard_noedit",))]:
        for row in all_rows(p):
            mode = next((s for s in suffixes if row["policy"].endswith("_" + s)), None)
            if mode and row["model"] == model:
                put(points[int(row["nominal_q"])], mode, point(row, "token_efficiency"))
    for q, measured in sorted(points.items()):
        if not any(m.endswith("_placebo") for m in measured):
            continue
        rows = [(label, kind, measured[m]) for m, label, kind in (
            ("hard_noedit", "No edit reserve", "hard"),
            ("hard_slack", "Slack: frontier", "adaptive"),
            ("hard_slack_placebo", "Slack: random", "placebo"),
            ("hard_trigger", "Trigger: frontier", "adaptive"),
            ("hard_trigger_placebo", "Trigger: random", "placebo")) if m in measured]
        yield q, rows + window_controls(model, q, width, baselines)


def render(path, model, width, data, *, adaptive=False, placement=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    # A dedicated row for each measured configuration keeps even ties legible.
    bands, points, y = [], [], 0
    for q, rows in data:
        label = f"Q{q}"
        if q > width:
            label += f"  |  B{width} single-block saturation"
        elif q == width:
            label += "  |  Full width"
        bands.append((y, label))
        for name, kind, value in rows:
            y += 1
            points.append((y, name, kind, value))
        y += 1.5

    height = 1.7 + .29 * y
    ratios = [1.28, 1, 1, 1, 1] if adaptive else [1.28, 1, 1, 1]
    fig, axes = plt.subplots(1, len(ratios), figsize=(16.5 if adaptive else 13.4, height),
                             gridspec_kw={"width_ratios": ratios})
    fig.subplots_adjust(left=.035, right=.98, top=1-1.12/height,
                        bottom=.40/height, wspace=.12)
    title = "Adaptive selection (v2)" if adaptive else "Hard selection"
    if placement:
        title = "Matched old-position controls"
    fig.text(.035, 1-.24/height, f"{MODELS[model][0]}  |  {title} (B{width})",
             fontsize=17, weight="bold", color="#172b38")
    fig.text(.035, 1-.53/height,
             "Frozen 30   |   IBS/OBS use the original window C = nominal Q, not matched physical Q.",
             fontsize=10, color="#50616e")
    notes = "Only measured rows are shown. TPF = mean request N/F; token efficiency = sum(N) / sum(Q)."
    if adaptive:
        notes += " Actual Q/F = sum(Q) / sum(F)."
        if any(kind == "oracle" for _, _, kind, _ in points):
            notes += " Oracle1 includes probe rows and forwards."
    if any(q > width for q, _ in data):
        notes += f" Q > {width} has no IBS/OBS comparison."
    fig.text(.035, .12/height, notes, fontsize=9, color="#50616e")

    ink = "#172b38"
    styles = {"hard": ("o", "#087b74"), "dense": ("D", "#42546c"),
              "ibs": ("s", "#b05929"), "obs": ("^", "#794c96"),
              "adaptive": ("o", "#246da2"), "oracle": ("X", "#aa465b"),
              "placebo": ("P", "#596165"), "geometry": ("D", "#216c55")}
    for ax in axes:
        ax.set_ylim(y-.5, -.55)
        ax.set_yticks([])
        ax.tick_params(length=0, pad=5, labelcolor="#50616e", labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for line, _ in bands:
            ax.axhspan(line-.43, line+.43, color="#ecf1f4", lw=0, zorder=0)
    names = axes[0]
    names.set_xlim(0, 1)
    names.set_xticks([])
    names.set_title("Nominal Q / configuration", loc="left", color=ink, weight="bold")
    for line, label in bands:
        names.text(.02, line, label, va="center", fontsize=10, weight="bold", color=ink)
    for line, name, kind, _ in points:
        names.text(.06, line, name, va="center", fontsize=10, color=styles[kind][1])

    metrics = [("tpf", "TPF", ".3f"), ("correct", "Quality", "d"),
               ("efficiency", "Token efficiency", ".2%")]
    if adaptive:
        metrics.append(("q_per_f", "Actual Q/F", ".2f"))
    for ax, (key, title, fmt) in zip(axes[1:], metrics, strict=True):
        peak = 30 if key == "correct" else max(getattr(p, key) for _, _, _, p in points) or 1
        ax.set_xlim(-.04 * peak, 1.26 * peak)
        ax.set_title(title, loc="left", color=ink, weight="bold")
        ax.xaxis.tick_top()
        if key == "correct":
            ax.set_xticks((0, 10, 20, 30))
        else:
            ax.xaxis.set_major_locator(MaxNLocator(5, prune="lower"))
        if key == "efficiency":
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.grid(axis="x", color="#dce4e8", linewidth=.65, zorder=0)
        for line, _, kind, p in points:
            x = getattr(p, key)
            marker, color = styles[kind]
            ax.plot(x, line, marker=marker, color=color, markersize=5, linestyle="none")
            label = format(x, fmt) + ("/30" if key == "correct" else "")
            ax.annotate(label, (x, line), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=9, color=ink)
    path.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"svg.fonttype": "none", "svg.hashsalt": "hard-study"}):
        fig.savefig(path, metadata={"Date": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root")
    parser.add_argument("--study", choices=("hard", "adaptive", "placement"), default="hard")
    parser.add_argument("--model", choices=MODELS, action="append", help="render only this model; repeatable")
    parser.add_argument("--out-dir", type=Path, help="default: experiments/t2t/<study>/figures under root")
    args = parser.parse_args()
    hard = args.root / "experiments/t2t/hard"
    study = hard.parent / args.study
    adaptive = args.study != "hard"
    out = args.out_dir or study / "figures"
    try:
        baselines = read_baselines(args.root / "experiments/baselines/summary.csv")
        for model in dict.fromkeys(args.model or MODELS):
            _, width = MODELS[model]
            src = study / f"{model}-online-summary.csv"
            data = []
            if src.is_file():
                source = (placement_groups(src, hard / src.name, hard.parent / "adaptive" / src.name,
                                           model, width, baselines) if args.study == "placement" else
                          adaptive_groups(src, hard / src.name, model, width, baselines)
                          if adaptive else groups(src, model, width, baselines))
                data = list(source)
            if not data:
                print(f"Skipping {model}: no eligible dataset=all rows in {src}", file=sys.stderr)
                continue
            path = out / f"{model}-{args.study}-study.svg"
            render(path, model, width, data, adaptive=adaptive, placement=args.study == "placement")
            print(path)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

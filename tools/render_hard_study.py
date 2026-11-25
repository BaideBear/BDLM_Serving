"""Render the frozen-30 hard study and same-window IBS/OBS baselines to SVG."""
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
BASE_POLICY = re.compile(r"(ibs|obs)_(\d+)")


@dataclass(frozen=True)
class Point:
    tpf: float
    correct: int
    efficiency: float


def all_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["dataset"] == "all":
                yield row


def point(row, efficiency_column):
    tpf, correct, efficiency = float(row["tpf"]), int(row["correct"]), float(row[efficiency_column])
    if (int(row["requests"]) != 30 or not 0 <= correct <= 30
            or not math.isfinite(tpf) or tpf < 0 or not 0 <= efficiency <= 1):
        raise ValueError(f"invalid frozen-30 metrics: {row['model']} / {row['policy']}")
    return Point(tpf, correct, efficiency)


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
        if q <= width:
            for mode in ("ibs", "obs"):
                key = (model, q, mode)
                if key in baselines:
                    rows.append((f"{mode.upper()} (C={q})", mode, baselines[key]))
        yield q, rows


def render(path, model, width, data):
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
    fig, axes = plt.subplots(1, 4, figsize=(13.4, height),
                             gridspec_kw={"width_ratios": [1.28, 1, 1, 1]})
    fig.subplots_adjust(left=.035, right=.98, top=1-1.12/height,
                        bottom=.40/height, wspace=.12)
    fig.text(.035, 1-.24/height, f"{MODELS[model][0]}  |  Hard selection (B{width})",
             fontsize=17, weight="bold", color="#172b38")
    fig.text(.035, 1-.53/height,
             "Frozen 30   |   IBS/OBS use the original window C = nominal Q, not matched physical Q.",
             fontsize=10, color="#50616e")
    notes = "Only measured rows are shown. TPF = mean request N/F; token efficiency = sum(N) / sum(Q)."
    if any(q > width for q, _ in data):
        notes += f" Q > {width} has no IBS/OBS comparison."
    fig.text(.035, .12/height, notes, fontsize=9, color="#50616e")

    ink = "#172b38"
    styles = {"hard": ("o", "#087b74"), "dense": ("D", "#42546c"),
              "ibs": ("s", "#b05929"), "obs": ("^", "#794c96")}
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

    for ax, key, title, fmt in zip(
        axes[1:], ("tpf", "correct", "efficiency"),
        ("TPF", "Quality", "Token efficiency"), (".3f", "d", ".2%"),
    ):
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
    parser.add_argument("--model", choices=MODELS, action="append", help="render only this model; repeatable")
    parser.add_argument("--out-dir", type=Path, help="default: experiments/t2t/hard/figures under root")
    args = parser.parse_args()
    hard = args.root / "experiments/t2t/hard"
    out = args.out_dir or hard / "figures"
    try:
        baselines = read_baselines(args.root / "experiments/baselines/summary.csv")
        for model in dict.fromkeys(args.model or MODELS):
            _, width = MODELS[model]
            src = hard / f"{model}-online-summary.csv"
            data = list(groups(src, model, width, baselines)) if src.is_file() else []
            if not data:
                print(f"Skipping {model}: no eligible dataset=all rows in {src}", file=sys.stderr)
                continue
            path = out / f"{model}-hard-study.svg"
            render(path, model, width, data)
            print(path)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

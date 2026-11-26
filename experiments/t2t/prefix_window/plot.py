"""Plot finished prefix-window cohort summaries (Python 3.10+, Matplotlib).

From the repository root: python -B experiments/t2t/prefix_window/plot.py
Use --model gemma --q 32 to rebuild just the Gemma Q32 figure.
"""

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


HERE = Path(__file__).resolve().parent
MODELS = {"gemma": ("DiffusionGemma", 256), "llada21": ("LLaDA 2.1-mini", 32)}
COHORTS = {
    30: ("original-0-29", "A   Original 30"),
    60: ("added-30-89", "B   Added 60"),
    90: ("all-0-89", "C   Full 90"),
}
POLICY = re.compile(r"t2t_q(\d+)_f0_m(\d+)_hard_prefix_(\w+)")
MODES = {
    "zero": ("Zero add-back", "#172838", "s"),
    "frontier_right": ("Swap right", "#155e92", "o"),
    "random": ("Swap random", "#9b4614", "D"),
    "slack_right": ("Slack right", "#155e92", "^"),
    "slack_random": ("Slack random", "#9b4614", "v"),
    "slack_oldest": ("Slack oldest", "#30704e", "p"),
}


@dataclass(frozen=True)
class Metrics:
    requests: int
    correct: int
    tpf: float
    efficiency: float

    def __post_init__(self):
        if not (0 <= self.correct <= self.requests and self.requests > 0
                and math.isfinite(self.tpf) and self.tpf >= 0
                and 0 <= self.efficiency <= 1):
            raise ValueError(f"Invalid cohort metrics: {self}")

    def values(self):
        return self.correct / self.requests, self.tpf, self.efficiency

    def labels(self):
        return f"{self.correct}/{self.requests}", f"{self.tpf:.3f}", f"{self.efficiency:.2%}"


@dataclass(frozen=True)
class Arm:
    policy: str
    mode: str
    k: int
    metrics: dict[int, Metrics]
    sources: tuple[str, ...]
    truncated: frozenset[int] = frozenset()


def parse_policy(policy, q):
    match = POLICY.fullmatch(policy)
    if not match:
        raise ValueError(f"Unrecognized prefix-window policy: {policy}")
    budget, masks, mode = match.groups()
    k = q - int(masks)
    if (int(budget) != q or not 0 <= k < q or mode not in MODES
            or (k == 0) != (mode == "zero")):
        raise ValueError(f"Inconsistent Q/K or mode: {policy} (Q={q})")
    return mode, k


def check_windows(metrics):
    if set(metrics) != COHORTS.keys():
        raise ValueError("Expected one all-dataset row for each of original30, added60, full90")
    a, b, full = (metrics[n] for n in COHORTS)
    if (a.correct + b.correct != full.correct
            or not math.isclose((30 * a.tpf + 60 * b.tpf) / 90, full.tpf, abs_tol=1e-9)):
        raise ValueError("Full90 counts or macro TPF disagree with original30 + added60")


def read_comparison(rows, source):
    baseline, arm, work, truncated = {}, {}, {}, set()
    identity = None
    for row in rows:
        model, q = row["model"], int(row["nominal_q"])
        base_policy, arm_policy = row["baseline_policy"], row["arm_policy"]
        current = (model, q, base_policy, arm_policy)
        if identity is None:
            identity = current
            if model not in MODELS:
                raise ValueError(f"Unrecognized model: {model}")
            if parse_policy(base_policy, q) != ("zero", 0):
                raise ValueError("Baseline must be the matched zero add-back policy")
            mode, k = parse_policy(arm_policy, q)
        elif identity != current:
            raise ValueError("A cohort CSV must contain one model, Q, and policy pair")
        n = int(row["window"])
        if n not in COHORTS or row["cohort"] != COHORTS[n][0] or n != int(row["requests"]):
            raise ValueError(f"Wrong cohort or cohort size: {row['cohort']}")
        if n in arm or int(row["failed"]) != 0:
            raise ValueError(f"Duplicate or failed cohort: {row['cohort']}")
        if int(row["truncated"]):
            truncated.add(n)
        for side, target in (("baseline", baseline), ("arm", arm)):
            target[n] = Metrics(n, int(row[f"{side}_correct"]), float(row[f"{side}_macro_tpf"]),
                                float(row[f"{side}_pooled_token_efficiency"]))
        work[n] = tuple(int(row[f"baseline_{field}"]) for field in ("N", "F", "Q", "physical_Q"))
    check_windows(baseline)
    check_windows(arm)
    if mode == "zero" and arm != baseline:
        raise ValueError("Zero self-comparison disagrees with its baseline")
    return (model, q), Arm(base_policy, "zero", 0, baseline, (source,)), work, Arm(
        arm_policy, mode, k, arm, (source,), frozenset(truncated))


def read_native(path, model):
    if not path.is_file():
        return None
    policy = f"native{MODELS[model][1]}"
    windows = {name: n for n, (name, _) in COHORTS.items()}
    metrics = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (row["model"], row["policy"], row["dataset"]) != (model, policy, "all"):
                continue
            n = windows.get(row["cohort"])
            if n != int(row["requests"]) or n in metrics:
                raise ValueError(f"Wrong or duplicate native cohort in {path}: {row['cohort']}")
            metrics[n] = Metrics(n, int(row["correct"]), float(row["macro_tpf"]),
                                 float(row["token_efficiency"]))
    if not metrics:
        return None
    check_windows(metrics)
    return Arm(policy, "native", 0, metrics, (path.name,))


def load_groups(folder, model=None, q=None):
    groups = {}
    for path in sorted(folder.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            table = csv.DictReader(stream)
            if not {"scope", "nominal_q", "arm_policy", "baseline_policy"}.issubset(table.fieldnames or ()):
                continue
            rows = [row for row in table if row["scope"] == "all"]
        if rows and not any((model is None or row["model"] == model)
                            and (q is None or row["nominal_q"] == str(q)) for row in rows):
            continue
        try:
            key, zero, work, arm = read_comparison(rows, path.name)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
        other_zero, other_work, arms = groups.setdefault(key, (zero, work, {}))
        if (zero.policy, zero.metrics, work) != (other_zero.policy, other_zero.metrics, other_work):
            raise ValueError(f"{path}: inconsistent matched zero baseline for {key}")
        if arm.mode != "zero":
            if arm.policy in arms:
                raise ValueError(f"{path}: duplicate arm {arm.policy}")
            arms[arm.policy] = arm
    if not groups:
        raise ValueError(f"No finished prefix-window comparisons in {folder} for model={model}, Q={q}")
    result = {}
    for key, (zero, _, arms) in sorted(groups.items()):
        ordered = [zero, *sorted(arms.values(), key=lambda a: (a.k, list(MODES).index(a.mode)))]
        native = read_native(folder / "native90.csv", key[0])
        result[key] = [*ordered, native] if native else ordered
    return result


def arm_style(arm, block, n):
    if arm.mode == "native":
        return f"Native B{block}", "#626e77", "P"
    name, color, marker = MODES[arm.mode]
    suffix = f", K={arm.k}"
    return name + suffix + (" *" if n in arm.truncated else ""), color, marker


def make_figure(model, q, arms):
    name, block = MODELS[model]
    native = arms[-1].mode == "native"
    last_matched = len(arms) - 1 - native
    y_positions = [i + (0.55 if a.mode == "native" else 0) for i, a in enumerate(arms)]
    fig = plt.figure(figsize=(9.4, max(7.4, 3.03 + 0.78 * len(arms))), facecolor="white")
    left = 0.171
    grid = fig.add_gridspec(3, 3, left=left, right=0.916, bottom=0.12, top=0.833,
                           wspace=0.54, hspace=0.48)
    ink, muted = "#172838", "#52616d"
    fig.text(0.035, 0.966, f"{name} B{block} | Prefix-window add-back", fontsize=20,
             color=ink, fontweight="bold", va="top")
    fig.text(0.037, 0.918, f"Nominal Q = {q}   /   Frozen90 benchmark", fontsize=11,
             color=muted, va="top")

    limits = []
    for j in range(3):
        values = [arm.metrics[n].values()[j] for arm in arms for n in COHORTS]
        lo, hi = min(values), max(values)
        pad = max((hi - lo) * 0.085, (0.008, 0.08, 0.002)[j])
        limits.append((max(0, lo - pad), hi + pad))

    for i, (n, (_, cohort)) in enumerate(COHORTS.items()):
        for j, title in enumerate(("Quality (accuracy)", "Macro TPF", "Pooled token efficiency")):
            ax = fig.add_subplot(grid[i, j])
            if j == 0:
                bounds = ax.get_position()
                fig.text(left, bounds.y1 + 0.013, cohort, color=ink, fontsize=12,
                         fontweight="bold", va="bottom")
            if i == 0:
                fig.text(ax.get_position().x0, 0.874, title, color=muted,
                         fontsize=10, fontweight="bold", va="bottom")
            # Keep a fixed numeric gutter so close/equal values never overlap.
            ax.set_xlim(*limits[j])
            ax.set_ylim(y_positions[-1] + 0.65, -0.63)
            ax.axhspan(-0.47, 0.47, color="#edf3f7", zorder=0)
            ax.vlines(arms[0].metrics[n].values()[j], -0.46, last_matched + 0.43,
                      colors="#8796a2", linestyles=(0, (2, 2)), linewidth=0.85, zorder=2)
            if native:
                ax.axhline(y_positions[-1] - 0.78, color="#a2acb4", linewidth=0.75)
            for y, arm in zip(y_positions, arms):
                label, color, marker = arm_style(arm, block, n)
                weight = "bold" if arm.mode == "zero" else "normal"
                value = arm.metrics[n]
                ax.scatter(value.values()[j], y, s=36, marker=marker, color=color,
                           edgecolor="white", linewidth=0.55, zorder=3, clip_on=False)
                annotation = ax.text(1.06, y, value.labels()[j], transform=ax.get_yaxis_transform(),
                                     fontsize=9.5, va="center", color=color, fontweight=weight)
                annotation.set_gid(f"value-{n}-{arm.policy}-{j}")
                if j == 0:
                    ax.text(-0.055, y, label, transform=ax.get_yaxis_transform(), ha="right",
                            va="center", fontsize=9, color=color, fontweight=weight)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            if j != 1:
                ticks = ax.get_xticks()
                step_pp = 100 * (ticks[1] - ticks[0])
                decimals = next(d for d in range(7) if math.isclose(round(step_pp, d), step_pp))
                ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=decimals))
            ax.set_yticks([])
            ax.grid(axis="x", color="#e0e6ea", linewidth=0.65)
            ax.set_axisbelow(True)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#a2acb4")
            ax.tick_params(axis="x", labelsize=8.5, colors=muted, length=2.5, pad=2.5)

    fig.text(0.037, 0.051, "TPF: mean per-request tokens per forward.   "
             "Token efficiency: total output tokens / total accounted query rows.",
             fontsize=8.5, color=muted)
    notes = ["Dashed lines: same-Q zero control."]
    if native:
        notes.append("Native: full-width reference, not a matched-Q arm.")
    if any(arm.truncated for arm in arms):
        notes.append("* Paired CSV reports truncation.")
    fig.text(0.037, 0.026, "   ".join(notes), fontsize=8.5, color=muted)
    return fig


def save_figure(model, q, arms, output):
    output.mkdir(parents=True, exist_ok=True)
    stem = output / f"{model}-q{q}"
    sources = ", ".join(sorted({name for arm in arms for name in arm.sources}))
    title = f"{MODELS[model][0]} prefix-window add-back at Q{q}"
    with plt.rc_context({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "svg.hashsalt": "t2t-prefix-window-v1", "pdf.fonttype": 42}):
        fig = make_figure(model, q, arms)
        try:
            for ext, meta in (
                ("pdf", {"Subject": sources, "CreationDate": None}),
                ("svg", {"Description": sources, "Date": None}),
                ("png", {"Description": sources}),
            ):
                path = stem.with_suffix(f".{ext}")
                fig.savefig(path, dpi=300, metadata={"Title": title, **meta})
                if ext == "svg":
                    lines = path.read_text(encoding="utf-8").splitlines()
                    path.write_text("\n".join(line.rstrip() for line in lines) + "\n",
                                    encoding="utf-8", newline="\n")
                print(path)
        finally:
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=HERE)
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--q", type=int, help="Nominal Q; omit to plot all available budgets")
    args = parser.parse_args()
    try:
        groups = load_groups(args.input_dir, model=args.model, q=args.q)
    except ValueError as exc:
        parser.error(str(exc))
    for (model, q), arms in groups.items():
        save_figure(model, q, arms, args.output_dir)


if __name__ == "__main__":
    main()

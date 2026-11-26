"""Draw a single budget's online scores next to its frozen-trace dose curve."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


def read(path, q, cohort):
    with path.open(newline="") as f:
        return [row for row in csv.DictReader(f) if int(row["q"]) == q and row["cohort"] == cohort]


def plot(folder, q, cohort, output):
    live = read(folder / "gemma-ratio-online.csv", q, cohort)
    replay = read(folder / "gemma-ratio-opportunity.csv", q, cohort)
    if not live or not replay:
        raise ValueError("Need both the qualified paired results and the zero-trajectory opportunity table")
    caps = sorted(int(r["k"]) for r in replay)
    ref = live[0]
    index = {(int(r["k"]), r["selector"]): r for r in live}
    styles = [("right", "#005985", "o", -.12), ("random", "#ad4e14", "^", .12)]
    fig, axs = plt.subplots(4, 1, figsize=(10.5, 8.5), sharex=True)
    fig.subplots_adjust(left=.11, right=.98, bottom=.11, top=.88, hspace=.3)
    fig.suptitle(f"DiffusionGemma | Q{q} | {cohort} | block 256", x=.11, y=.98, ha="left",
                 fontweight="bold")
    for ax, (field, baseline, title, scale, fmt) in zip(axs, (
            ("correct", "zero_correct", f"Correct / {ref['requests']}", 1, ".0f"),
            ("tpf", "zero_tpf", "Macro TPF", 1, ".2f"),
            ("efficiency", "zero_efficiency", "Token efficiency (%)", 100, ".2f"),
            ("share_all_generation", None, "Prefix share of all Q (%)", 100, ".2f"))):
        b = float(ref[baseline]) * scale if baseline else 0
        ax.axhline(b, color="#999999", lw=1, ls=":")
        ax.scatter([0], [b], marker="s", color="#303840", s=30, label="Zero injection")
        ax.annotate(format(b, fmt), (0, b), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=8)
        for name, color, marker, offset in styles:
            pts = [(i + offset, float(index[k, name][field]) * scale) for i, k in enumerate(caps)
                   if (k, name) in index]
            ax.scatter([x for x, y in pts], [y for x, y in pts], marker=marker,
                       color=color, s=37, label=name.capitalize() + " (online)", zorder=3)
            for x, y in pts:
                ax.annotate(format(y, fmt), (x, y), xytext=(0, 7 if name == "right" else -12),
                            textcoords="offset points", ha="center", fontsize=7, color=color)
        if not baseline:
            projected = {int(r["k"]): 100*float(r[field]) for r in replay}
            ax.plot(range(len(caps)), [projected[k] for k in caps], color="#62776a",
                    lw=1.3, ls="--", label="Fixed zero trajectory (not online)")
        ax.margins(y=.4)
        ax.set_ylabel(title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#ebedef")
        ax.set_axisbelow(True)
    axs[-1].set_xticks(range(len(caps)), [str(k) if k != q-1 else f"{k}\nAll slack" for k in caps])
    axs[-1].set_xlabel("Prefix cap K (token rows per selected forward); absent markers = not scored")
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.105, .955), ncol=4, frameon=False)
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--q", type=int, required=True, choices=(16, 32, 64, 128, 256))
    p.add_argument("--cohort", default="all90", choices=("original30", "added60", "all90"))
    p.add_argument("--folder", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    plot(a.folder, a.q, a.cohort, a.out)

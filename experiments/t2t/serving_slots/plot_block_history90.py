"""Plot completed 90-request tiers without interpolating missing data."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
TIERS = [4, 8, 16, 32, 64]


def rows(name):
    with (ROOT / "controls" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def curve(ax, series, field, label, color, style, shift):
    fmt = ".0f" if field == "correct" else ".2f"
    scale = 100 if field == "token_efficiency" else 1
    y = [float(series[c][field]) * scale if c in series else float("nan") for c in TIERS]
    ax.plot(range(len(TIERS)), y, style, color=color, label=label, linewidth=1.7, markersize=4)
    for j, c in enumerate(TIERS):
        if c in series:
            ax.annotate(format(y[j], fmt), (j, y[j]), xytext=(0, shift),
                        textcoords="offset points", ha="center", color=color, fontsize=8)


def main():
    controls = [r for r in rows("block-history90-audited.csv")
                if r["model"] == "llada21" and r["cohort"] == "all-0-89"]
    history = {int(r["C"]): r for r in rows("block-history90-llada.csv") if r["cohort"] == "all90"}
    ibs = {int(r["setting"]): r for r in controls if r["method"] == "ibs"}
    native, = [r for r in controls if r["method"] == "native"]
    assert all(int(r["requests"]) == 90 for r in [native, *ibs.values(), *history.values()])
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.9), layout="constrained")
    for ax, (field, title) in zip(axs, (("correct", "Correct / 90"), ("macro_tpf", "Macro TPF"),
                                        ("token_efficiency", "Physical token efficiency (%)"))):
        curve(ax, history, field, "Block-local + selected history", "#05714e", "-o", 8)
        curve(ax, ibs, field, "Optimus IBS (uncapped)", "#3157aa", "--s", -13)
        value = float(native[field]) * (100 if field == "token_efficiency" else 1)
        ax.axhline(value, color="#7a525a", linestyle=":", linewidth=1.6,
                   label=f"Native B32 ({value:.0f})" if field == "correct" else f"Native B32 ({value:.2f})")
        ax.set_xticks(range(len(TIERS)), TIERS)
        ax.margins(x=.12, y=.28)
        ax.grid(axis="y", alpha=.17)
        ax.set(xlabel="Nominal budget C", title=title)
        ax.legend(loc="best", fontsize=7, frameon=False)
    fig.suptitle("LLaDA 2.1 | frozen90 | same nominal tier, different physical Q")
    fig.savefig(ROOT / "figures/block-history90-llada.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

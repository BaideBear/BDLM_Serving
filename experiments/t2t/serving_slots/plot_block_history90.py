"""Plot completed 90-request tiers without interpolating missing data."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
TIERS = {"llada": [4, 8, 16, 32, 64], "gemma": [32, 64, 128, 256, 512]}


def rows(name):
    with (ROOT / "controls" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def curve(ax, tiers, series, other, field, label, color, style, tie_above):
    fmt = ".0f" if field == "correct" else ".2f"
    scale = 100 if field == "token_efficiency" else 1
    y = [float(series[c][field]) * scale if c in series else float("nan") for c in tiers]
    ax.plot(range(len(tiers)), y, style, color=color, label=label, linewidth=1.7, markersize=4)
    for j, c in enumerate(tiers):
        if c in series:
            diff = float(series[c][field]) - float(other[c][field])
            shift = 8 if diff > 0 or (diff == 0 and tie_above) else -13
            ax.annotate(format(y[j], fmt), (j, y[j]), xytext=(0, shift),
                        textcoords="offset points", ha="center", color=color, fontsize=8)


def plot(model):
    tiers = TIERS[model]
    label, key = ("LLaDA 2.1", "llada21") if model == "llada" else ("DiffusionGemma", "gemma")
    controls = [r for r in rows("block-history90-audited.csv")
                if r["model"] == key and r["cohort"] == "all-0-89"]
    history = {int(r["C"]): r for r in rows(f"block-history90-{model}.csv") if r["cohort"] == "all90"}
    ibs = {int(r["setting"]): r for r in controls if r["method"] == "ibs"}
    if model == "gemma":
        ibs[512], = (r for r in rows("block-history90-gemma-ibs512-h800.csv") if r["cohort"] == "all-0-89")
    native, = [r for r in controls if r["method"] == "native"]
    assert all(int(r["requests"]) == 90 for r in [native, *ibs.values(), *history.values()])
    assert set(ibs) == set(history) == set(tiers)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.9), layout="constrained")
    for ax, (field, title) in zip(axs, (("correct", "Correct / 90"), ("macro_tpf", "Macro TPF"),
                                        ("token_efficiency", "Physical token efficiency (%)"))):
        curve(ax, tiers, history, ibs, field, "Block-local + selected history", "#05714e", "-o", True)
        curve(ax, tiers, ibs, history, field, "Optimus IBS (uncapped)", "#3157aa", "--s", False)
        value = float(native[field]) * (100 if field == "token_efficiency" else 1)
        ax.axhline(value, color="#7a525a", linestyle=":", linewidth=1.6,
                   label=f"Native B{native['setting']} ({value:{'.0f' if field == 'correct' else '.2f'}})")
        ax.set_xticks(range(len(tiers)), tiers)
        ax.margins(x=.12, y=.28)
        ax.grid(axis="y", alpha=.17)
        ax.set(xlabel="Nominal budget C", title=title)
        ax.legend(loc="best", fontsize=7, frameon=False)
    fig.suptitle(f"{label} | frozen90 | same nominal tier, different physical Q")
    if model == "gemma":
        fig.supxlabel("C32–128: selected history (A800). C256 and C512: native full read (H800).", fontsize=9)
    fig.savefig(ROOT / f"figures/block-history90-{model}.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    for name in TIERS:
        plot(name)

"""Render an annotated fixed-Q256 comparison from the archived cohort table."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parent
    with (root/"cohorts.csv").open() as src:
        table = {r["arm"]:r for r in csv.DictReader(src) if r["cohort"]=="all90"}
    methods = ("native","step","paid")
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.1), constrained_layout=True)
    for ax,key,title,fmt,scale,limit in zip(
            axes, ("correct","TPF","TE"), ("Correct / 90","Tokens per forward","Token efficiency"),
            ("{:.0f}","{:.3f}","{:.2f}%"), (1,1,100), (95,32,12)):
        ys = [float(table[k][key])*scale for k in methods]
        ax.bar(range(3), ys, width=0.58, color=["#81909c","#147f72","#ae674c"])
        ax.axhline(ys[0], color="#364854", linestyle="--", linewidth=0.9)
        for idx,y in enumerate(ys):
            ax.text(idx, y+.015*limit, fmt.format(y), ha="center", fontsize=10)
        ax.set_xticks(range(3), ["Native","Step","Verify"])
        ax.set_ylim(0,limit)
        ax.set_title(title, loc="left", fontsize=11)
        ax.spines[["right","top"]].set_visible(False)
    fig.suptitle("DiffusionGemma 26B  |  fixed Q256  |  90 paired requests", fontsize=14)
    fig.savefig(root/"frozen90.png",dpi=155)
    plt.close(fig)


if __name__ == "__main__":
    main()

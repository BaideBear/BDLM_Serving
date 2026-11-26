"""Rebuild the Gemma native lag diagnostic figure from published count tables."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


def by_q(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    index = {int(r["q"]): r for r in rows}
    if set(index) != {16,32,64,128,256} or len(rows) != 5:
        raise ValueError(f"Expected five distinct budgets in {path}")
    return index


def plot(folder):
    bounded, exact = by_q(folder/"summary.csv"), by_q(folder/"exact.csv")
    plt.rcParams.update({"svg.hashsalt":"bdlm-gemma-lag-v1", "svg.fonttype":"none", "font.size":10})
    fig, ax = plt.subplots(figsize=(8.6,3.7), layout="constrained")
    labels = []
    for j,q in enumerate(sorted(exact)):
        b,e = bounded[q],exact[q]
        base = int(b["zero_total_hits"])
        if base != int(e["zero_total_hits"]):
            raise ValueError("Control differs between archives")
        lo,hi = (int(b[k])-base for k in ("lag_total_lo","lag_total_hi"))
        measured = int(e["exact_total_hits"])-base
        rand = float(b["uniform_total_hits"])-base
        if not lo<=measured<=hi:
            raise ValueError("Measured count outside archived bounds")
        ax.plot([lo,hi],[j,j],lw=4,color="#b4c0cd",solid_capstyle="round",
                label="Range from old logs" if j==0 else None)
        ax.scatter(rand,j+.09,marker="D",s=29,color="#b04c1e",label="Uniform expectation" if j==0 else None)
        ax.scatter(measured,j-.09,s=39,color="#106753",label="Measured lag rank" if j==0 else None)
        ax.annotate(f"{measured:+d}", (measured,j-.09),xytext=(5,-1),textcoords="offset points",
                    color="#075443",ha="left",va="bottom",fontsize=9)
        labels.append(f"Q{q}   ({int(b['events'])} contests)")
    ax.axvline(0, color="#697580", linewidth=.7, zorder=0)
    ax.set_yticks(range(5),labels)
    ax.set_ylim(4.55,-.68)
    ax.margins(x=.11)
    ax.grid(axis="x",lw=.45,alpha=.35)
    ax.set_axisbelow(True)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(axis="y",length=0)
    ax.set_xlabel("Change in next-round high-drift row hits vs. zero reserve")
    ax.set_title("Gemma L15 | One neutral seat, 30 native requests",loc="left",weight="bold")
    fig.legend(*ax.get_legend_handles_labels(),loc="outside lower center",ncols=3,frameon=False)
    path = folder/"lag.svg"
    fig.savefig(path,metadata={"Date":None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    plt.close(fig)


if __name__ == "__main__":
    plot(Path(__file__).resolve().parent)

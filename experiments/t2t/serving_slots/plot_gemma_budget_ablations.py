"""Plot matched fixed-budget Gemma metrics from the repository's aggregate CSVs."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


HERE = Path(__file__).resolve().parent
OFFERS = (32, 64, 128, 256, 512)
STYLES = {
    "00": ("#007c78", "o", "00  Within block"),
    "10": ("#b74f10", "s", "10  Cross-block"),
    "11": ("#8552a1", "^", "11  Cross-block + build"),
    "ibs": ("#2763ac", "v", "Optimus IBS (cap)"),
    "obs": ("#ad3e67", "D", "Optimus OBS (cap)"),
    "flow": ("#717a23", "P", "FlowBlock W1, W2"),
}
NATIVE = "#181e25"


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load(n):
    vals = {k: {} for k in (*STYLES, "native")}
    def add(k, q, eff, tpf, correct):
        sample = (float(eff), float(tpf), int(correct))
        if q in vals[k] or not (0 <= sample[0] <= 100 and sample[1] > 0 and 0 <= sample[2] <= n):
            raise ValueError(f"Duplicate or invalid {k} Q{q}: {sample}")
        vals[k][q] = sample
    for r in rows(HERE / "controls/block-ablation-gemma-v1.csv"):
        if r["model"] == "gemma" and r["cohort"] == f"frozen{n}" and int(r["n"]) == n:
            k = "native" if r["arm"] == "native256" else r["arm"]
            add(k, int(r["q"]), r["efficiency_pct"], r["macro_tpf"], r["correct"])
    cohort = {90: "all-0-89", 30: "original-0-29"}[n]
    for r in rows(HERE / "controls/fixed-family-references.csv"):
        if r["model"] != "gemma" or r["cohort"] != cohort or int(r["requests"]) != n:
            continue
        k = {"flow_w1": "flow", "flow_w2": "flow", "ibs": "ibs", "obs": "obs"}.get(r["method"])
        if k:
            add(k, int(r["budget"]), 100*float(r["efficiency"]), r["macro_TPF"], r["correct"])
    for k, expected in {"00": OFFERS, "10": OFFERS, "11": OFFERS, "ibs": OFFERS[:-1],
                        "obs": OFFERS[:-1], "flow": OFFERS[-2:], "native": (256,)}.items():
        if set(vals[k]) != set(expected):
            raise ValueError(f"Missing or unexpected matched offers: {k}")
    return vals


def render(n, out):
    data = load(n)
    native = data["native"][256]
    fig = plt.figure(figsize=(14.15, 8.35), facecolor="white")
    gs = fig.add_gridspec(1, 3, left=.063, right=.977, bottom=.458, top=.755, wspace=.25)
    axes = [fig.add_subplot(gs[0, j]) for j in range(3)]
    fig.text(.045, .944, f"DiffusionGemma 26B  |  Fixed Q budget", fontsize=23, weight="bold")
    fig.text(.045, .906, f"Frozen {n} questions   /   Physical block size 256   /   Matched GPU inference",
             fontsize=12, color="#4c5762")
    def fmt(j, x):
        return (f"{x:.2f}%", f"{x:.3f}", f"{x:.0f}/{n}")[j]
    for j, (ax, title) in enumerate(zip(axes, ["Token efficiency (%)", "Tokens per forward", f"Quality (correct / {n})"])):
        ax.set_xscale("log", base=2)
        ax.set_xlim(27, 605)
        ax.set_xticks(OFFERS, [str(q) for q in OFFERS])
        ax.set_xlabel("Fixed Q budget", color="#37434e")
        ax.set_title(title, loc="left", fontweight="bold", pad=12, fontsize=13)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e0e5e7", lw=.8)
        ax.spines[["right", "top"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#b4bdc4")
        ax.tick_params(labelsize=10)
        ax.axhline(native[j], color=NATIVE, ls=(0, (5, 3)), lw=1.65, zorder=1.7)
        for k, (c, marker, _) in STYLES.items():
            x = sorted(data[k])
            ax.plot(x, [data[k][q][j] for q in x], color=c, marker=marker, ms=6.3,
                    mfc="white" if k in ("ibs", "obs", "flow") else c, mew=1.2,
                    lw=1.9, ls="-" if k in ("00", "10", "11") else "--")
        ax.plot(256, native[j], marker="*", color=NATIVE, ls="none", ms=13,
                mec="white", mew=.6, zorder=5)
        ax.text(.02, .965, "Native Q256: " + fmt(j, native[j]), va="top", transform=ax.transAxes,
                fontsize=9, color=NATIVE, bbox={"fc": "white", "ec": "none", "alpha": .92, "pad": 1.7})
        if j == 2:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.margins(y=.16)
    handles = [Line2D([], [], color=NATIVE, ls="--", marker="*", ms=12, label="Native Q256 (reference)")]
    for k, (c, marker, label) in STYLES.items():
        handles.append(Line2D([], [], color=c, lw=1.9, ls="-" if k in ("00", "10", "11") else "--",
                              marker=marker, mfc=c if k in ("00", "10", "11") else "white", label=label))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.04, .883), ncol=4,
               frameon=False, fontsize=10, columnspacing=2.6, handlelength=2.7)
    fig.text(.045, .368, "Measured values", fontsize=13, weight="bold")
    fig.text(.192, .368, f"Each cell: efficiency (%)  /  TPF  /  correct out of {n}",
             fontsize=10, color="#4c5762")
    ax = fig.add_axes([.045, .112, .923, .23])
    ax.axis("off")
    names = ["Budget", "00", "10", "11", "Optimus IBS", "Optimus OBS", "FlowBlock"]
    table = [[f"Q{q}"] + [(" / ".join((f"{v[0]:.2f}", f"{v[1]:.3f}", str(v[2])))
                              if (v := data[k].get(q)) else "not measured") for k in STYLES] for q in OFFERS]
    tab = ax.table(cellText=table, colLabels=names, loc="center", bbox=[0, 0, 1, 1],
                   colWidths=[.07, .15, .15, .15, .16, .16, .16], cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.4)
    for (i, col), cell in tab.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#eaf0f2" if i == 0 else "#f3f6f7" if i % 2 else "white")
        if i == 0 or col == 0:
            cell.get_text().set_weight("bold")
    fig.text(.045, .078, "Black dashed line = native reference across budgets; star = its only measured budget. "
             "At Q512 the three ablation traces coincide.", fontsize=9.4, color="#47515a")
    if n == 30:
        extra = {r["row"]: r for r in rows(HERE.parents[1] / "baselines/gemma-baselines.csv")}
        r = extra["B3"]
        note = (f"B3 (variable Q, not on fixed-budget axis): {float(r['token_efficiency_pct']):.2f}% / "
                f"{float(r['tpf']):.3f} / {r['correct']}/{n}. FlowBlock W3 and W4 exceed Q512.")
    else:
        note = "Optimus uses a decoder Q cap. FlowBlock W1 and native coincide at Q256. B3 is available on frozen30 only."
    fig.text(.045, .051, note, fontsize=9.4, color="#47515a")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        path = out / f"gemma-fixed-q-frozen{n}.{ext}"
        fig.savefig(path, dpi=145, metadata={"Date": None} if ext == "svg" else {})
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "gemma-q-v1"})
    for n in (90, 30):
        render(n, args.out)

"""Redraw the five frozen six-request pairs from their compact GPU receipts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((HERE / name).read_bytes())


def metrics(arm):
    tpf = next(arm[k] for k in ("macro_TPF", "TPF", "tpf") if k in arm)
    te = next(arm[k] for k in ("TE", "te") if k in arm)
    if abs(te - arm["N"] / arm["Q"]) > 1e-8:
        raise ValueError("Token efficiency does not match the physical decoder bill")
    return arm["correct"], tpf, 100 * te


def groups():
    q8 = load("llada-q8-six.json")["totals"]
    q16 = load("llada-q16-six.json")["groups"]["all6"]
    q128 = load("gemma-q128-live-six.json")["totals"]
    seat = load("../prefix-clamp/gemma-q64-physical-seat-six.json")["totals"]
    tail = load("llada-q64-first-tail-six.json")
    arms = {k: dict(v, correct=sum(c["arms"][k]["correct"] for c in tail["cases"]))
            for k, v in tail["totals"].items()}
    return [
        ("LLaDA Q8 | IBS\nRight edge vs. spread", [q8["right"], q8["spread"]]),
        ("LLaDA Q16 | IBS\nRight edge vs. spread", [q16["right"], q16["spread"],
                                                    q16["zero_dose_reference"]]),
        ("Gemma Q128 | Live\nRight edge vs. spread", [q128["stock"], q128["treatment"]]),
        ("Gemma Q64 | IBS\nWarm MASK vs. prefix", [seat["stock"], seat["seat"]]),
        ("LLaDA Q64 | Flow W2\nFull vs. half successor", [arms["stock"], arms["candidate"]]),
    ]


def plot():
    rows = groups()
    colors = ["#48627a", "#087d70", "#ac631f"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 5.9), sharey=True)
    labels = ["Same-device control", "Single change", "No optional prefix (Q16 only)"]
    for col, (ax, name, limit) in enumerate(zip(axes, ("Correct / 6", "Tokens per forward",
                                                        "Token efficiency"), (7.6, 18.5, 42))):
        for i, (_, arms) in enumerate(rows):
            for j, arm in enumerate(arms):
                v = metrics(arm)[col]
                y = i + (j - (len(arms)-1)/2)*.23
                ax.barh(y, v, height=.18, color=colors[j])
                label = (f"{v}/6", f"{v:.3f}", f"{v:.2f}%")[col]
                ax.text(v + .015*limit, y, label, va="center", fontsize=8.6)
        ax.set_xlim(0, limit)
        ax.set_title(name, loc="left", fontweight="bold", fontsize=12)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e3e7e9")
        ax.spines[:].set_visible(False)
        ax.tick_params(axis="both", length=0, labelsize=9)
    axes[0].set_yticks(range(len(rows)), [label for label, _ in rows])
    axes[0].set_ylim(len(rows)-.54, -.56)
    fig.subplots_adjust(left=.21, right=.985, bottom=.12, top=.79, wspace=.18)
    fig.suptitle("Fixed-budget single-request ablations", x=.035, y=.965,
                 ha="left", fontsize=16, fontweight="bold")
    fig.legend([Patch(color=c) for c in colors], labels, loc="upper left",
               bbox_to_anchor=(.027, .918), ncol=3, frameon=False)
    fig.text(.035, .032, "Same six requests in each arm (7, 13, 25, 33, 55, 71). "
             "TPF = mean of per-request N/F; efficiency = total N / actual decoder Q.", fontsize=9)
    fig.savefig(HERE / "fixed-q-six.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot()

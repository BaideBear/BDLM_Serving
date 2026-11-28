"""Plot verified fixed-offer files only. Run from any directory."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(sources, output):
    fig, axes = plt.subplots(len(sources), 3, figsize=(11, 1.3+2.65*len(sources)), squeeze=False)
    splits = ("original30", "new60", "frozen90")
    for axrow, source in zip(axes, sources):
        doc = json.loads(source.read_bytes())
        model = {"llada":"LLaDA 2.1", "gemma":"DiffusionGemma"}[doc["model"]]
        vals = [doc["cohorts"][s] for s in splits]
        for col, (key, title) in enumerate((("correct","Correct / total"), ("tpf","Tokens per forward"),
                                          ("te","Token efficiency"))):
            ax = axrow[col]
            for name, label, color, style, offset in (("stock","Paid IBS","#46576a","--",-9),
                                                      ("apply","Repeat agreement","#007966","-",7)):
                def value(v):
                    return 100*v[name][key]/v["requests"] if key=="correct" else v[name][key]*(100 if key=="te" else 1)
                y = [value(v) for v in vals]
                ax.plot(range(3), y, linestyle=style, marker="o", color=color, label=label)
                for i, (v, point) in enumerate(zip(vals, y)):
                    text = f'{v[name][key]}/{v["requests"]}' if key=="correct" else (
                        f"{point:.2f}%" if key=="te" else f"{point:.3f}")
                    ax.annotate(text, (i,point), xytext=(0,offset), textcoords="offset points",
                                va="top" if offset<0 else "bottom", ha="center", color=color, fontsize=9)
            ax.set_xticks(range(3), ["Original 30","Additional 60","All 90"])
            ax.set_xlim(-.32,2.32)
            ax.margins(y=.34)
            ax.spines[["top","right"]].set_visible(False)
            ax.set_title(title)
        axrow[0].set_ylabel(f"{model} Q{doc['fixed_q']}\nAccuracy (%)")
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc="upper right", frameon=False, ncol=2)
    fig.suptitle("Fixed budget, matched stopping rule", x=.06, ha="left", fontweight="bold")
    fig.text(.065,.035,"Same-device pairs. All 90 includes both cohorts; horizontal categories are not budgets.", fontsize=9)
    fig.tight_layout(rect=(.01,.09,1,.92))
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    plot(a.inputs, a.out)

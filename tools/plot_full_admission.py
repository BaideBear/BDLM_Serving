"""Render fixed90_gate's audited rows only; no partial or imputed series."""
import argparse
import csv
import json
from pathlib import Path


def draw(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    matrix = json.loads((folder/"full-admission-status.json").read_bytes())
    with (folder/"metrics.csv").open() as stream:
        refs = list(csv.DictReader(stream))
    names = ["Full admission", "IBS capped", "Native", "Flow W1", "Flow W2"]
    colors = ["#00755b", "#3c5f9f", "#4e4e4e", "#b65319", "#893c6f"]
    markers = ["o", "^", "s", "+", "D"]
    for model in ("llada21", "gemma"):
        cells = [c for c in matrix["tiers"] if c["model"] == model]
        if not any("cohorts" in c for c in cells):
            continue
        offers = [c["q"] for c in cells]
        values = {}
        for c in cells:
            if "cohorts" in c:
                values["Full admission", c["q"]] = c["cohorts"]["frozen90"]
        for r in refs:
            if r["model"] == model and r["cohort"] == "frozen90" and r["method"] in names[1:]:
                values[r["method"], int(r["budget"])] = r
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
        fig.subplots_adjust(left=.055, right=.98, top=.77, bottom=.42, wspace=.23)
        for ax, (key, label, factor, precision) in zip(axes, [
                ("correct", "Correct / 90", 1, 0), ("macro_TPF", "Mean request TPF", 1, 3),
                ("token_efficiency", "Token efficiency (%)", 100, 2)]):
            grid = []
            for j, name in enumerate(names):
                ys = []
                for q in offers:
                    v = values.get((name,q),{}).get(key)
                    ys.append(float(v)*factor if v not in (None, "") else float("nan"))
                style = "-" if j == 0 else "--"
                ax.plot(range(len(offers)), ys, style, color=colors[j], marker=markers[j], lw=1.7)
                if sum(v == v for v in ys) == 1:
                    i = next(i for i,v in enumerate(ys) if v == v)
                    ax.hlines(ys[i], i-.15, i+.15, colors=colors[j], linestyles=style)
                grid.append([name]+[f"{v:.{precision}f}" if v == v else "-" for v in ys])
            ax.set(xticks=range(len(offers)), xticklabels=offers, title=label, xlabel="Rolling Q offer")
            ax.grid(axis="y", alpha=.2)
            ax.spines[["top","right"]].set_visible(False)
            tb=ax.table(cellText=grid, colLabels=["Q", *offers], colWidths=[.28]+[.144]*5,
                        cellLoc="center", bbox=[-.04,-.91,1.04,.58])
            tb.auto_set_font_size(False)
            tb.set_fontsize(7.8)
            for (i,j), cell in tb.get_celld().items():
                cell.set_edgecolor("#d9dfe0")
                if not i:
                    cell.set_facecolor("#edf2f2")
                elif j == 0:
                    cell.get_text().set_color(colors[i-1])
        fig.suptitle(("LLaDA 2.1" if model=="llada21" else "DiffusionGemma")+
                     ": fixed-offer requests on the same 90 prompts", fontsize=18, x=.055, ha="left")
        fig.legend([Line2D([],[],color=c,marker=m,linestyle="-" if i==0 else "--")
                    for i,(c,m) in enumerate(zip(colors,markers))],names,loc="upper center",
                   bbox_to_anchor=(.52,.921),ncol=len(names),frameon=False)
        fig.text(.045,.022,"Only measured budgets shown. Dashed = baseline. The candidate may exceed its offer "
                 "on first successor projection; TE = final non-special tokens / actual history-reading decoder Q.")
        fig.savefig(folder/f"{model}-full-admission-frozen90.png", dpi=145, facecolor="white")
        plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path, nargs="?", default=Path(__file__).resolve().parents[1]/
                   "experiments/t2t/serving_slots/controls/fixed-request-comparison")
    draw(p.parse_args().folder)

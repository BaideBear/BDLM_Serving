"""Plot only verified fixed-Q16 cohorts. Repeated six is a subset, not extra data."""
import argparse
import json
from pathlib import Path


def draw(source, dest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(source.read_bytes())
    assert (data["model"], data["fixed_q"]) == ("llada21", 16)
    cohorts = [("screen6","First 6"), ("outside84","Other 84"), ("frozen90","All 90")]
    fig, axes = plt.subplots(1,3,figsize=(12.9,4.4))
    arms = [("stock","control","Paid IBS","#8995a5"),("apply","control","Repeat","#387093"),
            ("apply","allowance","Repeat + allowance","#ad5b37")]
    for ax, metric, title in zip(axes, ("correct","tpf","te"),
                                 ("Correct answers (%)", "Mean request TPF", "Token efficiency (%)")):
        for j,(parent,arm,name,color) in enumerate(arms):
            for i,(key,_) in enumerate(cohorts):
                c = data["compared_to"][parent]["cohorts"][key]
                value = c[arm][metric]
                y = value*100/c["n"] if metric=="correct" else value if metric=="tpf" else value*100
                bar = ax.bar(i+(j-1)*.25,y,width=.235,color=color,label=name if i==0 else None)
                text = f"{value}/{c['n']}" if metric == "correct" else f"{y:.2f}"
                padding = 8 if metric == "correct" and j == 1 else 2
                ax.bar_label(bar,labels=[text],padding=padding,fontsize=7)
        ax.set_xticks(range(3),[v for _,v in cohorts])
        ax.set_title(title,fontsize=11)
        ax.margins(y=.18)
        ax.grid(axis="y",alpha=.15)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("LLaDA 2.1 | Fixed Q16 allowance confirmation",x=.05,ha="left")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncols=3,frameon=False)
    fig.subplots_adjust(left=.055,right=.99,top=.83,bottom=.18,wspace=.22)
    fig.savefig(dest,dpi=150)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("dest",type=Path)
    a=p.parse_args()
    draw(a.source,a.dest)

"""Plot the compact verified Q32 confirmation, without raw traces or generations."""
import argparse
import json
from pathlib import Path


def render(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    groups = summary["cohorts"]["32"]
    if summary["verified"] != 90 or groups["all"]["pairs"] != 90:
        raise ValueError("Only render the sealed 90-pair result")
    views = [("all","All 90"), ("original30_panel","Original 30"), ("added60_panel","Added 60"),
             ("not_on_these_two_panels75","Outside pilots (75)"),
             ("excluding_prior_long_index51","Excl. #51 (89)")]
    fig,axes = plt.subplots(1,3,figsize=(12.4,4.4),layout="constrained")
    fig.suptitle("DiffusionGemma | Fixed Q32 | Slack-only prefix refresh",fontsize=14,fontweight="bold")
    for ax,(title,key) in zip(axes,[("Quality (%)","correct"),("Tokens per forward","macro_TPF"),
                                    ("Token efficiency (%)","pooled_TE")]):
        values=[]
        for y,(name,label) in enumerate(views):
            r=groups[name]
            value=lambda arm: (100*r[arm][key]/r["pairs"] if key=="correct" else
                               r[arm][key]*(100 if key=="pooled_TE" else 1))
            x0,x1=value("off"),value("on")
            values.extend((x0,x1))
            ax.plot([x0,x1],[y,y],color="#a8adb0",lw=1.4)
            ax.plot(x0,y,marker="o",mfc="white",mec="#365b79",ms=7,zorder=3)
            ax.plot(x1,y,marker="x",mec="#b13e36",mew=1.6,ms=7,zorder=4)
            text=(f'{r["off"][key]}/{r["pairs"]} → {r["on"][key]}/{r["pairs"]}'
                  if key=="correct" else f"{x0:.2f} → {x1:.2f}")
            ax.annotate(text,(max(x0,x1),y),xytext=(8,0),textcoords="offset points",
                        va="center",fontsize=8.5)
        width=max(values)-min(values)
        unit=max(width,1)
        ax.set_xlim(min(values)-.15*unit,max(values)+.95*unit)
        ax.set_ylim(len(views)-.5,-.5)
        ax.set_yticks(range(len(views)),[label for _,label in views] if ax is axes[0] else [])
        ax.set_title(title,fontsize=11)
        ax.grid(axis="x",alpha=.2)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(axis="y",length=0)
    fig.legend(handles=[
        Line2D([],[],marker="o",color="#365b79",mfc="white",ls="--",label="Capped IBS"),
        Line2D([],[],marker="x",color="#b13e36",ls="-",label="IBS + prefix refresh")],
        loc="outside lower center",ncol=2,frameon=False)
    fig.savefig(out,dpi=160,bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    render(json.loads(args.summary.read_bytes()),args.output)

"""Render frozen90 and its two constituent cohorts from the sealed audit outputs."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    controls = Path(__file__).resolve().parents[1]/"experiments/t2t/serving_slots/controls"
    folder = controls/"llada-flow-active-w2"
    ll = json.loads((folder/"summary.json").read_bytes())
    gg = json.loads((folder/"gemma-decoder-bills.json").read_bytes())
    with (controls/"gemma-fixed512-active-vs-flow.csv").open() as f:
        scores = {int(r["requests"]):r for r in csv.DictReader(f) if r["scope"] == "all"}
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    colors = ("#596579", "#16796d")
    for row, model in enumerate(("LLaDA 2.1  |  Q64", "DiffusionGemma  |  Q512")):
        for col, label in enumerate(("Correct answers (%)", "TPF", "Token efficiency (%)")):
            ax = axes[row,col]
            series = []
            for a, arm in enumerate(("flow_w2","active_w2")):
                ys, texts = [], []
                for k, n in (("original30",30),("new60",60),("frozen90",90)):
                    if row == 0:
                        d = ll[k][arm]
                        correct, tpf, te = d["correct"], d["macro_TPF"], d["pooled_TE"]
                    else:
                        d = gg[arm][k]
                        side = ("baseline","arm")[a]
                        s = scores[n]
                        assert int(s[side+"_N"]) == d["N"] and int(s[side+"_F"]) == d["F"]
                        assert int(s[side+"_Q"]) == d["decoder_Q"]+d["encoder_Q"]
                        assert abs(float(s[side+"_macro_tpf"])-d["macro_TPF"]) < 1.e-8
                        correct, tpf, te = int(s[side+"_correct"]), d["macro_TPF"], d["efficiency"]
                    y = (100*correct/n, tpf, 100*te)[col]
                    ys.append(y)
                    texts.append((f"{correct}/{n}", f"{tpf:.3f}", f"{100*te:.2f}%")[col])
                ax.plot(range(3),ys, ("o--","o-")[a],color=colors[a],lw=1.8,ms=4)
                series.append((ys, texts))
            for a,(ys,texts) in enumerate(series):
                for j,(y,txt) in enumerate(zip(ys,texts)):
                    other = series[1-a][0][j]
                    above = y > other or (a == 1 and y == other)
                    ax.annotate(txt,(j,y),xytext=(0,7 if above else -13),textcoords="offset points",
                                ha="center",color=colors[a],fontsize=8)
            if row == 0:
                ax.set_title(label,fontweight="bold",fontsize=10,pad=12)
            if col == 0:
                ax.set_ylabel(model,fontsize=9,fontweight="bold")
            ax.set_xticks(range(3),["Original 30","Added 60","All 90"],fontsize=8)
            ax.margins(x=.16,y=.36)
            ax.spines[["right","top"]].set_visible(False)
            ax.grid(axis="y",lw=.5,alpha=.25)
            ax.tick_params(axis="y",labelsize=8)
    fig.suptitle("FlowBlock W2 at a fixed two-block offer",x=.08,y=.98,ha="left",size=17,weight="bold")
    fig.legend([Line2D([],[],ls=s,color=c,marker="o",markersize=4) for s,c in zip(("--","-"),colors)],
               ["Original W2","Project only open blocks"],frameon=False,ncol=2,loc="upper left",
               bbox_to_anchor=(.073,.935),fontsize=9)
    fig.text(.08,.035,"TPF = mean of request N/F (generation encoder included).   Efficiency = total N / actual history-reading decoder Q.",fontsize=8)
    fig.subplots_adjust(left=.08,right=.98,bottom=.13,top=.80,hspace=.52,wspace=.35)
    fig.savefig(folder/"both-models.png",dpi=155)
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Plot paired phase controls and the two gate ablations from the completed Gemma CSV."""
import argparse
import csv
from pathlib import Path


def plot(source,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with source.open() as f:
        rows = {r["policy"]:r for r in csv.DictReader(f) if r["cohort"]=="all-0-89"}
    ids = [f"gate_{rule}_{phase}" for phase in ("high","low") for rule in ("all","tail","early")]
    if set(rows)!=set(ids) or any(int(rows[p]["requests"])!=90 for p in ids):
        raise ValueError("Missing phase or incomplete suite")
    y = [0,1,2,3.8,4.8,5.8]
    fig,axes=plt.subplots(1,4,figsize=(13,3.7),layout="constrained",sharey=True)
    labels=[f"{'HHLL' if p.endswith('high') else 'LLHH'} | {p.split('_')[1]}" for p in ids]
    for ax,field,title,scale,fmt in zip(axes,
        ("correct","macro_tpf","pooled_token_efficiency","physical_Q"),
        ("Correct / 90","Mean tokens / forward","Token efficiency (%)","Physical Q (thousands)"),
        (1,1,100,.001),(".0f",".3f",".2f",".1f"),strict=True):
        vals=[scale*float(rows[p][field]) for p in ids]
        bars=ax.barh(y,vals,height=.7,color=["#4c6371","#167166","#b46a34"]*2)
        ax.bar_label(bars,labels=[format(v,fmt) for v in vals],padding=3,fontsize=9)
        ax.set_xlim(0,max(vals)*1.29)
        ax.set_title(title,fontsize=11)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_yticks(y,labels)
    axes[0].invert_yaxis()
    fig.suptitle("DiffusionGemma | native Q256 / sparse Q128 | frozen90")
    fig.savefig(out,dpi=160)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("out",type=Path)
    plot(**vars(p.parse_args()))

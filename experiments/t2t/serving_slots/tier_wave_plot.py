"""Plot both complete LLaDA wave comparisons from their checked-in paired CSVs."""
import argparse
import csv
from pathlib import Path


def plot(folder, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels,rows=[],[]
    for q,src,method in ((32,"llada21-native32-vs-flow-w1.csv","Native"),
                         (64,"llada21-dual64-vs-flow-w2.csv","Dual canvas")):
        with (folder/src).open() as f:
            items=[r for r in csv.DictReader(f) if r["scope"]=="all" and r["window"]=="90"]
        row,=items
        if int(row["nominal_q"])!=q or row["requests"]!="90":
            raise ValueError("Incomplete or mislabeled tier")
        for side,name in (("baseline",f"FlowBlock W{q//32}"),("arm",method)):
            labels.append(f"Q{q}   {name}")
            rows.append([float(row[f"{side}_{k}"]) for k in
                         ("correct","macro_tpf","pooled_token_efficiency","physical_Q")])
    fig,axes=plt.subplots(1,4,figsize=(12.8,3.1),sharey=True,layout="constrained")
    for j,(ax,title,scale,fmt) in enumerate(zip(axes,("Correct / 90","Macro TPF","Token efficiency (%)",
                                     "Physical Q (thousands)"),(1,1,100,.001),(".0f",".3f",".2f",".2f"),strict=True)):
        vals=[r[j]*scale for r in rows]
        y=[0,1,2.5,3.5]
        bars=ax.barh(y,vals,color=["#087c78","#65758a"]*2,height=.68)
        ax.bar_label(bars,labels=[format(v,fmt) for v in vals],padding=3,fontsize=9)
        ax.set_xlim(0,max(vals)*1.32)
        ax.set_title(title,fontsize=11)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_yticks(y,labels)
    axes[0].invert_yaxis()
    fig.suptitle("LLaDA 2.1  |  Single-block Q32 and true two-block Q64  |  frozen90")
    fig.savefig(output,dpi=170)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("controls",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    plot(a.controls,a.out)

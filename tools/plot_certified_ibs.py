"""Plot only sealed 90-case, same-device IBS stop controls."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def scores(rows):
    n, f, q = (sum(r[k] for r in rows) for k in ("N","F","Q"))
    return dict(correct=sum(r["correct"] for r in rows),tpf=mean(r["N"]/r["F"] for r in rows),
                efficiency=100*n/q, N=n,F=f,Q=q)


def collect(path, model):
    if model == "Gemma Q32":
        doc=json.loads(path.read_bytes())
        rows=doc["cases"]
        assert doc["requests"]==90 and doc["q"]==32 and "rc=0" in doc["release"]
        groups={"Stock":"stock","Wait for zero debt":"strict","Pay and stop":"paid"}
        def pick(r,k):
            a=r["arms"][k]
            return {**a["bill"],"N":a["N"],"correct":a["correct"]}
    else:
        rows=[json.loads(x) for x in path.read_bytes().splitlines()]
        groups={"Stock":"stock","Pay and stop":"certificate"}
        def pick(r,k):
            return r["arms"][k]
    assert sorted(r["index"] for r in rows)==list(range(90))
    return {tag:{name:scores([pick(r,k) for r in rows if test(r["index"])])
                 for name,k in groups.items()}
            for tag,test in (("original30",lambda i:i<30),("new60",lambda i:i>=30),
                             ("frozen90",lambda i:True))}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gemma",type=Path,required=True,help="verified q32-frozen90-summary.json")
    ap.add_argument("--llada",type=Path,help="combined verified 90-case compact receipt NDJSON")
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    if not __debug__:
        raise RuntimeError("assertions must be enabled")
    data={}
    if a.llada:
        data["LLaDA Q4"]=collect(a.llada,"LLaDA Q4")
    data["Gemma Q32"]=collect(a.gemma,"Gemma Q32")
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/"cohorts.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=("model","cohort","policy","correct","tpf","efficiency","N","F","Q"),
                              lineterminator="\n")
        writer.writeheader()
        for model,cohorts in data.items():
            for cohort,arms in cohorts.items():
                for arm,v in arms.items():
                    writer.writerow(dict(model=model,cohort=cohort,policy=arm,**v))
    fig,axes=plt.subplots(len(data),3,figsize=(10.8,1.65+2.65*len(data)),squeeze=False)
    palette={"Stock":"#48596c","Wait for zero debt":"#a36217","Pay and stop":"#157362"}
    for axrow,(model,cohorts) in zip(axes,data.items()):
        arms=cohorts["frozen90"]
        for ax,(k,title,fmt) in zip(axrow,[("correct","Correct / 90",".0f"),("tpf","Tokens per forward",".3f"),
                                               ("efficiency","Token efficiency (%)",".2f")]):
            names=list(arms)
            vs=[v[k] for v in arms.values()]
            ax.axhline(vs[0],color=palette["Stock"],ls="--",lw=1,zorder=1)
            ax.scatter(range(len(vs)),vs,c=[palette[n] for n in names],s=47,zorder=2)
            for i,v in enumerate(vs):
                ax.annotate(format(v,fmt),(i,v),xytext=(0,8),textcoords="offset points",ha="center")
            pad=max(max(vs)-min(vs),max(vs)*.08,1.)
            ax.set_ylim(min(vs)-.36*pad,max(vs)+.55*pad)
            ax.set_xlim(-.45,len(vs)-.55)
            ax.set_xticks(range(len(names)),[n.replace("Wait for zero debt","Wait for\nzero debt").replace("Pay and stop","Pay and\nstop") for n in names])
            ax.set_title(title,loc="left",fontweight="bold",fontsize=11)
            ax.spines[["top","right"]].set_visible(False)
            ax.tick_params(labelsize=9)
        axrow[0].set_ylabel(model,fontweight="bold")
    fig.suptitle("What an irrevocable EOS changes in strict IBS",x=.065,ha="left",fontweight="bold",fontsize=15)
    fig.text(.065,.025,"90 frozen requests per policy; stock shown dashed. Decoder Q is capped on every call. Generation encoders count toward F.",fontsize=9)
    fig.tight_layout(rect=(.015,.075,.995,.91))
    fig.savefig(a.out/"frozen90.png",dpi=155)
    plt.close(fig)


if __name__=="__main__":
    main()

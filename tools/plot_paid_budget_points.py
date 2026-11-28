"""Plot only completed fixed offers vs their own IBS and archived comparators."""
from argparse import ArgumentParser
import csv
import json
from pathlib import Path


OFFERS = {"gemma": [32,64,128,256,512], "llada": [4,8,16,32,64]}
NAMES = {"gemma":"DiffusionGemma", "llada":"LLaDA 2.1"}
METHODS = {"Repeat + paid IBS": ("#006751","o"), "Matched paid IBS": ("#265997","s"),
           "Optimus IBS (nominal)": ("#617487","v"),
           "Live (historical)": ("#884e87","D"), "Native (historical)": ("#333333","x"),
           "Flow W1/W2 (historical)": ("#b35b13","^")}


def collect(files, historical):
    rows={}
    for path in files:
        doc=json.loads(path.read_bytes())
        model,q=doc["model"],doc["fixed_q"]
        v=doc["cohorts"]["frozen90"]
        if v["requests"] != 90 or q not in OFFERS[model] or not doc.get("hashes"):
            raise ValueError("incomplete or uncertified fixed-budget point")
        for arm,name in (("stock","Matched paid IBS"),("apply","Repeat + paid IBS")):
            x=v[arm]
            key=(model,q,name)
            if key in rows or x["correct"] not in range(91) or x["F"]<=0 or x["Q"]<=0:
                raise ValueError("duplicate or unbilled point")
            rows[key]=(x["correct"],x["tpf"],x["te"]*100)
    present={m for m,_,_ in rows}
    remap={"Live profile":"Live (historical)","Native":"Native (historical)","IBS nominal":"Optimus IBS (nominal)",
           "Flow W1":"Flow W1/W2 (historical)","Flow W2":"Flow W1/W2 (historical)"}
    with historical.open(newline="") as f:
        for r in csv.DictReader(f):
            model={"gemma":"gemma","llada21":"llada"}[r["model"]]
            if model not in present or r["method"] not in remap or r["cohort"] != "frozen90":
                continue
            q=int(r["budget"])
            key=(model,q,remap[r["method"]])
            if int(r["n"])!=90 or key in rows or q not in OFFERS[model]:
                raise ValueError("bad historical registration")
            rows[key]=(int(r["correct"]),float(r["macro_TPF"]),
                       100*float(r["token_efficiency"]) if r["token_efficiency"] else float("nan"))
            if r["method"] == "Flow W2":
                directory=historical.parent.parent/"llada-flow-active-w2"
                if model=="llada":
                    audit=json.loads((directory/"summary.json").read_bytes())["frozen90"]["flow_w2"]
                    denom=audit["Q"]
                else:
                    audit=json.loads((directory/"gemma-decoder-bills.json").read_bytes())["flow_w2"]["frozen90"]
                    denom=audit["decoder_Q"]
                if (audit["N"],audit["F"])!=(int(r["N"]),int(r["F"])) or abs(audit["macro_TPF"]-rows[key][1])>1e-9:
                    raise ValueError("not the same native Flow W2 registration")
                rows[key]=(*rows[key][:2],100*audit["N"]/denom)
    return rows


def draw(rows,out,matched_live=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from math import isnan
    models=[m for m in ("llada","gemma") if any(k[0]==m for k in rows)]
    fig=plt.figure(figsize=(16.3,1.5+5.0*len(models)))
    sections=fig.add_gridspec(len(models),1,left=.06,right=.985,top=1-1.23/fig.get_figheight(),
                              bottom=.56/fig.get_figheight(),hspace=.34)
    handles={}
    if matched_live:
        conf=json.loads(matched_live.read_bytes())
        point=conf["live"]
        if (conf["gpu"]!="GPU-94c61e79-d80b-cae2-d85f-6bd2a2c99a1c" or point["count"]!=90
                or "gemma" not in models or not conf["raw_sha256"]):
            raise ValueError("invalid A3 Live confirmation")
        matched=(point["correct"],point["macro_tpf"],100*point["pooled_te"])
    for index,model in enumerate(models):
        panel=sections[index].subgridspec(2,3,height_ratios=(2.25,1),hspace=.49,wspace=.25)
        qs=OFFERS[model]
        for j in range(3):
            ax=fig.add_subplot(panel[0,j])
            tabax=fig.add_subplot(panel[1,j])
            tabax.axis("off")
            table=[]
            for name,(color,marker) in METHODS.items():
                y=[rows.get((model,q,name),(float("nan"),)*3)[j] for q in qs]
                table.append([name]+["-" if isnan(v) else f"{v:.0f}" if j==0 else f"{v:.2f}" for v in y])
                if all(isnan(v) for v in y): continue
                line,=ax.plot(range(5),y,marker=marker,color=color,linestyle="-" if name.startswith("Repeat") else "--",
                              label=name,linewidth=1.6,markersize=5.5)
                handles[name]=line
            if matched_live and model=="gemma":
                ax.scatter([qs.index(128)],[matched[j]],s=115,marker="o",facecolors="none",
                           edgecolors="#121212",linewidths=1.1,zorder=8)
            ax.set(title=("Correct / 90","Tokens per forward","Token efficiency (%)")[j],
                   xticks=range(5),xticklabels=qs,xlabel="Fixed decoder budget Q")
            if j==0:ax.set_ylabel(NAMES[model],weight="bold")
            ax.grid(axis="y",alpha=.18);ax.spines[["right","top"]].set_visible(False)
            tbl=tabax.table(cellText=table,colLabels=["Fixed budget",*qs],cellLoc="center",
                            colWidths=[.39,*([.122]*5)],bbox=[-.035,0,1.055,1])
            tbl.auto_set_font_size(False);tbl.set_fontsize(7.1)
            for (i,k),c in tbl.get_celld().items():
                c.set_edgecolor("#dce1df"); c.set_linewidth(.45)
                if i==0:c.set_facecolor("#eef4f1")
                if i>0 and k==0:c.get_text().set_color(list(METHODS.values())[i-1][0])
    fig.suptitle("Paid admission at fixed Q | Frozen 90",x=.06,y=.986,ha="left",fontweight="bold",fontsize=15)
    fig.legend(handles.values(),handles.keys(),loc="upper center",bbox_to_anchor=(.51,.95),ncol=6,frameon=False)
    foot="Matched IBS = same GPU, scorer and stopping rule per offer. "
    if matched_live:
        foot+=f"Ring = same-A3 Gemma Q128 Live ({matched[0]}/90, {matched[1]:.3f} TPF, {matched[2]:.2f}% TE). "
    fig.text(.06,.015,foot+"Other references historical. Missing is not zero.",fontsize=9)
    fig.savefig(out,dpi=145,bbox_inches="tight",pad_inches=.17)
    plt.close(fig)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("files",nargs="+",type=Path)
    p.add_argument("--historical",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--matched-live",type=Path)
    args=p.parse_args()
    draw(collect(args.files,args.historical),args.out,args.matched_live)

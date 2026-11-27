"""Compare complete fixed-offer request cohorts with nominal and capped IBS."""
import csv
import json
from math import isclose
from pathlib import Path

from .factor_tables import write_csv

ROOT=Path(__file__).resolve().parent/"controls"
DEST=ROOT/"fixed-request-comparison"
MODELS={"llada21":(4,8,16,32,64),"gemma":(32,64,128,256,512)}
PROFILES={"llada21":"quarter_edit_warm","gemma":"dense_stop_joint_verified_eos"}
COHORTS={"original30":30,"new60":60,"frozen90":90}
NAMES={"added60":"new60","all90":"frozen90","original-0-29":"original30",
       "added-30-89":"new60","all-0-89":"frozen90"}


def read(name):
    with (ROOT/name).open(newline="") as f:
        return list(csv.DictReader(f))


def collect():
    rows={}

    def add(model,cohort,q,method,n,correct,tpf,output,f,history,source,peak=None):
        cohort=NAMES.get(cohort,cohort)
        q,n,correct,output,f=map(int,(q,n,correct,output,f))
        history=None if history in (None,"") else int(history)
        key=(model,cohort,q,method)
        if (key in rows or q not in MODELS[model] or COHORTS[cohort]!=n
                or not 0<=correct<=n or f<n or output<0 or not 0<float(tpf)<=q*2
                or (peak is not None and not 0<int(peak)<=q)
                or (history is not None and history<=0)):
            raise ValueError(f"Invalid or duplicate aggregate: {key}")
        rows[key]=dict(model=model,cohort=cohort,budget=q,method=method,n=n,correct=correct,
                       macro_TPF=float(tpf),token_efficiency=output/history if history else None,
                       N=output,F=f,history_Q=history,peak_decoder_Q=peak,source=source)

    lower=json.loads((DEST/"llada-lower.json").read_bytes())
    if lower["profile"]!=PROFILES["llada21"]:
        raise ValueError("Wrong low-budget request")
    for q,t in lower["tiers"].items():
        for c,r in t["cohorts"].items():
            add("llada21",c,q,"Request",r["n"],r["correct"],r["macro_TPF"],r["N"],r["F"],r["Q"],
                "fixed-request-comparison/llada-lower.json",r["peak"])
    name="llada-close/paired.json"
    for c,v in json.loads((ROOT/name).read_bytes())["cohorts"].items():
        if c not in COHORTS: continue
        r=v["arms"]["edit_once"]
        add("llada21",c,32,"Request",v["paired"],r["correct"],r["macro_TPF"],r["N"],r["F"],r["Q"],name)
    name="llada-paid-head/summary-quarter-warm-single-factor.csv"
    for r in read(name):
        if r["new_strict_cap"]!="True": raise ValueError("Unverified Q64")
        add("llada21",r["cohort"],64,"Request",r["requests"],r["new_correct"],r["new_macro_TPF"],r["new_N"],
            r["new_F"],r["new_Q"],name)
    name="gemma-factors/fixed-profile/verified-eos-vs-baselines.csv"
    for r in read(name):
        if r["method"]!="Verified EOS request": continue
        add("gemma",r["cohort"],r["budget"],"Request",r["n"],r["correct"],r["macro_TPF"],
            r["N"],r["F"],r["history_Q"],r["source"],r["peak_decoder_Q"])
    name="block-history90-matrix.csv"
    for r in read(name):
        m,b=r["model"],int(r["setting"])
        if r["method"]=="flow" and b in (1,2):
            label=f"Flow W{b}"
            b*=MODELS[m][-2]
        elif r["method"] in ("ibs","native"):
            label="IBS nominal" if r["method"]=="ibs" else "Native"
        else:
            continue
        if b not in MODELS[m]: continue
        add(m,r["cohort"],b,label,r["requests"],r["correct"],r["macro_tpf"],r["N"],r["F"],r["prefix_kv_Q"],name)
    name="fixed-family-references.csv"
    for r in read(name):
        if r["method"]!="ibs": continue
        # Capped IBS projects just the selected rows against nonempty history.
        # encoder_Q belongs in the physical bill and its forwards in F, never in TE.
        if int(r["decoder_Q"])+int(r["encoder_Q"])!=int(r["Q"]):
            raise ValueError("Broken IBS physical/decoder split")
        add(r["model"],r["cohort"],r["budget"],"IBS capped",r["requests"],r["correct"],r["macro_TPF"],
            r["N"],r["F"],r["decoder_Q"],name,r["max_decoder_Q"])
    for model,tiers in MODELS.items():
        for q in tiers:
            for c in COHORTS:
                methods={k[3] for k in rows if k[:3]==(model,c,q)}
                expected={"Request","IBS nominal"}|({"IBS capped"} if q!=tiers[-1] else {"Flow W2"})
                if q==tiers[-2]:expected|={"Native","Flow W1"}
                if model=="llada21" and q==4 and c not in lower["tiers"]["4"]["cohorts"]:
                    expected.remove("Request")
                if methods!=expected:raise ValueError(f"Wrong coverage: {model} {q} {c}: {methods}")
            for method in methods:
                group=[rows.get((model,c,q,method)) for c in COHORTS]
                if not all(group):continue
                a,b,total=group
                if any(a[k]+b[k]!=total[k] for k in ("n","correct","N","F")) or (
                        total["history_Q"] is not None and a["history_Q"]+b["history_Q"]!=total["history_Q"]
                        ) or not isclose((a["macro_TPF"]+2*b["macro_TPF"])/3,total["macro_TPF"],abs_tol=.001):
                    raise ValueError(f"Cohorts do not reconcile: {model} {q} {method}")
    return list(rows.values())


def render(rows):
    lines=["# Fixed offers: published requests versus measured baselines","",
           "One independent request at the same offer from prompt to final output. "
           "`Request` means `quarter_edit_warm` for LLaDA and `dense_stop_joint_verified_eos` for Gemma. "
           "They are not the best trial chosen separately at each point.","",
           "Each cell: **correct / macro TPF / token efficiency (%)**. Missed or unavailable metrics are `n/a`. "
           "The Q in the first column is a per-request decoder offer, not mean Q/F. "
           "Native and Flow are shown only at their actual block-size endpoints.","",
           "TPF averages per-request final nonspecial N / all post-prefill F (including Gemma encoder). "
           "TE pools N / selected decoder rows reading history KV, not N / decoder-plus-encoder rows. "
           "Flow TE is omitted where this selection is not separately verified; it is not zero. "
           "Nominal IBS is a fixed window, not necessarily a per-forward hard cap. At 2B only "
           "the native fixed-window IBS run is shown; it saturates at a single physical block. "
           "Contrasts with native/IBS/Flow are descriptive on the same frozen cases, sometimes on different GPUs. "
           "Differences are not claims of statistical significance.",""]
    methods=("Request","IBS capped","IBS nominal","Native","Flow W1","Flow W2")
    for c in ("frozen90","original30","new60"):
        lines+=[f"## {c}","",f"| Model | Q | {' | '.join(methods)} |","|---|---:|"+"---:|"*len(methods)]
        for model,tiers in MODELS.items():
            for q in tiers:
                g={r["method"]:r for r in rows if (r["model"],r["cohort"],r["budget"])==(model,c,q)}
                cells=[]
                for k in methods:
                    if k not in g:
                        cells.append("pending" if k=="Request" else "-")
                        continue
                    r=g[k]; te=r["token_efficiency"]
                    cells.append(f'{r["correct"]}/{r["n"]} / {r["macro_TPF"]:.3f} / '
                                 +(f"{100*te:.2f}%" if te is not None else "n/a"))
                lines.append(f"| {model} | {q} | "+" | ".join(cells)+" |")
        lines+=[""]
    lines+=["[Budget curves with numeric readouts](frozen90.png). "
            "[Numeric data and source paths](metrics.csv). [Lower-offer receipt manifest](llada-lower.json). "
            "Regenerate this table from the repository root with `python -m "
            "experiments.t2t.serving_slots.fixed_request_comparison`. "
            "Refresh only complete LLaDA lower-offer cohorts on the receipt host with "
            "`llada_fixed_lower_cohorts.py --q4 Q4_ROOT --cross CROSS_ROOT --manifest MANIFEST`.",
            ""]
    return "\n".join(lines)


def draw(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from math import nan
    methods={"Request":("#146859","o"),"IBS capped":("#365da0","^"),"IBS nominal":("#8275a5","v"),
             "Native":("#393d43","s"),"Flow W1":("#b75827","+"),"Flow W2":("#aa3159","D")}
    fig,axes=plt.subplots(2,3,figsize=(16.8,9.8))
    fig.subplots_adjust(left=.055,right=.985,bottom=.23,top=.88,hspace=1.22,wspace=.22)
    handles={}
    for ai,(model,tiers) in enumerate(MODELS.items()):
        for col,(metric,title,scale) in enumerate((("correct","Correct / 90",1),("macro_TPF","Tokens / forward",1),
                                                   ("token_efficiency","Token efficiency (%)",100))):
            ax=axes[ai,col]
            ax.set(title=title,xticks=range(5),xticklabels=tiers,xlabel="Fixed decoder offer Q",xlim=(-.2,4.2))
            ax.grid(axis="y",alpha=.2);ax.spines[["right","top"]].set_visible(False)
            cells=[]
            for name,(color,marker) in methods.items():
                vals={r["budget"]:r[metric] for r in rows if
                      (r["cohort"],r["model"],r["method"])==("frozen90",model,name) and r[metric] is not None}
                cells.append([name,*["-" if q not in vals else
                             str(vals[q]) if metric=="correct" else f"{vals[q]*scale:.2f}" for q in tiers]])
                if not vals:continue
                x=list(range(5));y=[vals.get(q,nan)*scale for q in tiers]
                line,=ax.plot(x,y,color=color,marker=marker,linestyle="-" if name=="Request" else "--",
                              lw=2 if name=="Request" else 1.3,ms=6,alpha=.95)
                handles[name]=line
                if len(vals)==1:
                    k=tiers.index(next(iter(vals)));ax.hlines(y[k],k-.14,k+.14,linestyle="--",color=color)
            table=ax.table(cellText=cells,colLabels=["Q",*tiers],colWidths=[.31,.138,.138,.138,.138,.138],
                           cellLoc="center",bbox=[0,-.86,1,.59])
            table.auto_set_font_size(False);table.set_fontsize(7.1)
            for (i,j),cell in table.get_celld().items():
                cell.set_edgecolor("#dadfe1");cell.set_linewidth(.45)
                if i==0:cell.set_facecolor("#edf1f1")
                elif j==0:cell.get_text().set_color(list(methods.values())[i-1][0])
            if col==0:
                ax.set_ylabel("LLaDA 2.1" if model=="llada21" else "DiffusionGemma",weight="bold")
    fig.suptitle("Single-request execution at fixed Q  |  frozen 90",x=.05,ha="left",fontsize=18)
    fig.legend(handles.values(),handles.keys(),loc="upper center",bbox_to_anchor=(.52,.949),ncol=6,frameon=False)
    fig.text(.055,.015,"Baselines are dashed. Native/W1/W2 are measured endpoints only; dash in table = unavailable. "
             "Each point is a separate constant-budget run.",fontsize=9)
    fig.savefig(DEST/"frozen90.png",dpi=150,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    data=collect()
    write_csv(DEST/"metrics.csv",data)
    (DEST/"README.md").write_text(render(data))
    draw(data)

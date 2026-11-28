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
    for q in (4,8,16):
        name=f"llada-slot-allocation/c{q}-window-frozen90.json"
        if not (ROOT/name).is_file():
            raise ValueError(f"Missing sealed {name}")
        result=json.loads((ROOT/name).read_bytes())
        reg=result["registration"]
        if (reg["protocol"]!="llada-fixed-width-window-v1" or reg["q"]!=q or
                sorted(reg["order"])!=list(range(90)) or set(result["cohorts"])!=set(COHORTS)):
            raise ValueError("Not a complete fixed-window trial")
        for c,v in result["cohorts"].items():
            r=v["arms"]["window"]; baseline=rows["llada21",c,q,"IBS capped"]
            delta=v["window_minus_external_capped_IBS"]
            if (baseline["correct"]+delta["rescues"]-delta["losses"]!=r["correct"] or
                    not isclose(baseline["macro_TPF"]+delta["delta_macro_TPF"],r["macro_TPF"],abs_tol=1e-6) or
                    not isclose(r["TE"],r["N"]/r["Q"],abs_tol=1e-9)):
                raise ValueError("Window and paired IBS reference disagree")
            add("llada21",c,q,"Active window",v["pairs"],r["correct"],r["macro_TPF"],r["N"],r["F"],r["Q"],name)
    name="llada-paid-head/q64-agreement-frozen90.json"
    repeat=json.loads((ROOT/name).read_bytes())
    if (repeat["finished"]!=90 or repeat["registration"]["protocol"]!="llada-q64-paid-repeat-v1"
            or repeat["registration"]["q"]!=64 or set(repeat["cohorts"])!=set(COHORTS)):
        raise ValueError("Incomplete fixed Q64 repeat")
    for c,v in repeat["cohorts"].items():
        old=v["old"]; ref=rows["llada21",c,64,"Request"]; r=v["repeat"]
        if (v["original_reference_mismatches"] or
                any(old[k]!=ref[rk] for k,rk in (("N","N"),("F","F"),("Q","history_Q"),("correct","correct"))) or
                r["correct"]!=old["correct"]+v["rescued"]-v["lost"] or not isclose(r["TE"],r["N"]/r["Q"])):
            raise ValueError("Q64 repeat control/bill mismatch")
        add("llada21",c,64,"Paid repeat",v["pairs"],r["correct"],r["macro_TPF"],r["N"],r["F"],r["Q"],name)
    name="llada-slot-allocation/c8-paid-frozen90.json"
    paid=json.loads((ROOT/name).read_bytes())
    if paid["registration"]["protocol"]!="llada-window-paid-mask-v1" or paid["registration"]["q"]!=8:
        raise ValueError("Wrong H0 paid sampling cohort")
    for c,v in paid["cohorts"].items():
        if c not in COHORTS: continue
        r=v["arms"]["paid"]
        if (r["failed"] or not isclose(r["TE"],r["N"]/r["Q"])
                or r["paid_waiting_rows"]<r["paid_waiting_admits"]):
            raise ValueError("Invalid H0 paid sampling accounting")
        add("llada21",c,8,"Paid MASK",v["pairs"],r["correct"],r["macro_TPF"],r["N"],r["F"],r["Q"],name)
    name="llada-window-sample-q16/summary.json"
    paid=json.loads((ROOT/name).read_bytes())
    if paid["q"]!=16 or paid["protocol"]!="llada-window-paid-mask-v1":
        raise ValueError("Wrong A4 paid sampling cohort")
    for raw,v in paid["cohorts"].items():
        c=NAMES.get(raw,raw)
        if c not in COHORTS: continue
        part=v["per_index"]
        start=30 if c=="new60" else 0
        if ({p["index"] for p in part}!=set(range(start,start+COHORTS[c])) or len(part)!=COHORTS[c]
                or v["control_replay"]!=len(part) or v["first_failures"] or v["control_replay_failures"]):
            raise ValueError("Unpaired or unsealed A4 paid sampling study")
        ref=rows["llada21",c,16,"Active window"]
        for a in ("old","new"):
            values={k:sum(p[a+"_"+k] for p in part) for k in ("N","F","Q","correct")}
            macro=sum(p[a+"_N"]/p[a+"_F"] for p in part)/len(part)
            if (any(v[a+"_"+k]!=x for k,x in values.items())
                    or not isclose(macro,v[a+"_macro_TPF"]) or v[a+"_failures"]
                    or not isclose(v[a+"_pooled_TE"],values["N"]/values["Q"])):
                raise ValueError("Paid sampling row sums disagree")
            if a=="old":
                if any(x!=ref["history_Q" if k=="Q" else k] for k,x in values.items()):
                    raise ValueError("A4 replay does not reproduce fixed A5 reference")
            else:
                add("llada21",c,16,"Paid MASK",len(part),values["correct"],macro,values["N"],values["F"],values["Q"],name)
    for q in (32,64):
        name=f"llada-paid-head/q{q}-frontier-frozen90.json"
        full=json.loads((ROOT/name).read_bytes())
        if (full["finished"]!=90 or full["registration"]["q"]!=q or
                full["registration"].get("scope","all")!="all"):
            raise ValueError("Expected a complete all-state frontier comparison")
        for c,v in full["cohorts"].items():
            ref=rows["llada21",c,q,"Request" if q==32 else "Paid repeat"]
            old,r=v["old"],v["frontier"]
            if (any(old[k]!=ref[rk] for k,rk in (("correct","correct"),("F","F"),("N","N"),("Q","history_Q"))) or
                    old["correct"]+v["rescued"]-v["lost"]!=r["correct"] or r["failed"] or
                    not isclose(r["TE"],r["N"]/r["Q"])):
                raise ValueError("Unmatched frontier control or invalid bill")
            add("llada21",c,q,"Paid frontier",v["n"],r["correct"],r["macro_TPF"],r["N"],r["F"],r["Q"],name)
    # The handle dispatches to these fixed paths by construction. The switch
    # receipts qualify that dispatch, not the quality of a mixed-offer path.
    qual=json.loads((ROOT/"strict-handle/llada-frontier32/report.json").read_bytes())
    arms={4:"Active window",8:"Active window",16:"Paid MASK",32:"Paid frontier",64:"Paid repeat"}
    if (qual["profile"]!="quarter_window_repeat64_paid16_frontier32" or qual["indices"]!=[0,10,20]
            or any(qual["arms"][f"fixed{q}"]["exact_replays"]!=3 for q in arms)
            or not qual["arms"]["switch25"]["policy_checks"].get("frontier32_rows")
            or len(qual["arms"]["switch25"]["directed_pairs"])!=25):
        raise ValueError("Live profile has no fixed-path and switching qualification")
    for model,tiers in MODELS.items():
        for q in tiers:
            for c in COHORTS:
                source=rows[model,c,q,arms[q] if model=="llada21" else "Request"]
                rows[model,c,q,"Live profile"]={**source,"method":"Live profile"}
    return list(rows.values())


def compare_live(rows):
    """Deltas against the actual comparators, not a per-metric oracle envelope."""
    at={(r["model"],r["cohort"],r["budget"],r["method"]):r for r in rows}
    lines=["# Fixed-budget decisions against baselines", "",
           "Each entry compares the published live request to one measured baseline on the same frozen tasks. "
           "Use hard-capped IBS below 2B, plus native and Flow at their actual block-size endpoints. "
           "Uncapped nominal windows are shown separately in the [full table](README.md).",
           "", "The deltas are live minus baseline, never vs a composite of different baselines. "
           "Correct deltas are whole questions (not percent); TE deltas are percentage points. "
           "'Both at least as high' is a point estimate, not a noninferiority test. "
           "The original and additional tasks are disjoint, not two new independent replications.",
           "", "| Model | Q | Baseline | Correct delta (30 / 60 / 90) | Macro TPF delta (30 / 60 / 90) | TE delta (90, pp) | Both at least as high (30 / 60 / 90) |",
           "|---|---:|---|---:|---:|---:|---|"]
    for model,tiers in MODELS.items():
        for q in tiers:
            methods=(["IBS capped"] if q<tiers[-1] else ["Flow W2"])
            if q==tiers[-2]:
                methods+=["Native","Flow W1"]
            for name in methods:
                score=[]; tpf=[]; wins=[]
                for c in ("original30","new60","frozen90"):
                    live=at[model,c,q,"Live profile"]; ref=at[model,c,q,name]
                    score.append(live["correct"]-ref["correct"])
                    tpf.append(live["macro_TPF"]-ref["macro_TPF"])
                    wins.append("yes" if min(score[-1],tpf[-1])>=0 else "no")
                # The sealed Gemma dense-stop source publishes four decimal places.
                tol=1.0e-4 if model=="gemma" and q==256 else 1e-6
                if score[0]+score[1]!=score[2] or not isclose(tpf[0]/3+2*tpf[1]/3,tpf[2],abs_tol=tol):
                    raise ValueError("The subgroup deltas do not reproduce the full cohort")
                eff=(f'{100*(live["token_efficiency"]-ref["token_efficiency"]):+.2f}'
                     if ref["token_efficiency"] is not None else "n/a")
                triple=" / ".join
                lines.append(f"| {model} | {q} | {name} | {triple(f'{v:+d}' for v in score)} | "
                             f"{triple(f'{v:+.3f}' for v in tpf)} | {eff} | {triple(wins)} |")
    return "\n".join(lines+["","Same offer does not mean the same number of rows over the entire request. "
                            "Cross-device results are descriptive. See [metrics.csv](metrics.csv) for "
                            "N, F, selected Q (when verified), source receipts, and the exact series.",""])


MAIN=("Live profile","IBS capped","IBS nominal","Native","Flow W1","Flow W2")
ABLATIONS=("Live profile","Request","Active window","Paid MASK","Paid repeat","Paid frontier")


def render(rows, *, ablations=False):
    methods=ABLATIONS if ablations else MAIN
    lines=(["# LLaDA fixed-offer ablations","",
            "The earlier `Request` is quarter_edit_warm. `Active window` changes selected head rows; "
            "`Paid MASK` samples projected waiting MASKs; `Paid repeat` accepts a repeated measured proposal; "
            "`Paid frontier` accepts a high-confidence contiguous run. The latter dropped Q64 accuracy by seven "
            "against paid repeat. The paired H800 Q8 control is 71/90, not the A800 window row (72/90). "
            "Compare subgroup and same-device receipts before attributing differences. "
            "[Ablation figure](ablations.png). [Main comparison](README.md).",""] if ablations else
           ["# Fixed offers: one executable profile versus measured baselines","",
           "Every point is an independent constant-budget request from prompt to final output. "
           "Only the published profile is a candidate here. No per-metric winner selection. "
           "For each model, Native/W1 are measured at B and W2 at 2B only. "
           "[Main figure](frozen90.png) | [Differences against each baseline](tradeoffs.md) | "
           "[Experimental variants (separate)](ablations.md).",""])
    lines += [
           "**Live profile** = `quarter_window_repeat64_paid16_frontier32` for LLaDA, "
           "`dense_stop_joint_verified_eos` for Gemma. Values are from the independent fixed90 "
           "runs implementing each exact path, not 90 reruns through the switching API. "
           "LLaDA uses Active window at Q4/8, Paid MASK at Q16, Paid frontier at Q32, "
           "and Paid repeat at Q64. [Switching qualification](../strict-handle/README.md).","",
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
    for c in ("frozen90","original30","new60"):
        lines+=[f"## {c}","",f"| Model | Q | {' | '.join(methods)} |","|---|---:|"+"---:|"*len(methods)]
        for model,tiers in MODELS.items():
            if ablations and model!="llada21":continue
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
    lines+=["[Same-offer deltas vs each baseline, including original30 and new60](tradeoffs.md). "
            "[Budget curves with numeric readouts](frozen90.png). "
            "[Numeric data and source paths](metrics.csv). [Lower-offer receipt manifest](llada-lower.json). "
            "Regenerate this table from the repository root with `python -m "
            "experiments.t2t.serving_slots.fixed_request_comparison`. "
            "Refresh only complete LLaDA lower-offer cohorts on the receipt host with "
            "`llada_fixed_lower_cohorts.py --q4 Q4_ROOT --cross CROSS_ROOT --manifest MANIFEST`.",
            ""]
    return "\n".join(lines)


def draw(rows, *, ablations=False, shared_output=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from math import nan
    styles={"Live profile":("#146859","o"),"Request":("#879288","."),"Paid MASK":("#925008","X"),"Paid frontier":("#006697","*"),
             "Paid repeat":("#b33576","d"),"Active window":("#475477","P"),
             "IBS capped":("#365da0","^"),"IBS nominal":("#8275a5","v"),
             "Native":("#393d43","s"),"Flow W1":("#b75827","+"),"Flow W2":("#aa3159","D")}
    if shared_output is not None and ablations:
        raise ValueError("The common planner is not a Live-profile ablation")
    main_candidate="Live profile" if shared_output is None else "Shared planner"
    if shared_output is not None:
        styles[main_candidate]=("#0f6958","o")
        styles["Live profile"]=("#895399","x")
    methods=ABLATIONS if ablations else MAIN if shared_output is None else (main_candidate, *MAIN)
    models=("llada21",) if ablations else tuple(MODELS)
    fig,axes=plt.subplots(len(models),3,figsize=(16.8,5.9 if ablations else 10.2),squeeze=False)
    fig.subplots_adjust(left=.055,right=.985,bottom=.37 if ablations else .215,top=.77 if ablations else .875,hspace=1.27,wspace=.22)
    handles={}
    for ai,model in enumerate(models):
        tiers=MODELS[model]
        for col,(metric,title,scale) in enumerate((("correct","Correct / 90",1),("macro_TPF","Tokens / forward",1),
                                                   ("token_efficiency","Token efficiency (%)",100))):
            ax=axes[ai,col]
            ax.set(title=title,xticks=range(5),xticklabels=tiers,xlabel="Fixed decoder offer Q",xlim=(-.2,4.2))
            ax.grid(axis="y",alpha=.2);ax.spines[["right","top"]].set_visible(False)
            cells=[]; colors=[]
            for name in methods:
                color,marker=styles[name]
                colors.append(color)
                vals={r["budget"]:r[metric] for r in rows if
                      (r["cohort"],r["model"],r["method"])==("frozen90",model,name) and r[metric] is not None}
                cells.append([name,*["-" if q not in vals else
                             str(vals[q]) if metric=="correct" else f"{vals[q]*scale:.2f}" for q in tiers]])
                if not vals:continue
                x=list(range(5));y=[vals.get(q,nan)*scale for q in tiers]
                candidate=name==main_candidate or ablations
                line,=ax.plot(x,y,color=color,marker=marker,linestyle="-" if candidate else "--",
                              lw=2 if candidate else 1.3,ms=6,alpha=.95)
                handles[name]=line
                if len(vals)==1:
                    k=tiers.index(next(iter(vals)));ax.hlines(y[k],k-.14,k+.14,linestyle="-" if candidate else "--",color=color)
            table=ax.table(cellText=cells,colLabels=["Q",*tiers],colWidths=[.31,.138,.138,.138,.138,.138],
                           cellLoc="center",bbox=[0,-.86,1,.59])
            table.auto_set_font_size(False);table.set_fontsize(7.1)
            for (i,j),cell in table.get_celld().items():
                cell.set_edgecolor("#dadfe1");cell.set_linewidth(.45)
                if i==0:cell.set_facecolor("#edf1f1")
                elif j==0:cell.get_text().set_color(colors[i-1])
            if col==0:
                ax.set_ylabel("LLaDA 2.1" if model=="llada21" else "DiffusionGemma",weight="bold")
    title=("Shared planner vs measured baselines" if shared_output is not None else
           "LLaDA 2.1 ablations" if ablations else "One executable profile vs measured baselines")
    fig.suptitle(title+
                  "  |  fixed Q, frozen 90",x=.05,ha="left",fontsize=18)
    fig.legend(handles.values(),handles.keys(),loc="upper center",bbox_to_anchor=(.52,.933 if ablations else .95),ncol=len(methods),frameon=False)
    fig.text(.055,.015,("Each experimental arm has its own series; never combine the best points across rows."
             if ablations else "Dashed = baseline. Native/W1/W2 are actual endpoints, not constant reference lines. "
             "Nominal IBS is not necessarily capped. Missing history-row counts are not zero."),fontsize=9)
    fig.savefig(shared_output if shared_output is not None else DEST/("ablations.png" if ablations else "frozen90.png"),
                dpi=150,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    data=collect()
    write_csv(DEST/"metrics.csv",data)
    (DEST/"README.md").write_text(render(data))
    (DEST/"tradeoffs.md").write_text(compare_live(data))
    (DEST/"ablations.md").write_text(render(data,ablations=True))
    draw(data)
    draw(data,ablations=True)

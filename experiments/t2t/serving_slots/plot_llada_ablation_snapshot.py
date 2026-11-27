"""Extract matched aggregate-only LLaDA snapshots, or redraw them from CSV."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
TIERS = (4, 8, 16, 32, 64)
ARMS = ("00", "10", "11")
STYLE = {
    "00": ("#007e78", "o", "-", "00  Within block"),
    "10": ("#b34f12", "s", "-", "10  Cross-block"),
    "11": ("#855297", "^", "-", "11  Cross-block + full build"),
    "old": ("#4478ac", "D", "--", "Earlier kernel (nominal Q)"),
    "native": ("#242a2d", "*", "--", "Native Q32"),
    "flow": ("#778026", "P", "--", "FLowBlock W1"),
}
FIELDS = ("cohort", "n", "gsm8k", "math500", "humaneval", "method", "q", "correct",
          "N", "F", "Q", "macro_tpf", "eff_pct", "peak_decoder_q", "over_offer_F")


def extract(study, dest):
    refs = {}
    for name, file, side in (
        ("native", "native-llada21-paired.csv", "arm"),
        ("flow", "llada-registered-wave-scores/llada21-native32-vs-flow-w1-paired.csv", "baseline"),
        ("old16", "llada-native32-comparison/llada21-native32-vs-q16-paired.csv", "arm"),
        ("old32", "llada-native32-comparison/llada21-native32-vs-q32-paired.csv", "arm"),
    ):
        items = {}
        with (study.parent/file).open() as f:
            for r in csv.DictReader(f):
                i = int(r["index"])
                rec = dict(dataset=r["dataset"], task_id=r["task_id"], seed=int(r["seed"]),
                           correct=int(r[side+"_correct"]), TPF=float(r[side+"_tpf"]),
                           **{k:int(r[side+"_"+k]) for k in ("N","F","Q")})
                if i in items and items[i] != rec:
                    raise ValueError(f"Conflicting reference for {name} #{i}")
                items[i] = rec
        refs[name] = items
    # Freeze the filename inventory before opening any rows of an active study.
    inventory = {p.name for p in study.glob("*.json.gz")}
    ids = [i for i in range(90) if all(f"{i:02}-q{q}-{a}.json.gz" in inventory
                                        for q in TIERS for a in ARMS)]
    if not set(range(30)).issubset(ids):
        raise ValueError("Original 30 not complete")
    inputs = {}
    arms = {}
    for q in TIERS:
        for a in ARMS:
            subset = {}
            for i in ids:
                with gzip.open(study/f"{i:02}-q{q}-{a}.json.gz") as f:
                    r = json.load(f)
                assert (r["index"],r["q"],r["arm"]) == (i,q,a)
                assert inputs.setdefault(i,r["input_sha256"]) == r["input_sha256"]
                assert all(all(r[k] == ref[i][k] for k in ("dataset","task_id","seed")) for ref in refs.values())
                subset[i] = {k:r[k] for k in ("N","F","Q","TPF","correct","max_decoder_Q","over_offer_F")}
            arms[a,q] = subset
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        wr.writeheader()
        for cohort, use in (("matched",ids), ("original30",list(range(30)))):
            datasets = Counter(refs["native"][i]["dataset"] for i in use)
            for (method,q), group in {**arms, ("native",32):refs["native"], ("flow",32):refs["flow"],
                                       ("old",16):refs["old16"], ("old",32):refs["old32"]}.items():
                rs = [group[i] for i in use]
                n,fwd,rows = (sum(r[k] for r in rs) for k in ("N","F","Q"))
                wr.writerow(dict(cohort=cohort,n=len(use),**datasets,method=method,q=q,
                    correct=sum(r["correct"] for r in rs),N=n,F=fwd,Q=rows,macro_tpf=mean(r["TPF"] for r in rs),
                    eff_pct=100*n/rows,peak_decoder_q=max(r.get("max_decoder_Q",0) for r in rs) or "",
                    over_offer_F=sum(r["over_offer_F"] for r in rs) if method in ARMS else ""))
    print("snapshot cases", len(inventory), "common questions", len(ids), flush=True)


def render(path, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    with path.open(newline="") as f:
        all_rows = list(csv.DictReader(f))
    for cohort in ("matched", "original30"):
        rows = [r for r in all_rows if r["cohort"] == cohort]
        data = {(r["method"],int(r["q"])):r for r in rows}
        n = int(rows[0]["n"])
        fig = plt.figure(figsize=(14.1,8.3), facecolor="white")
        top = fig.add_gridspec(1,3,left=.054,right=.972,top=.774,bottom=.435,wspace=.23)
        title = "Matched snapshot" if cohort == "matched" and n != 90 else "Frozen cohort"
        fig.text(.045,.94,f"LLaDA 2.1   |   {title}: {n} questions",fontsize=24,weight="bold",color="#20292d")
        counts = "   /   ".join(k+" "+rows[0][k] for k in ("gsm8k","math500","humaneval"))
        fig.text(.046,.901,f"Same questions and seeds across all budgets and methods    |    {counts}",
                 fontsize=11.3,color="#505b62")
        handles = [Line2D([],[],color=c,marker=mk,ls=ls,lw=1.8,label=label)
                   for c,mk,ls,label in STYLE.values()]
        fig.legend(handles=handles,ncol=6,loc="upper left",bbox_to_anchor=(.04,.875),
                   frameon=False,fontsize=9.5,columnspacing=1.55)
        for j,(field,heading) in enumerate((("correct",f"Correct answers (out of {n})"),
                                            ("macro_tpf","Tokens per forward"),("eff_pct","Token efficiency (%)"))):
            ax = fig.add_subplot(top[0,j])
            for m,(color,mk,ls,_) in STYLE.items():
                xs = sorted(q for method,q in data if method==m)
                ys = [float(data[m,q][field]) for q in xs]
                if m in ("native","flow"):
                    ax.axhline(ys[0],color=color,ls="--",lw=1.25,alpha=.85)
                ax.plot(xs,ys,marker=mk,color=color,ls=ls,mfc=color if m in ARMS else "white",
                        lw=2,ms=7 if m!="native" else 11,mew=1.4)
            ax.set_xscale("log",base=2)
            ax.set_xlim(3.45,74)
            ax.set_xticks(TIERS,labels=[str(t) for t in TIERS])
            ax.set_xlabel("Fixed Q budget")
            ax.set_title(heading,loc="left",weight="bold",fontsize=13,pad=9)
            ax.grid(axis="y",color="#e1e4e5",lw=.8)
            ax.set_axisbelow(True)
            ax.spines[["right","top"]].set_visible(False)
            ax.spines[["bottom","left"]].set_color("#afb8be")
            ax.margins(y=.12)
            if j==0: ax.yaxis.set_major_locator(MaxNLocator(integer=True,nbins=7))
        fig.text(.045,.354,"Measured values",weight="bold",fontsize=13)
        fig.text(.197,.354,f"Each cell: correct / {n}   |   TPF   |   token efficiency",fontsize=10,color="#465058")
        fmt = lambda r: f"{r['correct']}/{n}   |   {float(r['macro_tpf']):.3f}   |   {float(r['eff_pct']):.2f}%"
        cells = [[f"Q{q}"]+[fmt(data[m,q]) if (m,q) in data else "-" for m in STYLE] for q in TIERS]
        ax=fig.add_axes([.044,.128,.923,.20])
        ax.axis("off")
        tab=ax.table(cellText=cells,colLabels=["Budget","00","10","11","Earlier kernel","Native","FLowBlock W1"],
                     colWidths=[.06]+[.1567]*6,cellLoc="center",bbox=[0,0,1,1])
        tab.auto_set_font_size(False)
        tab.set_fontsize(8.7)
        for (i,j),cell in tab.get_celld().items():
            cell.set_edgecolor("white")
            cell.set_facecolor("#e6edef" if i==0 else "#f3f6f7" if i%2 else "white")
            if i==0 or j==0:cell.get_text().set_weight("bold")
        fig.text(.045,.084,"Dashed horizontals: Native and FLowBlock W1 are measured at Q32 only. "
                 "Q64 = two full blocks: all three current policies coincide.",fontsize=9.4,color="#45515a")
        fig.text(.045,.051,"11 can use 32 real decoder rows for a child-block build at Q4 / Q8 / Q16. "
                 "Earlier-kernel points are nominal budgets, not strict-cap controls.",fontsize=9.4,color="#45515a")
        out.mkdir(parents=True,exist_ok=True)
        fig.savefig(out/f"llada21-fixed-q-{cohort}{n if cohort=='matched' else ''}.png",dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",type=Path,help="Optional: read GPU receipts to update CSV")
    p.add_argument("--csv",type=Path,required=True)
    p.add_argument("--out",type=Path,default=HERE/"figures")
    args=p.parse_args()
    if args.study: extract(args.study,args.csv)
    render(args.csv,args.out)

"""Publish the measured fixed profile against same-cohort external controls."""
import csv
from pathlib import Path

from .factor_tables import write_csv

ROOT = Path(__file__).resolve().parent / "controls"
FACTORS = ROOT / "gemma-factors"
TIERS = (32,64,128,256,512)
COHORTS = (("original30","original30",30),("new60","added60",60),("frozen90","all90",90))


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def collect():
    studies = {k:read(FACTORS/k/"paired.csv") for k in ("close","profile-close","capacity")}
    baselines = read(ROOT/"block-history90-matrix.csv")
    rows = []
    for cohort, external, n in COHORTS:
        for c in TIERS:
            file = "profile-close" if c in (128,512) else "close"
            arms = [(file,"fixed90_fused")]
            if c==256:
                arms.append(("capacity","full-head variant"))
            for study, label in arms:
                matches = [r for r in studies[study] if r["cohort"]==cohort and
                           int(r["budget"])==c and r["side"]=="trial"]
                if not matches and study=="capacity":
                    continue
                r, = matches
                if int(r["n"])!=n or int(r["peak_decoder_Q"])>c or int(r["history_Q"])!=int(r["decoder_Q"]):
                    raise ValueError(f"unqualified {study} {cohort} {c}")
                rows.append(dict(cohort=cohort,method=label,budget=c,n=n,correct=int(r["correct"]),
                    macro_TPF=float(r["macro_TPF"]),token_efficiency=float(r["token_efficiency"]),
                    avg_decoder_Q=int(r["decoder_Q"])/int(r["decoder_F"]),peak_decoder_Q=int(r["peak_decoder_Q"]),
                    F=int(r["F"]),N=int(r["N"]),history_Q=int(r["history_Q"]),physical_Q=int(r["physical_Q"]),
                    source=f"gemma-factors/{study}/paired.csv"))
        for method, setting, c, label in [
            *(("ibs",q,q,"Optimus IBS") for q in TIERS), ("native",256,256,"Native / Flow W1"),
            ("flow",2,512,"Flow W2"),
        ]:
            r, = [v for v in baselines if v["model"]=="gemma" and v["cohort"]==external and
                  v["method"]==method and int(v["setting"])==setting]
            if int(r["requests"])!=n:
                raise ValueError("incomplete external baseline")
            rows.append(dict(cohort=cohort,method=label,budget=c,n=n,correct=int(r["correct"]),
                macro_TPF=float(r["macro_tpf"]),token_efficiency=float(r["token_efficiency"]) if r["token_efficiency"] else None,
                avg_decoder_Q=None,peak_decoder_Q=None,F=int(r["F"]),N=int(r["N"]),
                history_Q=int(r["prefix_kv_Q"]) if r["prefix_kv_Q"] else None,physical_Q=int(r["physical_Q"]),
                source="block-history90-matrix.csv:"+r["source"]))
    return rows


def plot(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cohort, = {r["cohort"] for r in rows}
    n, = {r["n"] for r in rows}
    palette={"fixed90_fused":"#087469","Optimus IBS":"#6343a0","Native / Flow W1":"#596779",
             "Flow W2":"#a65216","full-head variant":"#b02c53"}
    fig, axes = plt.subplots(1,3,figsize=(15,5.7))
    handles = {}
    for ax, (col,title,scale) in zip(axes, (("correct",f"Correct / {n}",1),
                ("macro_TPF","Tokens per forward",1),("token_efficiency","Token efficiency (%)",100))):
        for name in palette:
            group = sorted((v for v in rows if v["method"]==name and v[col] is not None),
                           key=lambda v:v["budget"])
            if not group:
                continue
            x=[TIERS.index(v["budget"]) for v in group]
            y=[v[col]*scale for v in group]
            color=palette[name]
            if name=="Native / Flow W1":
                h = ax.axhline(y[0],linestyle="--",color=color,lw=1.25)
            elif name=="Flow W2":
                h,=ax.plot([3.75,4.17],y*2,"--",color=color,lw=2)
                ax.scatter(x,y,marker="s",facecolors="none",edgecolors=color,s=40)
            else:
                h,=ax.plot(x,y,marker={"fixed90_fused":"o","Optimus IBS":"^","full-head variant":"D"}[name],
                           ls="--" if name=="Optimus IBS" else "-",color=color,lw=2,ms=6)
            handles[name]=h
        ax.set(xticks=range(len(TIERS)),xticklabels=TIERS,xlabel="Offered decoder Q",title=title,xlim=(-.28,4.28))
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.22)
    fig.subplots_adjust(left=.05,right=.985,wspace=.21,top=.81,bottom=.43)
    fig.suptitle(f"DiffusionGemma  |  {cohort}  |  fixed offer for the entire request",fontsize=16,x=.05,ha="left")
    for j, ax in enumerate(axes):
        entries = []
        for name in palette:
            group = {v["budget"]:v for v in rows if v["method"]==name}
            if not group: continue
            col=("correct","macro_TPF","token_efficiency")[j]
            def val(c):
                z=group.get(c,{}).get(col)
                return "-" if z is None else str(z) if j==0 else f"{(100 if j==2 else 1)*z:.2f}"
            entries.append((name,[val(c) for c in TIERS]))
        table=ax.table(cellText=[[name,*v] for name,v in entries],colLabels=["Q",*TIERS],
                       colWidths=[.35,.13,.13,.13,.13,.13],cellLoc="center",bbox=[0,-.94,1,.63])
        table.auto_set_font_size(False)
        table.set_fontsize(7.6)
        for (i,k), cell in table.get_celld().items():
            cell.set_edgecolor("#e3e5e8")
            if i==0: cell.set_facecolor("#f2f3f5")
            elif k==0: cell.get_text().set_color(palette[entries[i-1][0]])
    fig.legend(handles.values(),handles.keys(),loc="upper center",bbox_to_anchor=(.52,.94),
               ncol=len(handles),frameon=False,fontsize=8)
    fig.text(.05,.02,"Dashed: external controls. The native reference runs at Q256 only. IBS 512 saturates at one block; a dash in the table means unavailable.",fontsize=9)
    fig.savefig(path,dpi=160,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    dst=FACTORS/"fixed-profile"
    dst.mkdir(exist_ok=True)
    data=collect()
    write_csv(dst/"metrics.csv",data)
    for cohort,_,_ in COHORTS:
        plot(dst/f"{cohort}.png",[r for r in data if r["cohort"]==cohort])

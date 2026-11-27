"""Publish the measured fixed profile against same-cohort external controls."""
import csv
import json
from pathlib import Path

from .factor_tables import write_csv

ROOT = Path(__file__).resolve().parent / "controls"
FACTORS = ROOT / "gemma-factors"
TIERS = (32,64,128,256,512)
COHORTS = (("original30","original30",30),("new60","added60",60),("frozen90","all90",90))
ROLES = ("mandatory","prefix","unresolved","clean-other","verify","rebuild")


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def collect():
    studies = {k:read(FACTORS/k/"paired.csv") for k in ("close","profile-close","capacity","density","promotion")}
    baselines = read(ROOT/"block-history90-matrix.csv")
    stops = {r['cohort']:r for r in map(json.loads,(FACTORS/"fullstop"/"dense-summary.jsonl").read_text().splitlines())}
    joint_cohorts={r["cohort"]:r for r in read(FACTORS/"joint-verify"/"cohorts.csv")}
    joint_pairs=read(FACTORS/"joint-verify"/"pairs.csv")
    rows = []
    for cohort, external, n in COHORTS:
        for c in TIERS:
            file = "profile-close" if c in (128,512) else "close"
            arms = [(file,"fixed90_fused",None)]
            if c==512:
                arms.append(("promotion","sample on rebuild","C512-promotion1"))
            if c==256:
                arms.extend([("capacity","full-head variant",None),
                             ("density","dense at high MASK","C256-density-high"),
                             ("density","dense at low MASK","C256-density-low")])
            for study, label, arm in arms:
                matches = [r for r in studies[study] if r["cohort"]==cohort and
                           int(r["budget"])==c and r["side"]=="trial" and (arm is None or r["arm"]==arm)]
                if not matches and study=="capacity":
                    continue
                r, = matches
                if int(r["n"])!=n or int(r["peak_decoder_Q"])>c or int(r["history_Q"])!=int(r["decoder_Q"]):
                    raise ValueError(f"unqualified {study} {cohort} {c}")
                if study=="promotion":
                    ref, = [v for v in studies[study] if v['cohort']==cohort and v['side']=='reference']
                    if (any(json.loads(r[k]) for k in ("changed_output_indices","changed_encoder_work_indices",
                            "wrong_F_delta_indices","wrong_Q_delta_indices"))
                            or int(ref['F'])-int(r['F'])!=int(r['promotion_F'])
                            or int(ref['decoder_Q'])-int(r['decoder_Q'])!=256*int(r['promotion_F'])):
                        raise ValueError("promotion is not an exact work reduction")
                roles = {f"Q_{k}":int(r[f"Q_{k}"]) for k in ROLES}
                if sum(roles.values())!=int(r["decoder_Q"]):
                    raise ValueError(f"unbalanced decoder ledger at {study} {cohort} {c}")
                rows.append(dict(cohort=cohort,method=label,budget=c,n=n,correct=int(r["correct"]),
                    macro_TPF=float(r["macro_TPF"]),token_efficiency=float(r["token_efficiency"]),
                    avg_decoder_Q=int(r["decoder_Q"])/int(r["decoder_F"]),peak_decoder_Q=int(r["peak_decoder_Q"]),
                    F=int(r["F"]),N=int(r["N"]),history_Q=int(r["history_Q"]),physical_Q=int(r["physical_Q"]),
                    source=f"gemma-factors/{study}/paired.csv",**roles))
        result=stops[cohort]
        stop=result["on"]
        if (result["tier"]!=256 or result["matched"]!=n or stop["scored"]!=n or result["prior_control_different"]
                or stop['ticks']*256!=stop['decoder_Q'] or sum(stop['roles'].values())!=stop['decoder_Q']):
            raise ValueError("Unqualified dense head stop")
        rows.append(dict(cohort=cohort,method="dense + native stop",budget=256,n=n,correct=stop["correct"],
            macro_TPF=stop["macro_TPF"],token_efficiency=stop["token_efficiency"],avg_decoder_Q=256,peak_decoder_Q=256,
            F=stop["F"],N=stop["N"],history_Q=stop["decoder_Q"],physical_Q=stop["decoder_Q"]+stop["encoder_Q"],
            source="gemma-factors/fullstop/dense-summary.jsonl",**{f"Q_{k}":stop["roles"][k] for k in ROLES}))
        joint=joint_cohorts[cohort]
        idx=set(range(30) if cohort=="original30" else range(30,90) if cohort=="new60" else range(90))
        pairs=[r for r in joint_pairs if int(r["index"]) in idx]
        reference,=(r for r in rows if r["cohort"]==cohort and r["method"]=="sample on rebuild")
        if (len(pairs)!=n or {int(r["index"]) for r in pairs}!=idx or int(joint["matched"])!=n or
                any(r["same_output"]!="True" or r["history_exact"]!="1" or int(r["F_residual"])
                    or int(r["decoder_Q_residual"]) or int(r["on_max_decoder_Q"])>512 for r in pairs) or
                any(int(joint["off_"+key])!=reference[col] for key,col in (
                    ("N","N"),("F","F"),("correct","correct"),("decoder_Q","history_Q"))) or
                int(joint["on_F"])+int(joint["on_events"])!=reference["F"] or
                int(joint["on_decoder_Q"])+int(joint["on_head_dirty_Q"])!=reference["history_Q"]):
            raise ValueError("Incomplete or unbalanced joint verification")
        decoder=int(joint["on_decoder_Q"])
        rows.append(dict(cohort=cohort,method="joint verify",budget=512,n=n,correct=int(joint["on_correct"]),
            macro_TPF=float(joint["on_macro_TPF"]),token_efficiency=int(joint["on_N"])/decoder,
            avg_decoder_Q=decoder/int(joint["on_decoder_F"]),peak_decoder_Q=max(int(r["on_max_decoder_Q"]) for r in pairs),
            F=int(joint["on_F"]),N=int(joint["on_N"]),history_Q=decoder,
            physical_Q=decoder+int(joint["on_encoder_Q"]),source="gemma-factors/joint-verify/cohorts.csv"))
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


def public_profile(rows):
    """Whole-request endpoints of one runtime profile, not maxima per metric."""
    arm={32:"fixed90_fused",64:"fixed90_fused",128:"fixed90_fused",256:"dense + native stop",512:"joint verify"}
    return [dict(r,method="Request") if arm[r["budget"]]==r["method"] else r for r in rows
            if arm[r["budget"]]==r["method"] or r["method"] in ("Native / Flow W1","Optimus IBS","Flow W2")]


def plot(path, rows, compact=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cohort, = {r["cohort"] for r in rows}
    n, = {r["n"] for r in rows}
    palette={"fixed90_fused":"#087469","Optimus IBS":"#6343a0","Native / Flow W1":"#596779",
             "Flow W2":"#a65216","full-head variant":"#b02c53",
             "dense at high MASK":"#1964ae","dense at low MASK":"#78620d","dense + native stop":"#191e24",
             "sample on rebuild":"#1670a0","joint verify":"#ac3860","Request":"#087469"}
    fig, axes = plt.subplots(1,3,figsize=(16,6.4))
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
                h,=ax.plot(x,y,marker={"fixed90_fused":"o","Optimus IBS":"^","full-head variant":"D",
                                       "dense at high MASK":"+","dense at low MASK":"x","dense + native stop":"*",
                                       "sample on rebuild":"P","joint verify":"X","Request":"o"}[name],
                           ls="--" if name=="Optimus IBS" else "-",color=color,lw=2,ms=8)
            handles[name]=h
        ax.set(xticks=range(len(TIERS)),xticklabels=TIERS,xlabel="Offered decoder Q",title=title,xlim=(-.28,4.28))
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.22)
    fig.subplots_adjust(left=.05,right=.985,wspace=.21,top=.81,bottom=.44)
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
                       colWidths=[.35,.13,.13,.13,.13,.13],cellLoc="center",bbox=[0,-1.02,1,.74])
        table.auto_set_font_size(False)
        table.set_fontsize(7.6)
        for (i,k), cell in table.get_celld().items():
            cell.set_edgecolor("#e3e5e8")
            if i==0: cell.set_facecolor("#f2f3f5")
            elif k==0: cell.get_text().set_color(palette[entries[i-1][0]])
    fig.legend(handles.values(),handles.keys(),loc="upper center",bbox_to_anchor=(.52,.94),
               ncol=min(6,len(handles)),frameon=False,fontsize=8)
    note=("Request: strict fixed decoder offer, sparse through 128, dense stop at 256, two-block joint verification at 512."
          if compact else "High/low MASK: dense head when unresolved count is at least/below 128.")
    fig.text(.05,.02,note+" Native runs at Q256; IBS 512 saturates. Missing values are not zero.",fontsize=9)
    fig.savefig(path,dpi=160,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    dst=FACTORS/"fixed-profile"
    dst.mkdir(exist_ok=True)
    data=collect()
    write_csv(dst/"metrics.csv",data)
    write_csv(dst/"request-vs-baselines.csv",public_profile(data))
    for cohort,_,_ in COHORTS:
        cohort_rows=[r for r in data if r["cohort"]==cohort]
        plot(dst/f"{cohort}.png",cohort_rows)
        plot(dst/f"request-{cohort}.png",public_profile(cohort_rows),compact=True)

"""Join aggregate paired scores to audited physical counts, then plot fixed and switched paths."""
import argparse
import csv
from itertools import groupby
from pathlib import Path

METRICS = ("correct","macro_tpf","pooled_token_efficiency","N","F","physical_Q")


def records(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def collect(work,pairs):
    measured = {}
    for r in records(work):
        key = (r["model"],r["policy"],int(r["requests"]))
        if key in measured:
            raise ValueError(f"Repeated measurement {key}")
        measured[key] = r
    scores = {}
    for path in sorted(pairs.glob("*-vs-*.csv")):
        if path.name.endswith("-paired.csv"):
            continue
        for r in records(path):
            if r["scope"] != "all":
                continue
            for side in ("baseline","arm"):
                key = (r["model"],r[f"{side}_policy"],int(r["requests"]))
                if key not in measured:
                    continue
                vals = {m:r[f"{side}_{m}"] for m in METRICS}
                if (scores.setdefault(key,vals)!=vals or
                        any(measured[key][m]!=vals[m] for m in ("N","F","physical_Q"))):
                    raise ValueError(f"Scores and physical work disagree: {key}")
    if scores.keys() != measured.keys():
        raise ValueError(f"Missing scores: {measured.keys()-scores.keys()}")
    names = {k[:2] for k in measured}
    if any({k[2] for k in measured if k[:2]==n}!={30,60,90} for n in names):
        raise ValueError("Require all three cohorts per policy")
    return [dict(r,**scores[k]) for k,r in measured.items()]


def label(policy):
    parts = policy.removeprefix("slottrace_").split("_")
    groups = [(s,len(list(g))) for s,g in groupby(parts)]
    if len(groups)==1:
        return "Fixed "+groups[0][0]
    return " / ".join(f"{q} x{n}" for q,n in groups)


def draw(rows,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with out.with_suffix(".csv").open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    all90 = sorted((r for r in rows if r["cohort"]=="all90"),
                   key=lambda r:(len(r["policy"].split("_")),r["policy"]))
    fig,axes = plt.subplots(1,3,figsize=(12,1.7+.52*len(all90)),layout="constrained")
    names = [label(r["policy"]) for r in all90]
    for j,(ax,m,title) in enumerate(zip(axes,METRICS,("Correct / 90","Tokens / forward","Token efficiency (%)"))):
        vals=[float(r[m])*(100 if j==2 else 1) for r in all90]
        bars=ax.barh(range(len(names)),vals,color=["#cb773c" if "r8" in n else "#387889" for n in names])
        ax.bar_label(bars,labels=[f"{v:.0f}" if j==0 else f"{v:.2f}" for v in vals],padding=3)
        ax.set_yticks(range(len(names)),names if j==0 else [])
        ax.invert_yaxis()
        ax.set_xlim(0, max(vals)*1.24)
        ax.spines[["right","top"]].set_visible(False)
        ax.set_title(title)
    fig.suptitle(f"{all90[0]['model']} | fixed and switching offers, frozen90")
    fig.savefig(out.with_suffix(".png"),dpi=150)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("work","pairs","out"):
        p.add_argument("--"+name,required=True,type=Path)
    a=p.parse_args()
    draw(collect(a.work,a.pairs),a.out)

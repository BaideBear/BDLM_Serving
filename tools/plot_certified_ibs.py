"""Plot only sealed 90-case, same-device IBS stop controls."""
import argparse
import csv
import json
from math import isclose
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def scores(rows):
    n, f, q = (sum(r[k] for r in rows) for k in ("N","F","Q"))
    return dict(correct=sum(r["correct"] for r in rows),tpf=mean(r["N"]/r["F"] for r in rows),
                efficiency=100*n/q, N=n,F=f,Q=q)


COHORTS = {"original30": range(30), "new60": range(30, 90), "frozen90": range(90)}
GEMMA_POLICIES = {"Stock": "stock", "Wait for zero debt": "strict", "Pay and stop": "paid"}
LLADA_POLICIES = {"Stock": "stock", "Pay and stop": "certificate"}


def collect(rows, policies):
    assert sorted(r["index"] for r in rows) == list(range(90)), "Expected exactly indices 0–89"
    by_index = {r["index"]: r["arms"] for r in rows}
    return {
        cohort: {label: scores([by_index[i][key] for i in indices])
                 for label, key in policies.items()}
        for cohort, indices in COHORTS.items()
    }


def collect_gemma(doc):
    q = doc["q"]
    assert doc["requests"] == 90
    release = doc["release"].split()
    assert {"rc=0", f"gpu={doc['gpu']}", "memory_mib=0", "apps=''", "locks=closed"} <= set(release)
    rows = []
    for case in doc["cases"]:
        arms = case["arms"]
        assert arms["stock"] == arms["control"], case["index"]
        stock = arms["stock"]
        for arm in arms.values():
            bill = arm["bill"]
            assert (arm["N"], arm["correct"]) == (stock["N"], stock["correct"]), case["index"]
            assert bill["F"] == bill["D"] + bill["E"] and bill["E"] == stock["bill"]["E"]
            assert 0 < bill["peak_decoder_Q"] <= q
        rows.append({"index": case["index"], "arms": {
            key: {**arm["bill"], "N": arm["N"], "correct": arm["correct"]}
            for key, arm in arms.items()
        }})
    data = collect(rows, GEMMA_POLICIES)
    for cohort, policies in data.items():
        for label, actual in policies.items():
            sealed = doc["cohorts"][cohort][GEMMA_POLICIES[label]]
            assert all(actual[k] == sealed[k] for k in ("correct", "N", "F", "Q"))
            assert isclose(actual["tpf"], sealed["mean_tpf"])
            assert isclose(actual["efficiency"], 100 * sealed["history_efficiency"])
    return data


def collect_llada(path):
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    for case in rows:
        stock = case["arms"]["stock"]
        for arm in case["arms"].values():
            assert all(arm[k] == stock[k] for k in ("correct", "N", "ids_sha")), case["index"]
    return collect(rows, LLADA_POLICIES)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gemma", type=Path, action="append", required=True,
                    help="verified Gemma q*-frozen90-summary.json; repeat once per tier")
    ap.add_argument("--llada",type=Path,help="combined verified 90-case compact receipt NDJSON")
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    if not __debug__:
        raise RuntimeError("assertions must be enabled")
    gemma = {}
    device = None
    for path in a.gemma:
        doc = json.loads(path.read_bytes())
        q = doc["q"]
        if q in gemma:
            ap.error(f"duplicate Gemma tier Q{q}: {path}")
        if device is not None and doc["gpu"] != device:
            ap.error(f"Gemma Q{q} was not run on the same device: {path}")
        device = doc["gpu"]
        gemma[q] = collect_gemma(doc)
    data={}
    if a.llada:
        data["LLaDA Q4"]=collect_llada(a.llada)
    data.update((f"Gemma Q{q}", gemma[q]) for q in sorted(gemma))
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
    fig.text(.065,.025,"Each row uses its own same-device stock reference (dashed). 90 requests per policy; suffix payments are billed.",fontsize=9)
    fig.tight_layout(rect=(.015,.065,.995,.94))
    fig.savefig(a.out/"frozen90.png",dpi=155)
    plt.close(fig)


if __name__=="__main__":
    main()

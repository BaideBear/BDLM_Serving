"""Verify the 17-arm frozen30 study and compare with the archived LLaDA 2.1 runs."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/"experiments/llada20-frozen30"
POLICIES = json.loads((DEST/"config.json").read_text())["policies"]
DISPLAY = ["native32", "flow_w1", "flow_w2", "prefix_w2", "flow_w3", "prefix_w3",
           "flow_w4", "prefix_w4", "b3_native32"] + [
    f"optimus_{mode}_c{c}" for mode in ("ibs","obs") for c in (4,8,16,32)]
MANIFEST = json.loads((ROOT/"benchmarks/frozen30/manifest.json").read_text())["samples"]


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def boolean(x):
    if str(x).lower() not in ("true", "false", "0", "1"):
        raise ValueError(x)
    return int(str(x).lower() in ("true", "1"))


def group(rows, policy):
    g = sorted((r for r in rows if r["policy"] == policy), key=lambda r: int(r["index"]))
    assert len(g) == len(MANIFEST) == 30
    for r, m in zip(g, MANIFEST):
        for k in ("index", "task_id", "dataset"):
            assert str(r[k]) == str(m[k]), (k, r[k], m[k])
    return g


def metrics(g):
    def total(k):
        return sum(int(r[k]) for r in g)
    n, f, q, qp = map(total, ("output_tokens", "generation_forwards", "generation_query_rows",
                              "generation_physical_padded_rows"))
    return dict(requests=len(g), n=n, f=f, q=q, physical_q=qp, mean_tpf=mean(
        int(r["output_tokens"])/int(r["generation_forwards"]) for r in g), efficiency=n/q,
        pooled_tpf=n/f, physical_efficiency=n/qp, q_per_forward=q/f,
        physical_q_per_forward=qp/f, correct=sum(boolean(r["correct"]) for r in g),
        truncated=sum(boolean(r["truncated"]) for r in g),
        failed=sum(r["status"] != "success" for r in g))


def export(run):
    scores = read(run/"scores.csv")
    assert len(scores) == 510 and {r["policy"] for r in scores} == set(POLICIES)
    rows = []
    for p in DISPLAY:
        for s, m in zip(group(scores, p), MANIFEST):
            r = json.loads((run/f'{m["index"]:02d}-{p}.json').read_text())
            assert r["model"] == s["model"] == "llada20"
            assert r["policy"] == p and r["generation_cap"] == 2048 and r["t2t_edits"] == 0
            assert all(r[k] == m[k] for k in ("seed", "task_id", "index", "dataset"))
            assert r["status"] == s["status"]
            for k in ("generation_query_rows","generation_forwards","output_tokens",
                      "generation_physical_padded_rows","max_q"):
                assert r[k] == int(s[k])
            assert r["generation_query_rows"] <= r["generation_physical_padded_rows"]
            assert sum(int(k)*v for k,v in r["q_histogram"].items()) == r["generation_query_rows"]
            assert sum(r["q_histogram"].values()) == r["generation_forwards"]
            if p.startswith("prefix"):
                assert r["generation_query_rows"] + r["removed_q"] == r["dense_equivalent_q"]
            rows.append({k: r[k] for k in ("policy","index","dataset","task_id","seed",
                "status","output_tokens","generation_forwards","generation_query_rows",
                "generation_physical_padded_rows","max_q","truncated","t2t_edits","removed_q",
                "dense_equivalent_q")} | dict(correct=boolean(s["correct"])))
    write(DEST/"requests.csv", rows)


def render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    current = read(DEST/"requests.csv")
    sources = {name: [r for r in read(ROOT/f"experiments/{name}/requests.csv")
                      if r["model"] == "llada21"] for name in ("baselines","flowblock-prefix")}
    comparisons, summaries, pairs = [], [], []
    for p in DISPLAY:
        source = "flowblock-prefix" if p.startswith("prefix") or p in {
            "flow_w2","flow_w3","flow_w4"} else "baselines"
        a,b = group(current,p),group(sources[source],p)
        for aa,bb in zip(a,b):
            pairs.append(dict(policy=p, index=aa["index"], dataset=aa["dataset"], task_id=aa["task_id"],
                              correct20=boolean(aa["correct"]), correct21=boolean(bb["correct"])))
        for dataset in ["all", "gsm8k", "math500", "humaneval"]:
            aa,bb = [v if dataset=="all" else [r for r in v if r["dataset"]==dataset] for v in (a,b)]
            x,y = metrics(aa),metrics(bb)
            for model,data in (("llada20",x),("llada21",y)):
                summaries.append(dict(model=model,policy=p,dataset=dataset,**data))
            comparisons.append(dict(policy=p,dataset=dataset,reference=source,
                correct20=x["correct"],correct21=y["correct"],tpf20=x["mean_tpf"],tpf21=y["mean_tpf"],
                pooled_tpf20=x["pooled_tpf"],pooled_tpf21=y["pooled_tpf"],
                efficiency20=x["efficiency"],efficiency21=y["efficiency"],
                physical_efficiency20=x["physical_efficiency"],physical_efficiency21=y["physical_efficiency"],
                f20=x["f"],f21=y["f"],physical_q20=x["physical_q"],physical_q21=y["physical_q"],
                q20=x["q"],q21=y["q"],only20=sum(boolean(r["correct"])>boolean(s["correct"]) for r,s in zip(aa,bb)),
                only21=sum(boolean(r["correct"])<boolean(s["correct"]) for r,s in zip(aa,bb))))
    write(DEST/"summary.csv",summaries)
    write(DEST/"comparison.csv",comparisons)
    write(DEST/"paired_quality.csv",pairs)
    write(DEST/"prefix_deltas.csv", prefix_deltas(current, sources["flowblock-prefix"]))
    write_markdown(summaries)
    write_prefix_markdown(summaries)
    (DEST/"figures").mkdir(exist_ok=True)
    for name, policies in (("comparison",DISPLAY),("flowblock_prefix",[p for p in DISPLAY
                            if p in {f"{kind}_w{w}" for kind in ("flow","prefix") for w in (2,3,4)}])):
        fig, axes = plt.subplots(1,3,figsize=(12, max(3.8,len(policies)*.41+1.2)),layout="constrained")
        for ax,key,title,k in zip(axes,("mean_tpf","efficiency","correct"),
                                  ("Delivered tokens / forward","Token efficiency (%)","Correct / 30"),(1,100,1)):
            y = np.arange(len(policies))
            for j,(model,color) in enumerate((("llada20","#138077"),("llada21","#ad4f2c"))):
                lookup={r["policy"]:r[key] for r in summaries if r["dataset"]=="all" and r["model"]==model}
                ax.barh(y+(.17 if j else -.17),[k*lookup[p] for p in policies],height=.32,color=color,label=model)
            ax.set_yticks(y, [p.replace("optimus_","").replace("_c"," ").replace("_w"," W") for p in policies] if ax is axes[0] else [""]*len(policies))
            ax.invert_yaxis()
            ax.set_title(title,fontsize=11)
            ax.spines[["top","right","left"]].set_visible(False)
        axes[-1].legend(loc="lower right",bbox_to_anchor=(1,1.06),ncol=2,frameon=False)
        fig.savefig(DEST/f"figures/{name}.png",dpi=165)
        plt.close(fig)


def write_markdown(summaries):
    lookup = {(r["model"], r["policy"]): r for r in summaries if r["dataset"] == "all"}
    lines = ["# Frozen30 results: LLaDA 2.0-mini and 2.1-mini", "",
             "Each entry is **2.0 / 2.1** for the same 30 requests. "
             "TPF is a mean across requests; pooled TPF weights by forwards. "
             "Efficiency is delivered tokens divided by request Q, or physical Q when indicated.", "",
             "| Policy | Quality / 30 | Mean TPF | Pooled TPF | Request efficiency | Physical efficiency |",
             "|---|---:|---:|---:|---:|---:|"]
    for p in DISPLAY:
        a, b = (lookup[m, p] for m in ("llada20", "llada21"))
        fields = [f'{a[k]:{fmt}} / {b[k]:{fmt}}' for k, fmt in (
            ("correct","d"), ("mean_tpf",".3f"), ("pooled_tpf",".3f"),
            ("efficiency",".2%"), ("physical_efficiency",".2%"))]
        lines.append("| " + " | ".join([f"`{p}`", *fields]) + " |")
    lines += ["", "## FlowBlock on 2.0: cost per request and per batch", "",
              "| W | Policy | Delivered N | Forwards | Request Q | Physical Q | Q/forward | Quality / 30 |",
              "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for w in (2, 3, 4):
        for method in ("flow", "prefix"):
            r = lookup["llada20",f"{method}_w{w}"]
            lines.append(f"| {w} | {method} | {r['n']:,} | {r['f']:,} | {r['q']:,} | "
                         f"{r['physical_q']:,} | {r['q_per_forward']:.2f} | {r['correct']} |")
    lines += ["", "Includes truncated and failed requests, with their actual costs. "
              "These are request-level decoding metrics, not measurements of wall-clock throughput. "
              "Native 2.0 recomputes its prefix; Native 2.1 uses a committed-prefix KV cache. "
              "Padding depends on the archived batch schedule.", "",
              "![Three-metric comparison](figures/comparison.png)", "",
              "![Dense and cached W=2/3/4](figures/flowblock_prefix.png)"]
    (DEST/"results.md").write_text("\n".join(lines)+"\n")


def prefix_deltas(current, previous):
    """PrefixCache minus dense; keep paired correctness flips and batch costs."""
    deltas = []
    for model, rows in (("llada20", current), ("llada21", previous)):
        for w in (2, 3, 4):
            a, b = (group(rows, f"{p}_w{w}") for p in ("flow", "prefix"))
            for dataset in ("all", "gsm8k", "math500", "humaneval"):
                aa, bb = (v if dataset == "all" else [r for r in v if r["dataset"] == dataset]
                          for v in (a, b))
                dense, cached = metrics(aa), metrics(bb)
                flips = [boolean(y["correct"])-boolean(x["correct"]) for x, y in zip(aa, bb)]
                assert sum(flips) == cached["correct"]-dense["correct"]
                deltas.append(dict(model=model, window=w, dataset=dataset,
                    only_dense=flips.count(-1), only_cache=flips.count(1),
                    **{"delta_"+k: cached[k]-dense[k] for k in (
                        "correct","mean_tpf","pooled_tpf","efficiency","physical_efficiency",
                        "f","n","q","physical_q","q_per_forward","physical_q_per_forward")}))
    return deltas


def write_prefix_markdown(summaries):
    lookup = {(r["model"],r["policy"]):r for r in summaries if r["dataset"] == "all"}
    text = ["# FlowBlock: dense vs per-block PrefixCache", "",
            "Every row uses the same 30 requests. Request Q excludes padding; physical Q includes it. "
            "These are costs summed across requests, not wall-clock speed.", ""]
    for m in ("llada20","llada21"):
        text += ["## "+m, "",
                 "| W | Method | Correct / 30 | Mean TPF | Pooled TPF | Efficiency | Physical efficiency | Forwards | Request Q | Physical Q |",
                 "|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for w in (2,3,4):
            for policy in ("flow", "prefix"):
                r = lookup[m,f"{policy}_w{w}"]
                text.append(f"| {w} | {policy} | {r['correct']} | {r['mean_tpf']:.3f} | "
                            f"{r['n']/r['f']:.3f} | {r['n']/r['q']:.2%} | {r['n']/r['physical_q']:.2%} | "
                            f"{r['f']:,} | {r['q']:,} | {r['physical_q']:,} |")
        text.append("")
    text += ["Mean TPF is the equally weighted mean of the 30 individual N/F ratios; pooled TPF is total N / total F. "
             "Quality includes all 30 responses, even length-capped outputs. The 2.1 reference for these six rows "
             "is the paired PrefixCache run, including its dense controls.", ""]
    (DEST/"flowblock-pairs.md").write_text("\n".join(text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="optional private scored run")
    args = parser.parse_args()
    if args.run:
        export(args.run)
    render()

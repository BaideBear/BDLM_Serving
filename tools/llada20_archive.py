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


PREFIX = [f"{p}_w{w}" for w in (2,3,4) for p in ("flow","prefix")]
CACHED = ROOT/"experiments/llada20-cached-b32"
HF = "native32_hf_uncached"
KV = "native32_cached"
REPORT = [HF, KV, *DISPLAY[1:]]


def render():
    current = read(DEST/"requests.csv")
    sources = {name: [r for r in read(ROOT/f"experiments/{name}/requests.csv")
                      if r["model"] == "llada21"] for name in ("baselines","flowblock-prefix")}
    comparisons, summaries, pairs = [], [], []
    hf = group(current, "native32")
    for dataset in ("all", "gsm8k", "math500", "humaneval"):
        summaries.append(dict(model="llada20",policy=HF,dataset=dataset, **metrics(
            hf if dataset == "all" else [r for r in hf if r["dataset"] == dataset])))
    for p in REPORT[1:]:
        source = "flowblock-prefix" if p.startswith("prefix") or p in {
            "flow_w2","flow_w3","flow_w4"} else "baselines"
        a = group(read(CACHED/"requests.csv"),p) if p == KV else group(current,p)
        b = group(sources[source],"native32" if p == KV else p)
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
    draw_figure("comparison", REPORT, summaries)
    draw_figure("flowblock_prefix", PREFIX, summaries)


def draw_figure(name, policies, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    (DEST/"figures").mkdir(exist_ok=True)
    slots = [(p, "llada20" if p == HF else None) for p in policies]
    height = len(slots)*.54 + 1.65
    fig, axes = plt.subplots(1,3,figsize=(14.4,height))
    fig.subplots_adjust(left=.125, right=.988, top=1-1.22/height, bottom=.42/height, wspace=.12)
    fig.text(.014,1-.28/height,"FlowBlock vs PrefixCache" if name == "flowblock_prefix"
             else "LLaDA 2.0-mini vs 2.1-mini", fontsize=18, weight="bold", va="center")
    fig.text(.014,1-.58/height,"Same 30 requests  |  10 GSM8K, 10 MATH-500, 10 HumanEval",
             fontsize=10, va="center", color="#435163")
    y = np.arange(len(slots))
    names = {KV:"KV-cached B32","b3_native32":"B3 B32"}
    native_labels = {"llada20": "HF uncached B32", "llada21": "KV-cached B32"}
    for w in (1,2,3,4):
        names[f"flow_w{w}"], names[f"prefix_w{w}"] = f"FlowBlock W={w}", f"PrefixCache W={w}"
    for mode in ("ibs","obs"):
        for w in (4,8,16,32):
            names[f"optimus_{mode}_c{w}"] = f"Optimus {mode.upper()}={w}"
    for ax,key,title,k,fmt in zip(axes,("mean_tpf","efficiency","correct"),
                                  ("Mean request TPF","Token efficiency (request Q)","Quality"),
                                  (1,100,1),(".3f",".2f",".0f")):
        peak = 0
        for i in range(0,len(slots),2):
            ax.axhspan(i-.5,i+.5,color="#f4f6f8",zorder=0,linewidth=0)
        for j,(model,color,label) in enumerate((("llada20","#138077","2.0-mini"),("llada21","#ad4f2c","2.1-mini"))):
            lookup={r["policy"]:r[key] for r in summaries if r["dataset"]=="all" and r["model"]==model}
            ys = [i + ((.2 if j else -.2) if m is None else 0)
                  for i, (p,m) in enumerate(slots) if m in (None,model)]
            values = [k*lookup[p] for p,m in slots if m in (None,model)]
            suffix = {"correct":"/30","efficiency":"%","mean_tpf":""}[key]
            labels = [format(v,fmt)+suffix for v in values]
            bars = ax.barh(ys,values,height=.32,color=color,label=label)
            ax.bar_label(bars,labels=labels,padding=3,fontsize=9.5,color="#17212c")
            peak = max(peak,max(values))
        ax.set_yticks(y,[native_labels[m] if m is not None else names[p] for p,m in slots]
                      if ax is axes[0] else [""]*len(y))
        ax.set_xlim(0,peak*1.23)
        if key == "correct":
            ax.set_xlim(0,36)
            ax.set_xticks([0,10,20,30])
        ax.set_ylim(len(y)-.5,-.5)
        ax.set_title(title,fontsize=12,loc="left",weight="bold",pad=9)
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.tick_params(axis="both",length=0,labelsize=10,pad=5)
        ax.set_axisbelow(True)
        ax.grid(axis="x",color="#dce3e7",linewidth=.65)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="center right",bbox_to_anchor=(.988,1-.42/height),
               ncol=2,frameon=False,fontsize=11)
    with matplotlib.rc_context({"svg.fonttype":"none","svg.hashsalt":"llada-frozen30"}):
        fig.savefig(DEST/f"figures/{name}.png",dpi=165)
        fig.savefig(DEST/f"figures/{name}.svg",metadata={"Date":None})
    plt.close(fig)


def write_markdown(summaries):
    lookup = {(r["model"], r["policy"]): r for r in summaries if r["dataset"] == "all"}
    lines = ["# Frozen30 results: LLaDA 2.0-mini and 2.1-mini", "",
             "Each entry is **2.0 / 2.1** for the same 30 requests. "
             "TPF is a mean across requests; pooled TPF weights by forwards. "
             "Efficiency is delivered tokens divided by request Q, or physical Q when indicated. "
             "The two B32 reference implementations are listed separately; a dash means not measured.", "",
             "| Policy | Quality / 30 | Mean TPF | Pooled TPF | Request efficiency | Physical efficiency |",
             "|---|---:|---:|---:|---:|---:|"]
    for p in REPORT:
        a = lookup["llada20", p]
        if p == HF:
            fields = [f'{a[k]:{fmt}} / -' for k, fmt in (
                ("correct","d"),("mean_tpf",".3f"),("pooled_tpf",".3f"),
                ("efficiency",".2%"),("physical_efficiency",".2%"))]
            lines.append("| " + " | ".join(["HF generate B32 (uncached)", *fields]) + " |")
            continue
        b = lookup["llada21", p]
        fields = [f'{a[k]:{fmt}} / {b[k]:{fmt}}' for k, fmt in (
            ("correct","d"), ("mean_tpf",".3f"), ("pooled_tpf",".3f"),
            ("efficiency",".2%"), ("physical_efficiency",".2%"))]
        lines.append("| " + " | ".join(["Committed-prefix KV B32" if p == KV else f"`{p}`", *fields]) + " |")
    counts = read(CACHED/"requests.csv")
    total = lambda k: sum(int(r[k]) for r in counts)
    lines += ["", "## LLaDA 2.0-mini: cached B32 accounting", "",
              "| Phase | Forwards | Actual Q | Included in decoding metrics? |",
              "|---|---:|---:|:---:|",
              f"| Prompt-only prefill | {total('prefill_f'):,} | {total('prefill_rows'):,} | No |",
              f"| Denoise | {total('denoise_f'):,} | {total('denoise_q'):,} | Yes |",
              f"| Commit (including terminal blocks) | {total('commit_f'):,} | {total('commit_q'):,} | Yes |",
              "", "Actual QKV projection widths checked in every layer on every forward. "
              "Single-request execution; no batch padding. Both cached systems include the terminal commit. "
              "[Per-request phase counts](../llada20-cached-b32/requests.csv)."]
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
              "Only the separate 2.0 HF uncached reference recomputes its prefix. "
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
            "These are costs summed across requests, not wall-clock speed.", "",
            "![Frozen30 quality, TPF and token efficiency](figures/flowblock_prefix.png)", ""]
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
    text += ["## Where the 2.0 query work went", "",
             "Both methods run the same packed width before retirement (`32 * W`). "
             "Extra full-width work = change in forward count times packed width. "
             "Rows skipped are measured against the dense width on the *cache trajectory*; "
             "they are not savings against the separately generated dense answers.", "",
             "| W | Extra forwards | Extra full-width rows | Rows skipped | Net request Q change | Extra padding | Net physical Q change |",
             "|--:|--:|--:|--:|--:|--:|--:|"]
    for w in (2,3,4):
        dense, cache = (lookup["llada20",f"{p}_w{w}"] for p in ("flow","prefix"))
        width = 32*w
        assert dense["q"] == width*dense["f"]
        df = cache["f"] - dense["f"]
        skipped = width*cache["f"] - cache["q"]
        dq = cache["q"] - dense["q"]
        dpq = cache["physical_q"] - dense["physical_q"]
        pad = (cache["physical_q"]-cache["q"]) - (dense["physical_q"]-dense["q"])
        assert skipped >= 0 and df*width-skipped == dq and dq+pad == dpq
        text.append(f"| {w} | {df:+,} | {df*width:+,} | {skipped:,} | {dq:+,} | {pad:+,} | {dpq:+,} |")
    text += ["", "Mean TPF is the equally weighted mean of the 30 individual N/F ratios; pooled TPF is total N / total F. "
             "Quality includes all 30 responses, even length-capped outputs. The 2.1 reference for these six rows "
             "is the paired PrefixCache run, including its dense controls.", ""]
    (DEST/"flowblock-pairs.md").write_text("\n".join(text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="optional private scored run")
    parser.add_argument("--cached-run", type=Path, help="separate scored 30-request cached B32 run")
    args = parser.parse_args()
    if args.run:
        export(args.run)
    if args.cached_run:
        from llada20_cached_audit import export_cached
        export_cached(args.cached_run, CACHED/"requests.csv", MANIFEST)
    render()

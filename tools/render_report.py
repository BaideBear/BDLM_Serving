"""Rebuild the three public result folders from the archived numeric observations."""
import argparse
import csv
import io
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

from result_catalog import BASE, PREFIX, OBS, MODELS, ORDER, CHUNKS, GEMMA_CHUNKS, ROOT, collect, tables, pairs

TITLES = {BASE:"全部基线", PREFIX:"FlowBlock：块内前缀 KV 缓存", OBS:"Optimus：OBS 跨块注意力掩码"}
EN = {BASE:"Baseline results", PREFIX:"FlowBlock: per-block prefix KV", OBS:"Optimus: block-causal OBS"}
SCOPE = {
    BASE:"四个模型的 Native、FlowBlock W=1/2/3/4、B3，以及已测的 Optimus IBS/OBS 固定窗口 4/8/16/32。"
         "Gemma 用 B=256，IBS/OBS 还覆盖 64/128/256；其他三个模型用 B=32。B3 为仓库适配，不是官方实现。",
    PREFIX:"在原版 packed FlowBlock 的每个块内，已编码且未再修改的连续前缀保留 KV、不再算 Q；块间掩码和窗口推进不变。"
           "直接对照为**同模型、同 W、同批实验的原版 FlowBlock**。Gemma 未测此前缀缓存。",
    OBS:"只在原 OBS 可见性上叠加原生 B32 的块间自回归掩码，不裁剪超窗 Q。**原 OBS 是直接对照**，"
        "原 IBS 是另一种执行模式的参考。窗口全程固定。仅 LLaDA 2.1 和 SDAR 测过此改动；LLaDA 2.0、Gemma 未测。",
}
COMMON = ("每个数值来自同一组 30 题（GSM8K、MATH-500、HumanEval 各 10）。"
          "TPF = 逐题 N/F 的均值；token efficiency = ΣN/ΣQ；质量 = 正确数/30。"
          "Q 是实际计算的请求行；纯 prompt prefill 单列，不计入 F/Q。失败及触及长度上限的请求均保留。")


def name(p):
    if p in ("native","hf_uncached","b3"):
        return {"native":"Native","hf_uncached":"HF uncached (reference)","b3":"B3"}[p]
    if "_w" in p:
        kind, w = p.split("_w")
        return ("FlowBlock" if kind == "flow" else "Prefix KV")+f" W={w}"
    arm, c = p.split("_")
    return ("OBS+B32" if arm == "bobs" else arm.upper())+f" C={c}"


def csv_text(rows):
    out = io.StringIO()
    w = csv.DictWriter(out, rows[0], lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return out.getvalue()


def metrics_cell(r):
    return f'{r["tpf"]:.3f} · {r["efficiency"]:.2%} · {r["correct"]}/30'


def report(study, allrows, paired):
    lines = ["# "+TITLES[study], "", SCOPE[study], "", COMMON, "",
             "[逐题数值](requests.csv) · [全体与各数据集汇总](summary.csv) · [矢量图](figures/comparison.svg)"
             + (" · [配对差值、逐题得失](comparisons.csv)" if paired else ""), "",
             f"![{TITLES[study]}：TPF、token efficiency、质量](figures/comparison.png)", ""]
    for m, model in MODELS.items():
        rows = [r for r in allrows if r["model"] == m]
        if not rows:
            continue
        lines += ["## "+model, ""]
        if m == "gemma" and study == BASE:
            lines += ["[IBS 与 OBS 的配对数值图](figures/gemma-optimus.svg) · "
                      "[可直接编辑和重画的 CSV](gemma-optimus.csv) · "
                      "[Gemma 的采样适配与实测 Q 口径](../../archive/diffusiongemma-optimus/)。",""]
        if study == BASE:
            lines += ["| 策略 | 实测 Q/F | TPF | Token efficiency | 质量 | 运行来源 |",
                      "|---|---:|---:|---:|---:|---|"]
            for r in rows:
                p = r["policy"]
                label = name(p)
                if p == "native":
                    label += " B256" if m == "gemma" else " B32（已完成块缓存 KV）"
                if r["role"] == "paired_repeat":
                    label += "（配对实验重测）"
                lines.append(f'| {label} | {r["q_per_f"]:.2f} | {r["tpf"]:.3f} | {r["efficiency"]:.2%} | '
                             f'{r["correct"]}/30 | [{r["run"]}](../../archive/{r["run"]}/) |')
        else:
            lookup = {r["policy"]:r for r in rows}
            lines += ["每格依次为 **TPF · token efficiency · 质量**。差值均为改进版减直接对照；效率差以百分点计。", ""]
            if study == PREFIX:
                lines += ["| W | 原版 FlowBlock | 前缀 KV | Δ TPF | Δ 效率 | Δ 正确题 |",
                          "|---:|---|---|---:|---:|---:|"]
            else:
                lines += ["| 窗口 | 原 IBS（参考） | 原 OBS（对照） | OBS+B32 | Δ TPF vs OBS | Δ 效率 vs OBS | Δ 正确题 vs OBS |",
                          "|---:|---|---|---|---:|---:|---:|"]
            for x in paired:
                if x["model"] != m or x["baseline_role"] != "control":
                    continue
                p = x["variant"]
                w = p.split("_")[-1].lstrip("w")
                cells = [metrics_cell(lookup["ibs_"+w])] if study == OBS else []
                cells += [metrics_cell(lookup[x["baseline"]]), metrics_cell(lookup[p])]
                lines.append(f'| {w} | ' + " | ".join(cells)
                             + f' | {x["delta_tpf"]:+.3f} | {x["delta_efficiency_pp"]:+.2f} pp | {x["delta_correct"]:+d} |')
        lines.append("")
    notes = {
        BASE:"两次真实测得的 W2/W3 不合并，配对重测单独标记。W4 若只在配对实验测过，即引用那一次。"
             "LLaDA 2.0 缓存 Native 的末块 commit 亦计入；HF uncached 是独立参考，不作为 Native 基线。"
             "[缓存版逐阶段计数](../../archive/llada20-cached-b32/requests.csv)。",
        PREFIX:"[实现](../../bdlm_serving/methods/flowblock/prefix_cache.py) · "
               "[原始 commit/KV 审计](../../archive/flowblock-prefix/README.md) · "
               "[LLaDA 2.0 W2/3/4 成本分解](../../archive/llada20-frozen30/flowblock-pairs.md)。"
               "退休后减少请求 Q 不代表含 padding 的物理 Q 或 forward 次数一定减少；两者均列在 CSV。",
        OBS:"[掩码实现](../../bdlm_serving/methods/optimus/boundary.py) · "
            "[逐题得失与掩码影响范围](../../archive/optimus-masked-obs/)。"
            "IBS/OBS 复用原始实测，并非本次额外运行；与 IBS 的差值也在 comparisons.csv。"
    }
    return "\n".join(lines+[notes[study], ""])


def matrix(study, allrows):
    index = {(r["model"],r["policy"],r["role"]):r for r in allrows}
    if study == BASE:
        keys = [(p,"uncached_reference" if p=="hf_uncached" else "baseline") for p in ORDER]
        keys += [(f"flow_w{w}","paired_repeat") for w in (2,3)]
        return (["2.0-mini","2.1-mini","SDAR","Gemma"], [
            (name(p)+(" (paired rerun)" if role == "paired_repeat" else ""),
             [index.get((m,p,role)) for m in MODELS]) for p,role in keys])
    lookup = {(r["model"],r["policy"]):r for r in allrows}
    grid = []
    for m in MODELS:
        if not any(r["model"] == m for r in allrows):
            continue
        for w in ((2,3,4) if study == PREFIX else CHUNKS):
            pp = [f"flow_w{w}",f"prefix_w{w}"] if study == PREFIX else [f"{p}_{w}" for p in ("ibs","obs","bobs")]
            grid.append((f'{MODELS[m]}   {"W" if study == PREFIX else "C"}={w}', [lookup[m,p] for p in pp]))
    return (["FlowBlock","Prefix KV"] if study == PREFIX else ["IBS","OBS","OBS+B32"], grid)


def chart(path, study, grid, *, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols, rows = grid
    metrics = ("tpf","efficiency","correct")
    fig, axes = plt.subplots(1,3,figsize=(14, 2.15+.34*len(rows)))
    fig.subplots_adjust(top=1-1.43/fig.get_figheight(),bottom=.36/fig.get_figheight(),left=.158,right=.99,wspace=.09)
    fig.text(.016,.974,title or EN[study],weight="bold",size=16,va="top",color="#182737")
    fig.text(.016,1-.49/fig.get_figheight(),"Frozen 30 requests   |   10 GSM8K, 10 MATH-500, 10 HumanEval   |   - = not measured",size=10,color="#475867")
    for ax,key,title,color in zip(axes,metrics,("Mean request TPF","Token efficiency","Correct / 30"),("#93c7dc","#8bc9bd","#e4b47d")):
        peak = 30 if key=="correct" else (max((r[key] for _,rr in rows for r in rr if r), default=0) or 1)
        for y,(_, rr) in enumerate(rows):
            if y%2==0:
                ax.axhspan(y-.5,y+.5,color="#f4f6f8",lw=0)
            for x,r in enumerate(rr):
                if r:
                    ax.barh(y,.89*r[key]/peak,height=.60,left=x+.03,color=color,zorder=2)
                text = "-" if r is None else (f'{r[key]:.3f}' if key=="tpf" else
                            f'{r[key]:.2%}' if key=="efficiency" else f'{r[key]}/30')
                ax.text(x+.48,y,text,ha="center",va="center",size=9.5,color="#182737")
        ax.set(xlim=(0,len(cols)),ylim=(len(rows)-.5,-.5))
        ax.set_xticks([i+.48 for i in range(len(cols))],cols,fontsize=9)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(rows)),[l for l,_ in rows] if ax is axes[0] else [])
        ax.set_title(title,loc="left",fontsize=12,weight="bold",pad=11)
        ax.tick_params(length=0,pad=5)
        for spine in ax.spines.values():
            spine.set_visible(False)
    path.parent.mkdir(parents=True,exist_ok=True)
    with matplotlib.rc_context({"svg.fonttype":"none","svg.hashsalt":"bdlm-result-catalog-v1"}):
        fig.savefig(path.with_suffix(".png"),dpi=160)
        fig.savefig(path.with_suffix(".svg"),metadata={"Date":None})
    plt.close(fig)


def verify_figure(path, grid):
    assert path.with_suffix(".png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    ns = "{http://www.w3.org/2000/svg}"
    actual = Counter(t.text for t in ET.parse(path.with_suffix(".svg")).iter(ns+"text"))
    expected = Counter()
    for _, series in grid[1]:
        for r in series:
            expected.update([f'{r["tpf"]:.3f}',f'{r["efficiency"]:.2%}',f'{r["correct"]}/30'] if r else ["-"]*3)
    assert not (expected-actual), (path,expected-actual)


def build(check=False):
    for study, groups in collect().items():
        reqs, summaries = tables(groups)
        direct = pairs(study, groups)
        overall = [r for r in summaries if r["dataset"]=="all"]
        files = {"requests.csv":csv_text(reqs),"summary.csv":csv_text(summaries),
                 "README.md":report(study,overall,direct)}
        if direct:
            files["comparisons.csv"] = csv_text(direct)
        folder = ROOT/"experiments"/study
        if study == BASE:
            lookup = {r["policy"]:r for r in overall if r["model"]=="gemma"}
            cells = [(f"C={c}",[lookup[f"{m}_{c}"] for m in ("ibs","obs")]) for c in GEMMA_CHUNKS]
            manual = [dict(row=label,series=mode,tpf=r["tpf"],token_efficiency_pct=r["efficiency"]*100,
                           correct=r["correct"]) for label,rr in cells for mode,r in zip(("IBS","OBS"),rr)]
            files["gemma-optimus.csv"] = csv_text(manual)
            opt_grid = (["IBS","OBS"],cells)
            opt_fig = folder/"figures/gemma-optimus"
            if not check:
                chart(opt_fig,study,opt_grid,title="DiffusionGemma B256: Optimus IBS vs OBS")
            verify_figure(opt_fig,opt_grid)
        if not check:
            folder.mkdir(parents=True,exist_ok=True)
        for name_, text in files.items():
            if check:
                assert (folder/name_).read_text()==text, f"Stale {study}/{name_}"
            else:
                (folder/name_).write_text(text)
        grid = matrix(study,overall)
        fig = folder/"figures/comparison"
        if not check:
            chart(fig,study,grid)
        verify_figure(fig,grid)
        print(f"{study}: {len(groups)} groups; {len(reqs)} request rows; {len(direct)} comparisons")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true",help="verify published tables and figure labels, without plotting")
    build(parser.parse_args().check)

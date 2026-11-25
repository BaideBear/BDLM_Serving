"""Render the comparison table or generate new figures from verified request data."""
import argparse
from pathlib import Path

from verify_results import STUDIES, audit

TITLES = ("固定基线", "FlowBlock PrefixCache", "Optimus OBS+B32")


def generate_markdown(rows):
    lines = ["# 三组实验指标", "", "TPF、质量、token efficiency 在同一模型内直接跨 batch 比较。"
             "平均 Q/F 是 ΣQ/ΣF；TPF 是逐题 N/F 均值。括号中的“复用”指第一组已计过的对照，不是新执行。", ""]
    for study, title in zip(STUDIES, TITLES):
        lines.extend([f"## {title}", ""])
        for model in ("llada21", "sdar"):
            lines.extend([f"### {model}", "", "|配置|平均 Q/F|最大 Q|TPF|质量|Token efficiency|N/物理 Q|失败|",
                          "|---|---:|---:|---:|---:|---:|---:|---:|"])
            for r in rows:
                if (r["study"], r["model"]) != (study, model):
                    continue
                name = r["policy"] + ("（复用）" if r["reused"] else "")
                lines.append(("|{}|{avg_Q:.2f}|{max_Q}|{TPF:.3f}|{correct}/30|"
                              "{efficiency:.2%}|{physical_efficiency:.2%}|{failed}|").format(name, **r))
            lines.append("")
    return "\n".join(lines)


def plot(rows, dest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dest.mkdir(parents=True, exist_ok=True)
    for study in STUDIES:
        for model in ("llada21", "sdar"):
            data = [r for r in rows if (r["study"], r["model"]) == (study, model)]
            fig, panels = plt.subplots(1, 3, figsize=(11, max(4, len(data)*.40+1.4)), sharey=True)
            for ax, metric, name, scale in zip(panels, ("TPF","correct","efficiency"),
                    ("Mean tokens / forward", "Correct / 30", "Token efficiency (%)"), (1,1,100)):
                values = [r[metric]*scale for r in data]
                bars = ax.barh(range(len(data)), values, color=[
                    "#ca6834" if r["policy"].startswith(("prefix", "optimus_bobs")) else "#43758b" for r in data])
                ax.bar_label(bars, fmt="%.0f" if metric=="correct" else "%.2f", padding=3, fontsize=8)
                ax.set_xlim(0, max(values)*1.22)
                ax.set_title(name)
            panels[0].set_yticks(range(len(data)), [r["policy"] for r in data])
            panels[0].invert_yaxis()
            fig.suptitle(f"{model}  |  {study}  |  30 paired tasks")
            fig.tight_layout()
            fig.savefig(dest / f"{study}-{model}.png", dpi=140)
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--plots", type=Path, help="output directory (requires matplotlib); not the archived originals")
    args = parser.parse_args()
    rows = audit()
    if args.markdown:
        args.markdown.write_text(generate_markdown(rows), encoding="utf-8")
    if args.plots:
        plot(rows, args.plots)
    if not args.markdown and not args.plots:
        print(generate_markdown(rows))

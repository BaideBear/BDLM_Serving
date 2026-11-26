"""Render a complete-case, descriptive budget comparison from the public CSV."""
import argparse
import csv
from pathlib import Path

from .grid import MODELS, ratio_caps

HERE = Path(__file__).resolve().parent


def decision_table(source):
    with source.open(newline="") as f:
        data = list(csv.DictReader(f))
    by = {(int(r["q"]), int(r["k"]), r["selector"], r["cohort"]): r for r in data}
    if len(by) != len(data):
        raise ValueError("Duplicate budget, K, selector and cohort")
    table = [
        "# Gemma：按预算比较前缀比例",
        "",
        "每个预算比较 6.25%、12.5%、25%、50%、75% 和填全部合法空位，共 6 个上限 × 2 种选法。"
        "同一配置命中早先绝对 K 试验时复用原结果，不当成重复测量。每个预算、每个选法都是同一 90 题。",
        "",
        "| Q | 零组正确（原30/新60/全90） | 零组 TPF；效率 | 正确数升且 TPF 不降 | 再要求效率不降 | 例：选法与 K | 正确（原30/新60/全90） | TPF；效率；全生成前缀占比 |",
        "|---:|---|---|---:|---:|---|---|---|",
    ]
    notes = []
    for q in MODELS["gemma"][1]:
        arms = [(q, k, s) for k in ratio_caps(q) for s in ("right", "random")]
        for arm in arms:
            for cohort, n in (("original30", 30), ("added60", 60), ("all90", 90)):
                r = by[(*arm, cohort)]
                if int(r["requests"]) != n:
                    raise ValueError(f"Incomplete: {arm}, {cohort}")
            if any(int(by[(*arm,"original30")][key]) + int(by[(*arm,"added60")][key])
                   != int(by[(*arm,"all90")][key]) for key in ("correct","zero_correct")):
                raise ValueError(f"Disjoint cohorts disagree: {arm}")
        values = [by[(*arm, "all90")] for arm in arms]
        baseline = {(r["zero_correct"], r["zero_tpf"], r["zero_efficiency"]) for r in values}
        if len(baseline) != 1:
            raise ValueError(f"Multiple controls at Q{q}")
        z, t, e = values[0]["zero_correct"], float(values[0]["zero_tpf"]), float(values[0]["zero_efficiency"])
        good = [r for r in values if int(r["correct"]) > int(z) and float(r["tpf"]) >= t]
        three = [r for r in good if float(r["efficiency"]) >= e]
        count = lambda arm, field: "/".join(by[(*arm, c)][field] for c in ("original30", "added60", "all90"))
        base = f"| {q} | {count(arms[0], 'zero_correct')} | {t:.3f}；{e:.2%} | {len(good)}/12 | {len(three)}/12 |"
        if not good:
            table.append(f"{base} 无 | — | — |")
            continue
        r = max(good, key=lambda x: (int(x["correct"]), float(x["efficiency"]), float(x["tpf"]), -int(x["k"])))
        arm = (q, int(r["k"]), r["selector"])
        label = ("右端" if arm[2] == "right" else "随机") + f" K{arm[1]}"
        table.append(f"{base} {label} | {count(arm, 'correct')} | {float(r['tpf']):.3f}；"
                     f"{float(r['efficiency']):.2%}；{float(r['share_all_generation']):.2%} |")
        notes.append(f"* Q{q}，{label}：相对零组救回 {r['rescued']}、丢失 {r['lost']}。"
                     f"质量差配对 95% 区间 [{100*float(r['quality_delta_ci_low']):+.2f}, "
                     f"{100*float(r['quality_delta_ci_high']):+.2f}] 个百分点；"
                     f"TPF 差区间 [{float(r['tpf_delta_ci_low']):+.3f}, {float(r['tpf_delta_ci_high']):+.3f}]。")
    table += ["", "表中计数是**配置数**，不是独立重复数。例子只从满足“正确数升且 TPF 不降”的配置中选；"
              "先选正确数最高，再按效率、TPF 排序。仅作描述，不是独立验证或多重比较调整后的结论。"
              "全生成前缀占比是实际额外前缀行 / 全部生成物理 Q，不是 K/Q，也不是 token efficiency。",
              "", *notes, "",
              "预算图（每点为实测；虚线仅为固定零组轨迹下的前缀机会）：" + " · ".join(
                  f"[Q{q}](figures/gemma-ratio-q{q}.png)" for q in MODELS["gemma"][1]),
              "", "[全部线上指标](gemma-ratio-online.csv) · [固定轨迹机会](gemma-ratio-opportunity.csv) · "
              "[原 30 历史 baseline 配对](historical30.csv)", ""]
    return "\n".join(table)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=HERE / "gemma-ratio-online.csv")
    p.add_argument("--out", type=Path, default=HERE / "gemma-ratio-decisions.md")
    a = p.parse_args()
    a.out.write_text(decision_table(a.source), encoding="utf-8")

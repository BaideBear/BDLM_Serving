# BDLM Serving

同一批 30 题上，单请求解码每次 forward 交付多少 token，实际计算多少 Q，最终答对多少题？结果按研究问题组织，不按模型版本或测试日期拆目录。

| 结果 | 覆盖范围 | 直接查看 |
|---|---|---|
| [全部基线](experiments/baselines/) | LLaDA 2.0-mini、2.1-mini、SDAR、DiffusionGemma；Native、FlowBlock W1–4、B3，及已测的 Optimus IBS/OBS | [数值图](experiments/baselines/figures/comparison.svg) · [汇总](experiments/baselines/summary.csv) |
| [FlowBlock 前缀 KV](experiments/flowblock-prefix/) | 2.0、2.1、SDAR；同一模型、同一 W 的原版 FlowBlock 与前缀 KV，两者都列实测值，W=2/3/4 | [数值图](experiments/flowblock-prefix/figures/comparison.svg) · [配对差值](experiments/flowblock-prefix/comparisons.csv) |
| [Optimus OBS 掩码](experiments/optimus-masked-obs/) | 2.1、SDAR；同一固定窗口的原 IBS、原 OBS、OBS+B32，窗口 4/8/16/32；直接对照是原 OBS | [数值图](experiments/optimus-masked-obs/figures/comparison.svg) · [配对差值](experiments/optimus-masked-obs/comparisons.csv) |

每组均为 GSM8K、MATH-500、HumanEval 各 10 题，不换题、不剔除失败与截断。TPF 是逐请求 `N/F` 的均值；token efficiency 是 `ΣN/ΣQ`；质量是正确数 `/30`。只计实际执行的解码 forward 和 Q，不计纯 prompt prefill；物理 Q（含批内 padding）另列。不同模型的 tokenizer、采样器与每次 forward 的成本不同，不能把 TPF 当作墙钟速度。

共 **84 组独立实测、2,520 次请求**。同一次基线可以在两张表展示，但只计一次执行。真实重测保留各自的数值、运行来源与配对关系，不平均。LLaDA 2.0 的 Native B32 主基线是已完成块缓存 KV 的实测；HF 无缓存结果只作参考。DiffusionGemma 的 [IBS/OBS 配对图](experiments/baselines/figures/gemma-optimus.svg)覆盖七档固定窗口 4–256。

### 复现与核验

`python tools/render_report.py --check` 仅用标准库即可从历史逐题记录核验当前三份报告及图中标注，无需权重或 GPU。安装 `matplotlib` 后运行 `python tools/render_report.py` 可重新生成表和图。运行、评分和环境说明见[复现文档](docs/reproduce.md)。

### 手填数据画图

在集群仓库根目录使用[绘图脚本](tools/plot_summary.py)和 [Gemma Optimus 空模板](tools/templates/gemma-optimus.csv)，不用 GPU、权重或原始答卷：

```bash
python -m pip install matplotlib
cp tools/templates/gemma-optimus.csv /tmp/gemma-optimus.csv
# 在 CSV 填入各组冻结 30 题的汇总，再运行：
python tools/plot_summary.py /tmp/gemma-optimus.csv --out /tmp/gemma-optimus --title "DiffusionGemma: Optimus IBS vs OBS"
```

输出同名 `.png` 和 `.svg`，图上直接标注三个数值。`row`、`series` 是显示标签，按首次出现顺序排列，可增加窗口、方法或模型。`tpf` 填逐题 `N/F` 的均值；`token_efficiency_pct` 填 `100 × ΣN/ΣQ`（如 `8.42`，不带 `%`）；`correct` 填 30 题中答对的整数题数。整组未测就留三格空，图上显示 `-`；真实零值填 `0`。手填图不更新 `experiments/` 的正式报告。

| 目录 | 用途 |
|---|---|
| [experiments](experiments/) | 三份正式结果：每份均含可读表、PNG/SVG、逐题与分数据集 CSV |
| [archive](archive/) | 按原始运行保留配置、原始数值和历史图，供追溯 |
| [third_party](third_party/README.md) | 固定版本的 FlowBlock 和 Optimus 上游源码、许可与哈希 |
| [methods](bdlm_serving/methods/) · [runtime](bdlm_serving/runtime/) | 控制器与模型执行适配；B3 是论文推导版本，非官方实现 |
| [evaluation](bdlm_serving/evaluation/) · [benchmarks](benchmarks/) · [tests](tests/) | 冻结题单、checkpoint 身份、评分器及回归测试 |

公开仓库不包含权重、原始答卷或集群日志。

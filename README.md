# BDLM Serving

**研究交接入口：[关键共识、实验路线、集群用法与恢复清单](docs/research-handoff.md)。** 从上下文压缩、换人或换机器恢复时先看这里；历史最好点、筛选回执和正式 90 题结果分开记录。

当前 T2T 实验分支：[Live Profile v1 冻结归档（两个模型、十档固定预算）](docs/live-profile-v1.md) · [单请求执行接口](docs/single-request.md) · [冻结90题同档基线、前缀、跨块掩码及切换结果](experiments/t2t/serving_slots/EXPANDED.md)。下表为保留的原30题归档。

同一批 30 题上，单请求解码每次 forward 交付多少 token，实际计算多少 Q，最终答对多少题？结果按研究问题组织，不按模型版本或测试日期拆目录。

| 结果 | 覆盖范围 | 直接查看 |
|---|---|---|
| [全部基线](experiments/baselines/) | LLaDA 2.0-mini、2.1-mini、SDAR、DiffusionGemma；Native、FlowBlock W1–4、B3，及已测的 Optimus IBS/OBS | [数值图](experiments/baselines/figures/comparison.svg) · [汇总](experiments/baselines/summary.csv) |
| [FlowBlock 前缀 KV](experiments/flowblock-prefix/) | 2.0、2.1、SDAR；同一模型、同一 W 的原版 FlowBlock 与前缀 KV，两者都列实测值，W=2/3/4 | [数值图](experiments/flowblock-prefix/figures/comparison.svg) · [配对差值](experiments/flowblock-prefix/comparisons.csv) |
| [Optimus OBS 掩码](experiments/optimus-masked-obs/) | 2.1、SDAR；同一固定窗口的原 IBS、原 OBS、OBS+B32，窗口 4/8/16/32；直接对照是原 OBS | [数值图](experiments/optimus-masked-obs/figures/comparison.svg) · [配对差值](experiments/optimus-masked-obs/comparisons.csv) |
| [预测下一轮 T2T](experiments/t2t/) | LLaDA 2.1 与 DiffusionGemma 的原生轨迹、预算分配与对照；离线命中率不等于在线性能 | [LLaDA 排名](experiments/t2t/llada21-rankings.csv) · [Gemma 排名](experiments/t2t/gemma-rankings.csv) |
| [Hard-v1 在线配额](experiments/t2t/hard/) | LLaDA 2.1 B32 与 DiffusionGemma B256；只收已完成的冻结 30 题组，按同模型、同名义窗口与原始 IBS/OBS 对照 | [LLaDA SVG](experiments/t2t/hard/figures/llada21-hard-study.svg) · [Gemma SVG](experiments/t2t/hard/figures/gemma-hard-study.svg) · [宏平均与配对](experiments/t2t/hard/macro-paired.csv) |

每组均为 GSM8K、MATH-500、HumanEval 各 10 题，不换题、不剔除失败与截断。TPF 是逐请求 `N/F` 的均值；token efficiency 是 `ΣN/ΣQ`；质量是正确数 `/30`。只计实际执行的解码 forward 和 Q，不计纯 prompt prefill；物理 Q（含批内 padding）另列。不同模型的 tokenizer、采样器与每次 forward 的成本不同，不能把 TPF 当作墙钟速度。

前三项归档共 **84 组独立实测、2,520 次请求**，不含单独更新的 T2T 在线研究。同一次基线可以在两张表展示，但只计一次执行。真实重测保留各自的数值、运行来源与配对关系，不平均。LLaDA 2.0 的 Native B32 主基线是已完成块缓存 KV 的实测；HF 无缓存结果只作参考。

**DiffusionGemma 26B-A4B（B=256）：[20 组基线合并图](experiments/baselines/figures/gemma-baselines.svg) · [对照表，含实测 Q/F](experiments/baselines/#diffusiongemma-26b-a4b) · [绘图 CSV](experiments/baselines/gemma-baselines.csv)**。图中并列 Native、FlowBlock W=1–4、B3，以及 4/8/16/32/64/128/256 七档固定窗口下的 Optimus IBS 和 OBS；同一窗口的两种模式相邻。

### 复现与核验

`python tools/render_report.py --check` 仅用标准库即可从历史逐题记录核验当前三份报告及图中标注，无需权重或 GPU。安装 `matplotlib` 后运行 `python tools/render_report.py` 可重新生成表和图。运行、评分和环境说明见[复现文档](docs/reproduce.md)。

### 手填数据画图

在集群仓库根目录使用[绘图脚本](tools/plot_summary.py)和 [Gemma Optimus 空模板](tools/templates/gemma-optimus.csv)，不用 GPU、权重或原始答卷：

```bash
python -m pip install matplotlib
# 从已发布的 20 组数值重画 Gemma 基线总图：
python tools/plot_summary.py experiments/baselines/gemma-baselines.csv --out /tmp/gemma-baselines --title "DiffusionGemma 26B-A4B: all baselines"
# 仅对比 Optimus，也可从空模板开始：
cp tools/templates/gemma-optimus.csv /tmp/gemma-optimus.csv
# 在 CSV 填入各组冻结 30 题的汇总，再运行：
python tools/plot_summary.py /tmp/gemma-optimus.csv --out /tmp/gemma-optimus --title "DiffusionGemma: Optimus IBS vs OBS"
```

输出同名 `.png` 和 `.svg`，图上直接标注三个数值。`row`、`series` 是显示标签，按首次出现顺序排列，可增加窗口、方法或模型。`tpf` 填逐题 `N/F` 的均值；`token_efficiency_pct` 填 `100 × ΣN/ΣQ`（如 `8.42`，不带 `%`）；`correct` 填 30 题中答对的整数题数。整组未测就留三格空，图上显示 `-`；真实零值填 `0`。手填图不更新 `experiments/` 的正式报告。

| 目录 | 用途 |
|---|---|
| [experiments](experiments/) | 已归档的基线/控制器报告和 T2T 研究，含数值表、图与逐题 CSV |
| [archive](archive/) | 按原始运行保留配置、原始数值和历史图，供追溯 |
| [third_party](third_party/README.md) | 固定版本的 FlowBlock 和 Optimus 上游源码、许可与哈希 |
| [methods](bdlm_serving/methods/) · [runtime](bdlm_serving/runtime/) | 控制器与模型执行适配；B3 是论文推导版本，非官方实现 |
| [evaluation](bdlm_serving/evaluation/) · [benchmarks](benchmarks/) · [tests](tests/) | 冻结题单、checkpoint 身份、评分器及回归测试 |

公开仓库不包含权重、原始答卷或集群日志。

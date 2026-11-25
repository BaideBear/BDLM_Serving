# 相同席位，换一批旧位置是否也有效？

本轮比较事先指定的四个名义 Q：LLaDA **4** 与 Gemma **32** 是上一轮筛出的正例，另外两组 LLaDA **16** 与 Gemma **64** 是反例。使用原有 30 题、生成种子、native 资格校验和生成成本口径，原始模型输出与权重留在集群；不以这批题证明泛化。

- `hard_slack_placebo`、`hard_trigger_placebo` 先根据本请求的状态计算原策略会预留几个旧位置，再从**所有**非 fresh 已解析位置里无放回均匀抽取同样数量；生成采样器、MASK 选择、强制刷新和中性回填保持不变。选择器有独立、确定性的 RNG。触发成功时，安慰剂的行可以不在前一轮 T2T 的邻近区域。
- 当两个选择器看到同一个状态时，fresh、实际选位行数、MASK 集合、额外复查席位数严格一致；不要求两条已经分叉的轨迹每轮仍具有相同状态或同样数目的席位。800 个构造状态的合同测试分别核对两种策略。
- 完整分组可以用共享入口重跑，例如 `python -m tools.run_t2t --model gemma --checkpoint MODEL --samples SAMPLES --reference NATIVE --output NEW_RUN --arm 32:0:28:hard_trigger_placebo`。用 `tools/t2t_results.py` 做原有固定判分。

### 物理工作量

完整运行的 `*-paired-cost.csv` 含以下分账：`selected_Q = mandatory_Q + mask_Q + reserved_seats + neutral_backfill_Q`；全程 `Q = selected_Q + nonsparse_nondiagnostic_Q + oracle_Q`。`mandatory_overflow_Q` 是 fresh 单独超过名义 Q 的部分，已经包含在 `selected_Q` 中。[定向组的分账](../adaptive/)和随机组使用相同公式。这里的“零预留”仍允许共享的中性旧行回填。

[LLaDA 成本对照](llada21-work-envelope.csv) · [Gemma 成本对照](gemma-work-envelope.csv)。以此前两个正例为对象，逐个与原始 IBS/OBS 的每个已归档窗口比较。名义窗口、总物理 Q、`ΣQ/ΣF`、逐请求实际 Q 不超过对方的题数分别报告。不插值，不把不同窗口当成同窗口，也不将行数当成实测耗时。这不是强行给 Q 加上限的实验。

同时满足质量严格更高、TPF 不降、总 Q 和平均 Q/F 都不增加的比较中，LLaDA Q4 空位组仅对原始 IBS16 达成这一条件，而且只有 11/30 题的总 Q 没有增加；Gemma Q32 触发组对任何已测 IBS/OBS 均未满足四项条件。两种方法相对各自**同名义窗口**的改进不能直接外推为相同工作量的收益。

### DiffusionGemma：120 个新增请求

| 名义 Q | 对照 | 质量（正确/30） | TPF | Token efficiency | 实际 Q/F |
|---|---|---:|---:|---:|---:|
| 32 | 零预留 | 28 | 7.980 | 13.00% | 58.93 |
| 32 | 空位定向 / 随机 | 29 / 29 | 7.842 / 7.936 | 12.71% / 12.91% | 59.96 / 59.08 |
| 32 | 触发定向 / 随机 | 29 / 28 | 8.111 / 7.994 | 12.97% / 12.95% | 60.02 / 59.65 |
| 64 | 零预留 | 30 | 11.996 | 11.73% | 99.59 |
| 64 | 空位定向 / 随机 | 29 / 27 | 11.417 / 12.109 | 11.36% / 11.75% | 98.30 / 99.41 |
| 64 | 触发定向 / 随机 | 27 / 27 | 11.364 / 12.036 | 11.28% / 11.76% | 98.49 / 99.16 |

Q32 触发定向与随机分别有 428、494 个预留席位，严格 T2T **均为 0**。方向和成本取决于预算、分配方式；这不是“命中旧位 T2T 提升质量”的证据。[图（含同窗口 IBS/OBS）](figures/gemma-placement-study.svg) · [评分](gemma-online-summary.csv) · [实算 Q 拆分和预留命中](gemma-paired-cost.csv)。作图命令：`python -m tools.render_hard_study --study placement`。

### LLaDA 的新排序

| 名义 Q | 对照 | 质量（正确/30） | TPF | Token efficiency | 实际 Q/F |
|---|---|---:|---:|---:|---:|
| 4 | 零预留 | 25 | 3.338 | 19.97% | 12.13 |
| 4 | 空位定向 / 随机 | 26 / 25 | 3.342 / 3.388 | 19.90% / 20.19% | 12.54 / 12.36 |
| 4 | 触发定向 / 随机 | 25 / 24 | 3.427 / 3.370 | 20.01% / 20.11% | 12.91 / 12.94 |
| 16 | 零预留 | 27 | 4.890 | 17.53% | 23.97 |
| 16 | 空位定向 / 随机 | 24 / 27 | 4.811 / 4.812 | 17.15% / 17.39% | 23.57 / 23.97 |
| 16 | 触发定向 / 随机 | 26 / 27 | 4.860 / 4.708 | 17.46% / 16.49% | 23.84 / 23.48 |

Q16 空位定向比随机命中更多旧位严格编辑（100/1,892 对 36/1,589），但少对 3 题；不应把严格编辑都当作有益编辑。[图（含同窗口 IBS/OBS）](figures/llada21-placement-study.svg) · [逐题配对](llada21-placement-comparison.csv) · [实算分账](llada21-paired-cost.csv)。

[不依赖 logits 的几何规则](geometry-preregister.md) · [同状态回放的分母、命中及局限](geometry-results.md)。在线使用同一分配器：`hard_slack_geometry` 只在现有空位替换排序，`hard_geometry` 固定预留，可挤占 MASK 但保留至少一个可用的推进席位。原生 B32 全宽没有选位自由度。

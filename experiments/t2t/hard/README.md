# Hard-v1：在线宏平均与配对

| 模型 | 宏平均及分数据集 | 逐题及内部对照 | 图 |
|---|---|---|---|
| LLaDA 2.1 B32 | [汇总](llada21-online-summary.csv) | [逐题](llada21-online-requests.csv) · [配对质量](llada21-online-paired-quality.csv) | [SVG](figures/llada21-hard-study.svg) |
| DiffusionGemma B256 | [汇总](gemma-online-summary.csv) | [逐题](gemma-online-requests.csv) · [配对质量](gemma-online-paired-quality.csv) | [SVG](figures/gemma-hard-study.svg) |

**样本与宏平均。** 冻结 30 题，各数据集 10 题；只收有 `complete-*.json`、30 题身份匹配且生成账本通过校验的组。失败和截断不剔除。质量是 `correct/30`，TPF 是逐题 `N/F` 的算术均值（不是 `ΣN/ΣF`），token efficiency 是 `ΣN/ΣQ`。纯 prompt prefill 不计入生成，实际 bootstrap、verify、commit/重编码计入。名义 Q 不等于实际 Q/F，也不能把 TPF 当作墙钟速度。

**同窗口宏平均与逐题配对。** [对照 CSV](macro-paired.csv) 对每个已完成在线组按同模型、原始 `ibs_Q` / `obs_Q`、`role=baseline` 匹配：[原始宏平均](../../baselines/summary.csv) · [原始逐题](../../baselines/requests.csv)。逐题连接键为 `(index, dataset, task_id)`；质量列区分双方正确、仅在线正确、仅基线正确、双方错误，`delta_correct = 仅在线正确 − 仅基线正确`。TPF 列列出逐题 `Δ(N/F)` 的宏平均和正/平/负题数。比较的是同名义窗口，不声称相同物理 Q/F；CSV 同时给出双方实测 Q/F。内部机制对照读上表的配对质量：frontier 对同 Q 同配额的 random，以及同 Q 零预留的 no-edit；native 是参考线。LLaDA Q64 若完成，只是单 B32 饱和，不配 IBS/OBS；两模型全宽在图中只保留一条 dense 记录。

**成功标准。** 同一对照上，`Δcorrect > 0` 且 `ΔTPF > 0` 才记为双升；质量持平且 TPF 上升单列为“保质提速”。一升一降是取舍，不以 token efficiency、覆盖率或单项提升替代双指标判定。配对计数是这 30 题上的描述性结果，不等于显著性或总体外推。

**读表例子。** LLaDA Q8 frontier 是 `27/30, 4.075 TPF`，原 IBS/OBS 分别为 `23/30, 2.293` 和 `21/30, 2.337`；质量配对分别是多对 4/6 题、没有反向丢题，满足双升，但实际 Q/F 更高。相对同 Q random 和 no-edit，frontier 各多对 1 题、TPF 却分别低 0.070 和 0.126，因此不能把外部双升归因于 frontier 排名。Gemma Q128 frontier 对原 IBS/OBS 双升；Q16 frontier 对两者都是 TPF 上升、质量下降。

复算：`python -m tools.t2t_results --samples SAMPLES --run HARD_V1_RUN --out experiments/t2t/hard --only-complete`；随后 `python tools/compare_hard_study.py` 与 `python tools/render_hard_study.py`。原始运行目录：n164 上的 `/data1/project_data/gx-erqw-20261123/runtime/t2t-native-1125/hard-v1-{llada21,gemma}`。

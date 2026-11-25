# 实验结果索引

每个研究问题只有一份正式结果。打开对应表即可同时查看基线和改进值，不需另查按模型拆分的文件。

| 研究问题 | 正式表 | 数值标注图 |
|---|---|---|
| Native、FlowBlock、B3 和 Optimus 各自表现如何？ | [四个模型的全部基线](../experiments/baselines/) | [基线图](../experiments/baselines/figures/comparison.svg) |
| 在 DiffusionGemma 原生 B256 上，IBS 与 OBS 的固定窗口如何取舍？ | [七档同题对照 CSV](../experiments/baselines/gemma-optimus.csv) | [IBS / OBS 数值图](../experiments/baselines/figures/gemma-optimus.svg) |
| 在 FlowBlock 的同一窗口内，缓存块内前缀是否有效？ | [原版对前缀 KV，W=2/3/4](../experiments/flowblock-prefix/) | [配对图](../experiments/flowblock-prefix/figures/comparison.svg) |
| 限制跨块注意力后，OBS 如何变化？ | [IBS、原 OBS、OBS+B32，窗口 4/8/16/32](../experiments/optimus-masked-obs/) | [三方案图](../experiments/optimus-masked-obs/figures/comparison.svg) |

TPF、质量、token efficiency 使用同一批 30 题。各实验的 `summary.csv` 另列分数据集汇总；`comparisons.csv` 同时列出对照与改进方案的原始指标、差值和逐题得失。历史图片和原始字段见[运行归档](../archive/)。

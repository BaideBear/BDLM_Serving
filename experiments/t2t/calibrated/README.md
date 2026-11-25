# 固定参数的 LLaDA Q16 预测器

在线测试保持原来 30 题不变，不按题号选择预测器。校准题来自同一份 GSM8K / MATH-500 / HumanEval 快照，属于**相同来源上的不同请求**，不是不同分布的外部盲测。先排除原清单里所有 `source_index` 和 prompt SHA-256，再按 `sha256("calibration-v1:{dataset}:{source_index}")` 的排序取不重复的 8 题/集。检查三个快照的字节摘要。未看这些题的生成输出、标签或测评分选择校准题。

在 24 题原生 B32 解码的 still-masked、非 fresh 旧位置上拟合一个 25 系数的 L2 逻辑回归。语义内容和 logits 不进模型，输入只来自已观测的位置、MASK、fresh、刚才的改写和时间。上线后每一道评测题共享同一份权重；无需在运行时训练，也不查询未计算位置的 logits。源数据保留在集群。原生采样器按原逻辑获取已算行的结果。

事先选定 Q16、最多 2 个旧位，同测两个策略：`hard_hazard_fixed` 不设风险门槛；`hard_hazard_gate` 仅在抢占 MASK 时使用 0.05 门槛，有现成空位不设门槛。其余沿用 mandatory fresh、首个 MASK、共享回填和全宽闭合。门槛来自已重复研究的 frozen30 原生留出回放，故虽然拟合请求不重叠，**不能**声称整个研究设计没有看过这 30 题。两种配置分别对照同预算零预留，同时报告正确数、TPF、token efficiency、实际 Q/F、实算 Q，及同状态本可进步的 MASK 被挤掉多少。

分别使用 `python -m tools.calibrate_hazard --raw SNAPSHOTS --samples FROZEN30 --checkpoint CHECKPOINT --output CALIBRATION`，以及 `python -m tools.run_t2t --model llada21 --hazard-weights CALIBRATION/weights.json --checkpoint CHECKPOINT --samples FROZEN30 --reference NATIVE --output RUN --arm 16:0:14:hard_hazard_fixed --arm 16:0:14:hard_hazard_gate`。前者会生成新的校准轨迹，没有 24 道旧的原生参考文件，不要将它说成 frozen30 输出资格验证。后者保留独立的 30/30 全宽复现门槛。

## 冻结参数与覆盖率

24 题提供 169 个可选旧编辑作为训练正例。[权重、校准题来源序号、种子与题面摘要](weights.json)可用于逐项重建；文件 SHA-256 为 `d4457807244a5941cc43dc7ed3b4ba665eba962b8730c73babf8705cf1476478`。在 frozen30 原生轨迹的 191 个可选旧编辑上，不限制实际配额的 top-4/8/16 分别命中 **88/145/190** 个，top-16 总共要列入 11,445 个位置（精度 1.66%）。[完整名单曲线](native-shortlist.csv)。

换成实际 Q16、最多两个旧席位，固定预留为 56 个预留命中、739 个 MASK 被挤占；5% 门槛为 **47 个预留命中、123 个 MASK 被挤占**。两组同状态总 selected Q 都是 19,800。门控保留了同预测器固定预留命中的 83.9%，少挤占 83.4% 的 MASK，但它对全体可选目标的额外召回是 47/191（24.6%），不是 83.9%。[同状态结果](native-replay.csv)。复算：`python -m tools.hazard_gate --weights WEIGHTS --raw NATIVE_LLADA --out OUT.csv --shortlist SHORTLIST.csv`；写入仍来自原生轨迹，不等于在线结果。

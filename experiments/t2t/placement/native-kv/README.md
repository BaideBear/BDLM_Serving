# DiffusionGemma：旧位置在哪里发生 KV 漂移？

**在本轮原生 forward 中，旧 token 没改，不代表它的 K、V 没变。** 固定 30 题，在 decoder 第 15 层（从 0 编号，sliding attention，独立 K/V projection）观察进入 attention 的当前 canvas 行：K 为归一化加 RoPE 后的张量，V 为归一化后的张量。报告不含 KV 激活。每题输出、种子、全部 forward 和 Q 与原生参考逐项一致；不引入额外 forward，不把观察开销算成部署提速。

| 同一 canvas 的相邻去噪轮 | 实测 |
|---|---:|
| 有候选旧行的轮次 / 跨轮比较次数 | 325 / 378 |
| 已接受、非上轮 fresh 的候选行次 | 40,350 |
| joint 漂移前 10% 的目标行次（逐轮进位，正标签可扩充平局） | 4,182 |
| 同名额 frontier 预测命中 | 511 / 4,182（12.22%） |
| 同池、同名额均匀抽取的期望命中 | 434.99（10.40%） |
| 相对均匀期望的命中倍数 | 1.175 |
| 在 4,182 个高漂移目标中 | 4,173 个仍接受同一 token；9 个撤销；0 个严格 T2T |

对每一行分别计算 `dK = RMS(K_t − K_prev)/RMS(K_prev)`，`dV` 同理；主标签是 `max(dK,dV)`。过滤数值噪声下限为 `1e-6`。排名分数只读此前已接受、改写或撤销的位置与查询时钟，不读当前 KV 或 logits；预测席位数固定，平局按位置处理。均匀期望先逐轮计算 `名额 × 高漂移行数 / 候选数` 再求和。每个 canvas 的第一轮无前驱，不跨 canvas 对齐。

所有候选行平均 dK、dV 分别是 0.373、0.519，高漂移目标中是 0.823、1.113。511 次高漂移命中均不是严格 T2T 命中，且单层激活的变化不衡量其对最终答案的影响。原生每步全查，**相邻轮漂移不是稀疏 rollout 相对最新真值的缓存误差**；本实验也不能证明刷新这些行会提高准确率。

[逐题指标](kvdrift-summary.csv) · [聚合统计与原生一致性](kvdrift-aggregate.json)。复现：`python -m tools.native_kv_drift --model gemma --checkpoint MODEL --samples FROZEN30 --reference NATIVE --output NEW_DIR --layer 15`。

**固定 Q 能覆盖多少？** 在上述 325 个有旧候选的状态，假设此时均允许选位，直接用实际分配器回放；各行的分母均为 4,182 个高漂移目标。下表计入中性旧位回填。“少选 MASK”相对同状态零预留。此处前一轮是原生全查，没有模拟稀疏轨迹上的过期缓存，不是端到端 TPF 或质量结果。

| Q | 零预留命中 | 只用空位命中 | 编辑触发命中 | 固定留 Q/8 命中 | 固定策略累计少选 MASK |
|---:|---:|---:|---:|---:|---:|
| 16 | 5 | 10 | 10 | 32 | 169 |
| 32 | 30 | 37 | 37 | 122 | 557 |
| 64 | 148 | 190 | 192 | 415 | 1,481 |
| 128 | 863 | 916 | 917 | 1,246 | 2,423 |
| 256 | 4,182 | 全选 | 全选 | 不适用 | 0 |

[全部固定预算、同名额期望与分母](kvdrift-budget-coverage.csv)。CPU 复算：`python -c 'from tools.native_kv_drift import offline_budget_coverage; offline_budget_coverage("NATIVE_DIAGNOSTIC_DIR")'`。

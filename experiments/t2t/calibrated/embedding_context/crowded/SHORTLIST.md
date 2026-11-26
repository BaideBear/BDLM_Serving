# Q8 同一换位：名单是否漏掉目标？

LLaDA 2.1，固定 30 题，冻结 5% 门控。只看原规则在原生前态**实际会改变 Q 集合**的 305 轮。保留原受保护 fresh、同一换出行、同一物理宽度。用无 logits 排序给出块内、基础 Q 集合之外的旧行名单。宽名单本身不送进模型；下表透视只回答：如果已经知道当轮标签、从名单选一行，最多能命中几轮？

| 换出的行 | 换位轮 | 名单 K=1 实际排名命中 | K=2 单席上限 | K=4 单席上限 | K=8 单席上限 | 所有旧行单席上限 | 原生当步接纳损失 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MASK | 120 | 16 | 23 | 29 | 39 | 42 | 45 |
| 普通旧回填 | 185 | 8 | 12 | 19 | 27 | 35 | 0 |
| 合计 | 305 | 24 | 35 | 48 | 66 | 77 | 45 |

在挤走 MASK 的 120 轮，原排序只命中 16 轮。另有 23 轮的目标其实已经在前八名中；不在前八名、但能在其余旧行中找到的只有 3 轮。还有 78 轮，即使给出全部可换入候选，也无法直接命中一个当步严格编辑。固定门控以后，精排序比扩大名单更有可验证的空间；同时需要解释为什么要在这 78 轮牺牲一个 MASK。旧行没有立即编辑不等于刷新它的 KV 一定没有意义。

**按行次的召回与单席上限不同。** 门控打开时，全部 6,432 个未选旧行中有 115 个目标（全体选位轮有 191 个可选旧位严格编辑）。前八名名单共 2,420 行次，覆盖 89/115：换走 MASK 时为 52/63，换走旧回填时为 37/52。由于同一步可有多个改写，即便透视也只能用一个 Q 找到其中的一个；不能把 89、66 或 39 当成可部署策略的实测命中。K=1 是已执行过的原生同前态排名，不意味着后来真实稀疏轨迹也会产生这些标签。

[计数](native-shortlist.csv) · [逐题，含无机会的题](native-shortlist-requests.csv) · [原生回放和门控审计](../gated/NATIVE_Q8E1.md)。

## 用上一步的邻居变化重排，是否能拿到这部分收益？

另外 24 题拟合，仅给原分数增加两项：八格内上一步实际写入引起的输入向量平均变化有多大、该变化和当前旧 token 的向量是否同向。具体见[事先冻结的规则](DELTA_PROTOCOL.md)。未交换的轮次不改门控，换入候选仍限前八名，每次只算一行，不看当前 logits 或 KV。

| 换出类型 | 原命中 | 重排命中 | 新增 / 丢失命中 | 改选次数 | 额外模型 Q |
|---|---:|---:|---:|---:|---:|
| MASK | 16 | 14 | 0 / 2 | 9 | 0 |
| 旧回填 | 8 | 10 | 2 / 0 | 27 | 0 |

**不进入 GPU 测试。** 两组总命中都是 24，合并会掩盖真正关心的区别：最需要判断收益、即挤走 MASK 的轮次，准确选中反而更少。两种排名都换走了 45 个当轮会被原生接纳的 MASK，物理开销不变。这个结果只否定该重排规则，不否定名单中存在更好的候选。[实测表](delta-rank.csv)。

从仓库根目录在 n164 复算（复用已验证的原生路径和 24 题独立拟合参数；CPU，无额外模型 forward）：

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 \
  ../venv/bin/python -B -m experiments.t2t.calibrated.embedding_context.native_gates \
  --shortlist-out experiments/t2t/calibrated/embedding_context/crowded
# Fit on calibration24 and replay frozen30; writes only the sanitized summary
# to a sibling private results directory by default.
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 \
  ../venv/bin/python -B -m experiments.t2t.calibrated.embedding_context.delta_rank
```

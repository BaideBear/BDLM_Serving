# Live Profile v1：单请求固定预算参考策略

冻结于 2026-11-28。此处的 v1 是一对**显式选择**的单请求策略，不是框架默认值，不是多请求调度器，也不是后来实验的逐档最优组合。复用时必须传下列完整的 `profile` 名称。后续候选应另外命名，不覆盖此版本。

| 模型 | 创建函数与 profile | 物理块 B | 外层每步可给的 decoder Q 上限 |
|---|---|---:|---|
| LLaDA 2.1-mini | `open_strict_llada` · `quarter_window_repeat64_paid16_frontier32` | 32 | 4、8、16、32、64 |
| DiffusionGemma 26B | `open_strict_gemma` · `dense_stop_joint_verified_eos` | 256 | 32、64、128、256、512 |

[冻结的实现与依赖树 `14283ae`](https://github.com/BaideBear/BDLM_Serving/tree/14283ae655294dd17824789003f7d3460c10f0e0) · [两个入口](https://github.com/BaideBear/BDLM_Serving/blob/14283ae655294dd17824789003f7d3460c10f0e0/bdlm_serving/methods/t2t/serving_profiles.py) · [题单与身份校验](../benchmarks/frozen90/manifest.json)

## 一步怎样执行

外层给 `step(q)` 一个档位。请求不重做 prefill，不重置缓存与采样状态。先给上轮写过、KV 尚未反映新 token 的位置分配 Q；若本轮只能还这部分，就不强行采样。剩余位置用于未解 MASK 和当前物理块的已解行。只使用实际算过的位置做决策；后块不能让前块看到未来。一次 `step` 有一个受限的 decoder forward；Gemma 需要的 encoder forward 另记，不能靠省略验证或 encoder 来提高 TPF。

| 模型与预算 | 选哪些位置、何时接受 |
|---|---|
| LLaDA Q4、8 | 以当前块第一个未解 MASK 为窗起点；在候选行中保留窗口内位置和可用前缀。固定小档不提前打开下块。 |
| LLaDA Q16 | 同一窗口。冷 MASK 和已驻留且等待采样的 MASK 若**在同一轮已实际投影**，让后者也参与采样；不额外占 Q。 |
| LLaDA Q32 | 当前块的 KV 已全部建立时，从首个未解 MASK 开始，额外接受连续的非 MASK 候选，每行实测置信度必须 `>0.5`。缺行、已解行或弱候选即截断。保留常规接受、T2T 及需要时的一次终端编辑，写回照常计费。 |
| LLaDA Q64 | 当前块已建立且最多剩 1/4 MASK 时才可开后块。已建立的当前块上，普通规则未接受的位置，只有连续两轮实测同一候选、两次置信度均 `>0.5` 才额外接受。降档使这个暂存的票失效，但不会抹去必须偿还的 KV 写回。Q32 连续规则不延伸到 Q64。 |
| Gemma Q32、64、128 | 在当前物理块内先还写回，再分配 MASK 与前缀。前缀从**本块连续已解前缀的右端**选择。Q32/64 的配额系数为可分配余量的 1/8，Q128 为 1/4，不是总 Q 的实测比例。 |
| Gemma Q256、512 | Q256 处理整块；Q512 在当前块剩余不超过一半时可加入后块。验证、写回和重建中能共用的采样合并；EOS 必须经连续终端验证确认，才跳过其后未执行的工作。 |

```python
from bdlm_serving.methods.t2t import serving_profiles as p

# model、input_ids/data 与采样参数沿用对应模型的已加载适配器。
if model_name == "llada21":
    h = p.open_strict_llada(model, input_ids, seed=seed, mask_id=mask_id,
                            eos_ids=eos_ids, profile="quarter_window_repeat64_paid16_frontier32")
else:
    h = p.open_strict_gemma(model, data, seed=seed, initial_offer=q,
                            profile="dense_stop_joint_verified_eos")

while h.status == "ready":
    e = h.step(q)  # 重现图中某个点时，全请求保持同一个 q
    assert 0 < e["decoder_Q"] <= q
output_ids = h.result()
```

## 固定预算结果

同一套冻结 90 题。N 是最终交付的非特殊 token 数；TPF 是逐题 `N / 所有生成阶段 forward 数` 的均值（含 Gemma encoder，不含纯 prompt prefill）；token efficiency 是 `ΣN / Σ被选中读取历史 KV 的 decoder Q`。它不是墙钟加速比，也不是算力利用率。

| LLaDA Q | 答对 / 90 | TPF | token efficiency | Gemma Q | 答对 / 90 | TPF | token efficiency |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 72 | 1.504 | 37.68% | 32 | 85 | 4.879 | 15.63% |
| 8 | 72 | 2.300 | 29.94% | 64 | 83 | 8.176 | 13.96% |
| 16 | 75 | 3.356 | 25.60% | 128 | 88 | 13.047 | 11.97% |
| 32 | 75 | 5.793 | 18.06% | 256 | 83 | 24.252 | 8.69% |
| 64 | 75 | 7.672 | 10.87% | 512 | 84 | 25.596 | 7.42% |

- **对照：**[完整指标、原 30 题与新增 60 题](../experiments/t2t/serving_slots/controls/fixed-request-comparison/README.md) · [主图](../experiments/t2t/serving_slots/controls/fixed-request-comparison/frozen90.png) · [同预算基线差值](../experiments/t2t/serving_slots/controls/fixed-request-comparison/tradeoffs.md)。LLaDA Q32 三项都高于严格 IBS，但同质量时 Flow W1 的 TPF 更高；Gemma Q128 对严格 IBS 多答对 9 题，同时 TPF 较低。不要把 v1 当作每档都胜出的算法。
- **来源：**[固定版本指标 CSV](https://github.com/BaideBear/BDLM_Serving/blob/14283ae655294dd17824789003f7d3460c10f0e0/experiments/t2t/serving_slots/controls/fixed-request-comparison/metrics.csv)，筛选 `method=Live profile` 和 `cohort=frozen90` 即得表中十行。每行指向对应实现分支的独立固定预算实测，不是对切换接口重新运行 90 题，也没有按指标择优。
- **切档证据：**[LLaDA](../experiments/t2t/serving_slots/controls/strict-handle/llada-frontier32/report.json) · [Gemma](../experiments/t2t/serving_slots/controls/strict-handle/gemma-verified-eos.json)。各有 5 档 × 3 题固定回放和全部 25 种有向档位转换的功能验证。这些运行验证请求状态、输出与计费，不证明混合预算路径更优。
- **重画：**从本归档提交的仓库根目录执行 `python -m experiments.t2t.serving_slots.fixed_request_comparison`。绘图不需要模型权重。主图只放 v1 和基线；[消融图](../experiments/t2t/serving_slots/controls/fixed-request-comparison/ablations.png) 单列其他策略。

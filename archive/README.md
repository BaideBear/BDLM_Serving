# 原始运行归档

正式结果按研究问题放在 [`experiments/`](../experiments/)；此处保留每次运行的发布材料，不要求它们具有相同 CSV 字段。正式结果的 CSV 用 `run`、`model`、`policy`、`index` 标识来源与请求；方案别名与复用检查见 [result_catalog.py](../tools/result_catalog.py)。

| 来源 | 原始观测 |
|---|---|
| [baselines](baselines/) | 2.1 与 SDAR 的固定基线扫描 |
| [flowblock-prefix](flowblock-prefix/) | 2.1 与 SDAR，W2/3/4 的原版和前缀 KV 配对 |
| [optimus-masked-obs](optimus-masked-obs/) | 2.1 与 SDAR，masked OBS；其中 IBS/OBS 为复用 |
| [llada20-frozen30](llada20-frozen30/) | 2.0 的原 17 组，包括前缀 KV 和 HF 无缓存参考 |
| [llada20-cached-b32](llada20-cached-b32/) | 2.0 缓存版 Native B32；有逐阶段 F/Q |
| [diffusiongemma-native](diffusiongemma-native/) · [diffusiongemma-wave](diffusiongemma-wave/) | Gemma B256 的原生推理、FlowBlock W1–4 与 B3；wave 中的 Native 为复用 |
| [diffusiongemma-optimus](diffusiongemma-optimus/) | Gemma B256 的固定 IBS/OBS，名义 C=4/8/16/32/64/128/256；逐题记录实际 decoder、encoder 重编码及跨界开销 |

历史对比图与原始实验说明保留在对应目录；新的对比请引用正式结果。

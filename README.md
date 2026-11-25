# BDLM Serving

LLaDA 2.0-mini、LLaDA 2.1-mini、SDAR-8B-Chat-b32 与 DiffusionGemma 的单请求解码实验。核心问题是：**对同一组题，每次 forward 能交付多少 token、需要计算多少行 Q、最终答对多少题？**

| 实验 | 测了什么 | 独立执行 | 看结果 |
|---|---|---:|---|
| [固定基线](experiments/baselines/) | Native B32、FlowBlock W1/2/3、B3 B32、Optimus IBS/OBS 各自固定 4/8/16/32 | 780 | [表](experiments/baselines/summary.csv) · [图](experiments/baselines/figures/comparison.png) |
| [FlowBlock PrefixCache](experiments/flowblock-prefix/) | W2/3/4，每个 W 分别测原 packed 与块内前缀缓存；包含 dense W4 | 360 | [表](experiments/flowblock-prefix/summary.csv) · [图](experiments/flowblock-prefix/figures/comparison.png) |
| [Optimus OBS + B32 掩码](experiments/optimus-masked-obs/) | 保留 OBS 的跨块选位，仅改变生成注意力可见性；与原 IBS、OBS 对比 | 240 | [表](experiments/optimus-masked-obs/summary.csv) · [LLaDA 图](experiments/optimus-masked-obs/figures/llada21.png) · [SDAR 图](experiments/optimus-masked-obs/figures/sdar.png) |

每个“模型 × 配置 × 实验”均使用**同一组 30 题**：GSM8K、MATH‑500、HumanEval 各 10 题。三组共 **1,380 次独立请求执行**。第三组表中的 480 条 IBS/OBS 对照来自第一组，没有重跑。每个实验目录含 `config.json`、逐请求 `requests.csv`、汇总 `summary.csv`、原始结果图及其来源摘要。

[DiffusionGemma 26B-A4B 原生推理](experiments/diffusiongemma-native/)复用同一套 30 题和评分器。已实测 30/30：质量 **28/30**，TPF **24.284**，token efficiency **8.42%**。连同前三组共 1,410 次独立请求执行。不同模型的 tokenizer 和每次 forward 的成本不同，TPF 不等于墙钟速度。

[Gemma 模型归档](experiments/diffusiongemma/)另含 FlowBlock W=1/2/3/4 与 B3，共新增 150 次执行。[三指标图](experiments/diffusiongemma-wave/figures/comparison.png)包含 Native 对照。

[LLaDA 2.0-mini 与 2.1-mini 对照](experiments/llada20-frozen30/)含 17 个配置、510 次新增执行，2.1 对照复用原始归档：[三指标总表](experiments/llada20-frozen30/results.md) · [对比图](experiments/llada20-frozen30/figures/comparison.png) · [W=2/3/4 前缀缓存的 Q 与 forward 成本分解](experiments/llada20-frozen30/flowblock-pairs.md)。不同版本的推理配置并不完全相同，尤其是 Native 的前缀重算与缓存。

**比较口径。** 同一模型内，TPF、质量和 token efficiency 直接跨 batch 比较。TPF 取逐请求 `N/F` 的均值；质量为 30 题正确数；token efficiency 用 `ΣN/ΣQ`，不是逐请求比例的均值。`Q` 是实际送算的请求行数，另报 `physical_Q`（含批内 padding）。窗口是名义配置，不是硬 Q 上限。batch 记录用于复现和解释 padding；不据此比较墙钟吞吐。

**结果怎么读。** LLaDA 上，PrefixCache 在 W2 将平均 Q/F 从 64.00 降到 51.29，但 TPF 从 9.262 降至 7.300，质量从 27/30 降至 23/30。减少单步 Q 不等于减少完成整个请求的工作。OBS+B32 在 LLaDA 窗口 32 将原 OBS 的 21/30 提高到 25/30，而窗口 16 则由 24/30 降至 22/30。加掩码不是所有配置都有收益。这些是固定小样本的配对观察，不是总体排名。

### 快速核对

```bash
git clone --recurse-submodules https://github.com/BaideBear/BDLM_Serving.git
cd BDLM_Serving
python3 tools/verify_results.py --table
```

这条命令只用 Python 标准库：从各题的 N、F、Q、正误重建主指标，检查没有换题，验证复用的对照和六张原图。不需要 GPU、数据集或模型。直接浏览也可打开[三组指标总表](docs/results.md)。

### 代码入口

| 路径 | 用途 |
|---|---|
| [third_party](third_party/README.md) | 版本固定的 FlowBlock 和 Optimus 完整上游源码、许可及控制文件哈希 |
| [methods/flowblock](bdlm_serving/methods/flowblock/) | packed 基线、按块延迟退休、稀疏 Q 与 KV 写回 |
| [methods/optimus](bdlm_serving/methods/optimus/) | 固定 IBS/OBS、选中行 T2T、原生 B32 可见性掩码 |
| [methods/b3](bdlm_serving/methods/b3/) | 论文推导的 B32 熵门控及两模型适配，非作者官方代码 |
| [runtime](bdlm_serving/runtime/) | 原生模型单请求执行、请求隔离、批处理和行数审计 |
| [evaluation](bdlm_serving/evaluation/) · [benchmarks](benchmarks/) | 冻结样本、checkpoint 身份、统一评分及非收敛记账 |
| [tests](tests/) · [复现实验](docs/reproduce.md) | 控制器 CPU 测试、可选真实模型结构检查、生成及评分命令 |

没有发布模型权重、原始答卷、人工聊天记录或集群工具。本仓库测的是统一后端上的控制策略，不是三个论文服务系统的吞吐排行榜。

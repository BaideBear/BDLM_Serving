# 统一单请求方法：先评固定预算

这次的主实验只比较**从生成开始到结束始终给定同一 Q** 的执行。一档是一组独立生成，不是在一条混合轨迹中筛出某些轮次。LLaDA2.1：Q4/8/16/32/64；DiffusionGemma：Q32/64/128/256/512。若某档实现尚未就绪，留空，不使用其它算法拼表。

## 当前候选

公共选位由 `bdlm_serving.methods.t2t.budget_family.choose` 实现。先为发生写入或接受状态变化的欠账分配实际投影；欠账用完预算时只刷新，不能跳过欠账继续采样。冷启动优先填首次投影的行。进入已初始化的块后，在本轮余量 `r` 中，额外前缀上限为 `floor(min(r, B)/8)`，限定本物理块已连续解码、已初始化且不欠账的位置，优先靠近第一个未解码位置，始终为可推进的未解码位置保留至少一行。其它余量用于未解码位置；不为花满预算而添加一次 forward。`1/8` 是冻结候选，不是测得最优值。`0/8` 是移除额外前缀的消融，不允许逐档拼接两个候选。

模型适配器保留各自的采样、T2T、位置编码和提交语义。Gemma 的单块实现接入 Q32/64/128/256。Q512 通过 `FixedFamilyRequest` 启用两个 B256 画布，在一次 `choose()` 和一次物理 forward 中共享预算；小档调用原单块实现。收尾是单独计费的全块验证，小预算物理上分片，任何一次 decoder forward 都不能超档。后续 encoder 全额计入生成 F/Q，但不计入 decoder 宽度。LLaDA Q64 和 Gemma Q512 均有真实 attention 的双向扰动测试，检验后块能读前块、前块看不到后块。

## 固定档验收

- 相同权重、输入、seed、评分器和长度上限。每题每档从 prompt 重新开始；即使同一 worker 复用模型，也不复用请求缓存、生成状态或 RNG。慢题不换题，失败不剔除。固定原30、新60、全90分别出表。
- 报正确数、宏 TPF `mean_i(N_i/F_i)`、token efficiency `sum(N)/sum(paid_Q)`、实际 decoder 最大宽度、均值和总 N/F/Q。付费 forward 包含生成阶段开块、去噪、刷新、验证、提交和 Gemma 后续 encoder；排除纯 prompt prefill。输入模型层的物理投影、输出 token 和终检 argmax 交叉核实。
- 对照为同一实际 decoder 限额的 Optimus IBS/OBS；大档另列相应的 FlowBlock W1/W2，若物理上不等价，单列实际花费，不写为同预算胜出。CPU 和 GPU 分表。两项目标若有取舍，列两个差值、救回/丢失以及按题配对区间；不把“不显著”等同持平，更不声称所有档位的质量必然单调。
- 首轮固定一个超参数规律跑完整档，不根据每档结果改权重。三条先行样本仅检验实现。历史 90 题已被多次分析，不能将新增 60 再称盲测；若要作泛化结论，另行冻结未看过的等配比样本。

## 顺序

优先占用可用的实验配额完成两模型固定档位、真实投影核算、基线配对和上限双块验证。已启动的旧实验不与新实验抢占同卡。暂不启动新的预算组合、路径搜索、响应曲线拟合或 Serving 排布。等固定档验收后，再把“先大后小”与固定档、等成本反向顺序作为独立问题检验；尾部不能用事后输出长度作为在线输入。历史混合路径不进入此次固定档主表。

## 复现

在 n164 对应 checkout 的 Gemma 环境下运行 `python -Bm experiments.t2t.serving_slots.fixed_family_eval --device cpu --tiers 128 256 64 32`，需隐藏 GPU 并限制四个 CPU 核。它将同一组 90 题对四个档位分别完整生成；实际原件留在集群 `prefix90/gemma-fixed-family-v1`。正式上卡须走受保护交接，不直接运行占卡命令。未收齐 90 条不可写入正式结果表。

双块补充入口为 `python -Bm experiments.t2t.serving_slots.gemma_fixed_dual_eval`，在相同的 Gemma 环境、CPU 32–35，运行同样的 90 题，独立登记 Q512。第一次前块可采样且生成上限允许时打开下一块；晋升保留后块的采样状态，重新投影它的 KV。前块 EOS 时即使后块被丢弃，已发生的投影仍然计费。逐题核验 85 个投影模块、实际联合调用、物理峰值 512、终检输出与 encoder 账本。四档单块 CPU 任务源码不因增加双块档而改变。

LLaDA 对应入口为 `python -Bm experiments.t2t.serving_slots.llada_fixed_family_eval --tiers 32 64 16 8 4`（LLaDA 环境、隐藏 GPU、CPU 36–39）。五档共用固定 M2T/T2T 阈值 0.9/0.5 和 `1/8` 选位公式。因此 Q4 的可选前缀名额按向下取整为零，强制刷新不受这项取整影响。保留原生绝对 B32 分界。单次纯 prompt prefill 包含最后不足一块的 prompt，之后不再计算其位置；与先仅 prefill 整块、在生成时再计算余下 prompt 的基线相比，这是明确的缓存策略差异。

LLaDA 本候选采用**写后刷新再提交**：一块已无 MASK、所有位置都已建过 KV 且没有未刷的 token 写入时才退休；产生写入后的实际 forward 全部计费。这里没有额外整块 dense commit，也不声称其深层 KV 或 T2T 收敛状态与原生整块重算等价。在完整样本上判断这种取舍，而非把减少 dense commit 说成无损。Q64 记录真双块调用；二层随机权重 checkpoint 代码的方向性测试分别扰动前/后块，真权重逐轮另核第一层有效加性 mask 与 20 层物理宽度。

以 `python -Bm experiments.t2t.serving_slots.fixed_family_report MODEL --device cpu`（`MODEL=llada21` 或 `gemma`）从集群原件重算。报告显式区分已登记任务、已封存题数、完整 cohort 和同一批题的早期检查；没有实现的档位不计入早期比较，正式原30/新60/全90只填齐全的档位。CPU 行不得与历史 GPU 行直接组成配对结论。

[固定预算 GPU 基线表](../experiments/t2t/serving_slots/controls/fixed-family-references.csv) 含两模型 20 个配置的原30、新60、全90三组统计。正确数来自归档评分，TPF、总行数、每次真实 decoder 峰值从 1,800 个请求的原始调用记录交叉核对；没有重新跑模型。Gemma Optimus 适配沿用归档的 M2T-only 更新规则，不声称与带 T2T 的候选是同一个采样器。Optimus 最大可比原生窗口至一块；两块档保留实算可比的 FlowBlock W2，不把一块档重命名为两块。复算：`python -Bm experiments.t2t.serving_slots.fixed_family_references`。

**预先指定的三题端到端检查（真权重 CPU，不是全90成绩）：** [LLaDA 五档](../experiments/t2t/serving_slots/controls/fixed-family-llada21-cpu3.csv) 和 [Gemma 四个单块档](../experiments/t2t/serving_slots/controls/fixed-family-gemma-cpu3.csv)。两表均只用冻结索引 0、10、20，每数据集一题，列出从头至尾的质量、TPF、效率、真实 decoder 峰值，以及额外前缀刷新/编辑的量。LLaDA Q4/Q8 在这三题均未选择额外前缀：Q4 的配额恒为零，Q8 只要先付至少一个 dirty，剩余配额即不足八行。三题均正常完成，不是超时导致计零；不能从这三题确认低质量的原因。Gemma 四档均为 3/3，宏 TPF 随档上升、效率下降；不据此推断正式质量持平。运行以上 report 时添加 `--smoke-csv` 可复现。

另有[严格旧执行器的三题刷新开销](../experiments/t2t/serving_slots/controls/gemma-strict-drain-grid-cpu3.json)，仅用于识别成本来源，不是统一候选的成绩。Q32/64/128 K0 的纯刷新分别有 2,837/2,929/2,753 个投影行次，其中只因未接受位置重新加噪而计算的分别为 1,581/1,772/1,871；这些数字不是可无损删除的工作量。

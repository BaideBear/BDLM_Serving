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

对相同三题[逐轮回放前缀资格](../experiments/t2t/serving_slots/controls/fixed-family-prefix-opportunity-cpu3.json)，我们先重建实际选中的位置并逐次对账，再分类未选原因。LLaDA Q4 有 296/707 次、Q8 有 401/627 次 forward 同时有干净的块内前缀和可用余量，但向下取整给出零个前缀名额。这两档三题均无可选前缀投影，并非没有机会。Q16/Q32 分别有 254/161 次 forward 确实选用了前缀。此回放不改变任何生成结果，也不估算改配额后的质量或 TPF。复算：`python -Bm experiments.t2t.serving_slots.fixed_prefix_opportunity`。

Gemma 的 Q512 GPU 档用同一冻结 `gemma_fixed_dual_eval.fixed` 做完整单请求生成。它的输出独立于 GPU0 的 360 份单块档原件；只有前序作业归还唯一 holder、360 份封存记录无更改后，后继才能接卡。GPU0 排队不代表已有 CUDA 结果；报表按实际封存样本计数。

GPU 同档配对入口：`python -Bm experiments.t2t.serving_slots.fixed_family_pairs`。先用同一模块的 `--prepare` 在集群离线重评 1,800 个基线输出，对照已归档的 60 个单元逐一核验正确数和 N/F/Q，再封存逐题账本（SHA256 `ff9caccbf776416f0237ddcc73af774fbe567c57929b359583f70fba128d64ce`）。不需要重新生成基线，不把 CPU 候选与 GPU 基线相减。配对输出限完整 cohort，逐题核对 index、dataset、task_id 和 seed，列出救回/丢失、两侧绝对指标和差值。质量按数据集分层用配对 Newcombe 区间，宏 TPF 用分层的配对均值差区间，效率用分层的配对比率差 bootstrap（4,000次）。区间均为逐对的 95% 区间，未校正多重比较。

[Gemma 五个固定档的首题对账](../experiments/t2t/serving_slots/controls/fixed-family-gemma-cpu-case0.csv) 是 Q512 首次全权重端到端验证，不是总体性能评测。五档均答对冻结 index 0。Q512 有 12 次双块计算，16 次 decoder 共实算 5,114 行，另有 1 次 B256 encoder；380 个交付 token，TPF 22.353，效率 7.08%。Q256 同题为 371 个交付 token、23 次 forward、3,826 行，TPF 16.130，效率 9.70%。两档均从同一 prompt 独立推理。复算：`python -Bm experiments.t2t.serving_slots.fixed_family_report gemma --device cpu --case-index 0`。

### 低档：先看全本块，还是重新计算已解码的行？

[CPU 三题的实际输出位置](../experiments/t2t/serving_slots/controls/fixed-family-cold-exposure-cpu3.csv) 按绝对位置追踪首次 M2T，计交付序列直至并包括 EOS，不计 EOS 后丢弃的位置。Q4 有 368/781 个输出位置在当前物理块的 KV 尚未建齐时就首次解码；Q8 为 279/853。等到本块的 key 都可见后，其中分别有 22、27 个位置再次进入模型，但**两档都没有一个重新进入采样器**。单纯刷新 KV 不能纠正 token。Q16 有 213/826 个位置提前解码，其中 20 个后来重新进入采样器、2 个发生 token 变化或撤销。Q32/Q64 没有这种提前解码。这是观测到的机制，不是错误归因。重算：`python -Bm experiments.t2t.serving_slots.fixed_cold_exposure`。

固定 Q4、Q8 各用同一组 index 0/10/20、同一模型和评分器，从头完整生成。三个独立实验组共 18 个请求，不改正式 CUDA 候选：

| 组 | 唯一改变 |
|---|---|
| `eager` | 原执行器冷起立即采样、零可选前缀。对这六个输入，原 `1/8` 也从未实际选到可选前缀；要求重跑的逐轮轨迹、输出、得分、F/Q 与原封存记录完全一致。 |
| `resident` | 新块首次投影不能一轮覆盖时，先按该档真实预算分片建 KV，保持全 MASK，不沿用旧 logits。随后从左侧重新投影开始采样。首轮能装下整个块时沿用原逻辑。建齐表示 key 地址均已驻留，不等于每层缓存与 dense B32 完全相同。 |
| `credit` | 保持立即采样。只有暖轮存在严格块内干净前缀且付完欠账仍有至少两行余量时，累计余量的 `1/8`；满一行才给最靠右的一个前缀，可 T2T，至少留一行给 MASK。未用余额不跨物理块。 |

`python -Bm experiments.t2t.serving_slots.llada_low_start_pilot` 用独立的 CPU 40–43 号核、BF16 跑真权重；复用正式评测器逐层核对实际 Q、掩码、写欠账与退休条件。`python -Bm experiments.t2t.serving_slots.llada_low_start_report --case 0` 仅列同题同档三组均已封存的结果；不加 `--case` 须 18 份原件及最终封存哈希齐全才出三题汇总。首个 Q4 `resident` 请求答对，N/F/Q 为 294/286/1103；历史同题 Q4 候选答错，249/234/872。该例中质量恢复，但付费投影更多，不能据此推广到其余题目或升级主算法。

[同题 Q4/Q8 三臂明细](../experiments/t2t/serving_slots/controls/llada21-low-start-cpu-case0.csv)：Q4 累计前缀 60 行次，无前缀 T2T，仍错；Q8 累计 59 行次，发生 3 次前缀 T2T，答对。这不是这 3 次编辑导致答对的证据，因为 MASK 选位也改变了。Q8 建齐再采样也答对：计入 40 次建块后共 155 次 forward，TPF 1.916；立即采样 288 次，TPF 1.069。两个档位的 `eager` 都已逐轮复现历史候选。Q8 时这题首块仅剩 6 行，首次 forward 就可见完整首块，因此 `resident` 的干预来自**后续物理块**。这六行只描述一题，不给出总体胜率。作为外部参照，同一题同一 CPU 的严格 Optimus IBS Q8（M2T 0.9、K0）答对，TPF 2.504、效率 34.35%；两个新 Q8 实验组都未超过它。

### 三题完整配对：哪一种冷启动值得扩大？

CPU 真权重，固定 index 0、10、20，每次请求重新开始。单元格为 **正确数 / 宏 TPF / token efficiency**。这些是机制实验，并非 90 题正式对照。

| 方法 | 固定 Q4 | 固定 Q8 |
|---|---|---|
| 立即采样（eager） | 0/3・1.217・29.25% | 0/3・1.419・19.65% |
| 先建齐整块（resident） | 2/3・1.075・27.49% | 3/3・1.850・24.48% |
| 跨轮积累可编辑前缀名额（credit） | 0/3・1.089・27.12% | 2/3・1.557・21.85% |
| 块未齐时禁低置信强写（confident） | 3/3・1.289・31.66% | 3/3・2.171・28.11% |
| 硬限 Optimus IBS（同 CPU，0.9，K0） | 此处未配对 | 3/3・2.533・33.32% |

`confident` 只改变一件事：本块首次投影尚未覆盖至块末时，仍接受当轮真实 MASK 上过阈值的提议，但不因“无人过线”而强写一个位置。最后一片和暖块沿用原规则。不查未投影行，不用未来信息。新增六个请求均已检查首次被拦下的写入与 eager 的实际写入一致，之前逐轮相同，所有 forward 照常计费。三题中 Q8 index 10 从 1.643 **降到** 1.557；不能声称每题都更快。

[三臂汇总](../experiments/t2t/serving_slots/controls/llada21-low-start-cpu3.csv) · [置信门控逐题明细](../experiments/t2t/serving_slots/controls/llada21-low-start-gate-cpu3.csv) · [Q8 与严格 IBS 的工作分解](../experiments/t2t/serving_slots/controls/llada21-low-start-work-cpu3.csv)。Q8 resident 的 462 次调用中 111 次只建 KV；它在已投影、仍为 MASK、且不欠账的位置又计算了 1,595 行次，IBS 为 480 行次。两者实际执行轨迹不同，仅凭计数不能断定是前沿顺序导致差距。复核入口为 `llada_low_start_report` 的默认模式、`--gate` 和 `--work`。

进一步把严格 IBS 每一轮的 token 状态、dirty 和清洁 MASK 席数固定，拿实际选中的清洁 MASK 与本块最左侧同数目的清洁 MASK 对照。三个样本合计 244/244 个含清洁 MASK 的 forward 完全相同（84、111、49 轮；共 1,309 行次）。[逐题复核](../experiments/t2t/serving_slots/controls/llada21-q8-ibs-clean-mask-rank-cpu3.csv) 可用 `--frontier` 重算。这个观察仅排除了 **IBS 轨迹上、相同角色配额下** 的 MASK 排序差异，不等于两个执行器分配了相同数量的 MASK、看到了相同的 KV 或产生相同的 logits。

**新增 60 题配对协议**：固定 Q4、Q8，各预算同一题比较未修改的 `1/8` 统一候选与仅拦截部分冷块低置信强写的候选，二者保留相同的暖块前缀配额。依次穿插 30/50/70、31/51/71 等三类数据，各 20 题，共 240 个独立生成请求。原三题不参与；这些题不是盲测。`llada_low_start_pilot --added60-pair` 生成，`llada_low_start_report --added60` 仅在全数完成、注册文件及逐题 hash、逐层 Q、首个干预位置、输入、评分器全部通过检查后输出配对结果，按数据集报告救回/丢失、正确数、宏 TPF 和实算效率。若总体正确数回落、TPF 下降或变化只集中在个别题，不将门控升级为默认；同设备、同 Q 的外部基线仍须另行验证。

### Gemma 五档三题检查

[五档同题实测与 F/Q 分解](../experiments/t2t/serving_slots/controls/fixed-family-gemma-cpu3-five-tiers.csv)。固定 Q32/64/128/256/512 的 CPU 三题均为 3/3，宏 TPF 依次为 **4.909、8.908、14.393、19.555、25.540**；效率为 **13.83%、13.16%、11.89%、10.20%、7.19%**。五档实际 decoder 峰值各自等于预算。Q512 相比 Q256 总 forward 从 54 降到 39，实算总行数却从 9,445 升到 12,815。该组说明真实双块换来了较少的轮数，未证明正确率提升，也未证明打过 GPU 基线。

### LLaDA 五档 GPU 同题基线

以下是同一 GPU 类型、每例固定 Q、index 0/10/20 的诊断，**不是 90 题主表**。候选是原冻结 `1/8`，不是 CPU 冷起门控。单元格为正确数与宏 TPF；完整[配对表](../experiments/t2t/serving_slots/controls/fixed-family-llada21-gpu3-paired.csv)同时列出了双方效率及实算 N/F/Q。

| 固定 Q | 候选 | 严格 IBS | 严格 OBS | FlowBlock |
|---|---|---|---|---|
| 4 | 0/3 · 1.105 | 3/3 · 1.400 | 3/3 · 1.493 | — |
| 8 | 1/3 · 1.326 | 3/3 · 2.270 | 3/3 · 2.736 | — |
| 16 | 3/3 · 2.537 | 3/3 · 3.468 | 1/3 · 3.860 | — |
| 32 | 3/3 · 4.765 | 3/3 · 3.992 | 3/3 · 4.215 | W1: 3/3 · 8.105 |
| 64 | 1/3 · 4.046 | — | — | W2: 3/3 · 12.279 |

三题中 Q32 候选持平两种 Optimus 的正确数，且宏 TPF 较高，但未追上 W1。Q64 比自身 Q32 的正确数和 TPF 都低，未满足更大预算有更好收益的目标。不能用 CPU 的 Q4/Q8 门控数替换 GPU 候选。复核：`python -Bm experiments.t2t.serving_slots.fixed_family_pairs --smoke3 llada21`。GPU4 的旧队列在前序对照评分缺依赖后退出，未登记固定档结果；本组由独立、排他的真实 holder 交接启动，使用原固定档数据/模型/源码和检查，不复用失败链的结果。

**Q64 的行花在哪里？** [三题 GPU 账本](../experiments/t2t/serving_slots/controls/fixed-family-llada21-tail-gpu3.csv)按投影发生时的位置划分：头块 3,853 行、后块 5,081 行；227 次 forward 同时计算两块，34 次只计算头块。后块首次投影共 25 轮，其中 16 轮头块已驻留、有未解 MASK，却没有采样任何头块 MASK。这 16 轮若保留原有实算行，在 Q64 余量内一共容得下 253 个额外头块 MASK 行次；这是容量，不是已经获得的 token。后块首解后最终留在交付位置的有 472 个，其中 17 个后来记录过编辑或撤销。最终输出边界以外共 248 个实算行次（其中 212 个投影时仍属后块）；不能把所有后块行都当成浪费。复算：`python -Bm experiments.t2t.serving_slots.fixed_tail_work --device cuda`。原30、新60及完整90可分别加 `--cohort original30/added60/full90`，样本不足不会出表。

**冷后块机制配对（CPU）**：固定 Q64、B32、同一 `1/8` 选位规则和跨块因果掩码，只干预“头块已驻留且有 MASK、后块正在首次投影”的轮次。`eager` 是冻结对照，要求从头复现三题既有 CPU 轨迹；`defer_tail` 投影同样的行并写 KV，但当轮不在冷后块采样，下一轮要采样须重新投影；`fill_head` 保留所有原有行及采样资格，只用实际余量从左向右补投、采样头块 MASK。新增行可能改变后块读到的头块 KV，因此 `fill_head` 测的是整个补算动作，不把效果单独归给采样。Q32 和第一轮头尾都冷时均不触发。

预先判读：`defer_tail` 若在真正拦下过原本写入的前提下救回错题而不丢原对题，则支持继续研究首次写入；`fill_head` 若实算额外头块行却不能降低每交付 token 的 forward 或以质量为代价，则不视为优于原版。所有开销均入账；三组完整同题后才给总体比较，不用于代替 90 题 GPU 主表。生成入口为 `llada_fixed_family_eval --tail-pilot --tiers 64`，封存后用 `fixed_tail_work --device cpu --pilot` 汇总逐题实算。

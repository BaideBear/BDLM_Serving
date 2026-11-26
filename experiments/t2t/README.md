# Predicting native token edits

## 当前：选位不使用 logits，硬刷新不截断

**LLaDA Q8：能多命中真实 T2T，不等于值得替换 MASK。** 只重排旧回填的定向策略为 **26/30、4.177 TPF、18.67% 效率**，好于同门控随机的 25/30、4.144，却未超过零预留的 26/30、4.201。允许用 5% 门槛换走一个 MASK 后，定向换入行直接改写 26 次，随机仅 9 次；最终定向 25/30、4.067，随机 27/30、4.146。两项定向方案皆未达到部署判据。[完整原生回放、真实 120 条请求、分账、逐题差值和图](calibrated/embedding_context/gated/README.md)。

**Gemma：已接受／待定边界不足以指导额外旧位刷新。** 在不挤占任何 MASK 的同前态原生回放里，用位置、历史写入和左右最近待定行距离预测 L15 高 KV 漂移。Q16、32、64 分别命中 6、36、195 行次，均比同预算较强对照少 4 次；Q128 多 7 次，Q256 本来就全选。未达到预设的上机条件。[五档预算、已有对照与判据](placement/native-kv/shortlist/boundary/README.md)。这是高漂移命中，不是严格 T2T 或在线质量。

**真实固定预算解码已测完。** LLaDA 的输入 embedding 排序相对同席位纯位置组，Q4/Q8/Q16 的 TPF 分别 +0.086/+0.056/+0.037，答对数 +0/+2/+0；三档相对零预留的 TPF 均下降。仅一档满足相对同席位对照“质量严格提高且 TPF 不降”，不采用固定预留作为默认。[三档实测、图、三部分预算与配对区间](calibrated/embedding_context/rollout/README.md)。

**LLaDA：输入 embedding 加邻近写入，在三档实际席位上都比纯位置预测多命中严格编辑。** 用另 24 题拟合，在相同 30 题、相同原生前态、同样的预留席位数下，Q4/1、Q8/1、Q16/2 从 9/23/56 提升到 12/30/59（全体可选旧位严格编辑 191 次）。只按 token 自身的 ID 风险查表则为 9/22/43，未通过。前者值得真实稀疏验证，这些离线数字不是质量或 TPF。[输入 embedding、实测 CPU 成本](calibrated/embedding_context/README.md) · [token 身份对照](calibrated/token_prior/README.md)。

**LLaDA Q4/Q8：完整随机对照与缓存年龄实测。** 同门控、同换出规则，仅改变换入旧位。原四倍规则在 Q4 相对均匀 A/B 多对 3/1 题且 TPF 更高，在 Q8 反而少对 5/6 题且 TPF 更低。按上次实际 Q 的年龄选最旧半池不读 logits；Q8/A 的换入编辑率从 2.22% 提至 3.59%，但少对一题、TPF 下降。两档都未达到预定标准，不设为默认。[对照与三项指标](exchange/placebo/README.md) · [缓存年龄四个直接执行臂与图](exchange/placebo/age/README.md)。

**Gemma Q32 条件换位未通过换种子复测。** 规则冻结后直接执行：原种子 29/30，对照 28/30，TPF 8.130 对 7.980；同题新种子为 27/30 对 28/30，TPF 7.927 对 7.977。两组共 19 次换入候选，干预当轮无直接严格 T2T。不能把旧位刷新说成成功预测了编辑，也不将此策略设为默认。[两组直接执行、图和逐题证据](placement/one-seat/repeat/README.md)。

| 要回答的问题 | 报告、数据与图 |
|---|---|
| 哪些严格 T2T 不在上一轮的刚写位置？哪些只会出现在本来就要全算的验证轮？位置预测能覆盖多少？ | [阶段拆分、位置预测与长名单召回](FRONTIER.md) |
| 一个严格编辑 A→B 在原生路径上发生后，最终交付的是哪一个 ID？ | [LLaDA 分阶段、分 fresh 统计](llada21-edit-retention.csv) · [Gemma](gemma-edit-retention.csv)。LLaDA 可额外选位的 191 次中，179 次交付 B、9 次交付 A、3 次交付第三个 ID，0 次位于返回范围外。Gemma 的 10 次均为 fresh，7 次交付 B、3 次交付 A。这是事件计数，不是唯一位置数；终点相同不要求中途从未改变，不代表对判分的因果贡献。 |
| 固定额外名额能否超过同 Q 的零预留、随机、原始 IBS 和 OBS？ | [两模型固定预算矩阵与图](hard/README.md) |
| 只用空位、编辑触发换位、让探针提前看编辑，是否有用？ | [自适应分配、逐题结果与实际探针成本](adaptive/README.md) |
| 名义 Q64 与实际执行 64 行是否相同？ | [LLaDA 双 B32 的 Q64 实测](dual64/README.md) |
| 同样的 MASK 与刷新席位下，是选位有效，还是随机也一样？ | [匹配席位的随机对照、实际 Q 成本与几何排序](placement/README.md) |
| 固定的十档预算中，哪档相对同 Q 零预留能做到质量提高且 TPF 不降？ | [十档判定](budget-verdicts.csv) · [完整配置、物理工作与同 Q 差值](budget-index.csv) |
| 实际算过的 Q 中，必刷、MASK、显式预留旧行、普通旧行回填各有多少？ | [原种子 17 组实测行数](physical-split.csv) · [构成图](figures/physical-split.png) · [分账和画图脚本](../../tools/physical_split.py)。四列之和等于 `selected_Q`，再加 `other_Q` 才是端到端 `total_Q`；`overflow_Q` 已含在选位工作中，不重复相加。|
| 加上已知 token 的词形，稀疏预算内能找回更多严格编辑吗？ | [LLaDA：独立 24 题拟合、原生 30 题回放](placement/lexical/README.md)。真实可预留席位上的命中差为 Q4 持平、Q8 多 1 次、Q16 少 3 次，未通过预先指定的标准。|
| 在完全相同的前置状态只换一个物理 Q，第一次分叉发生在哪里？ | [单席干预结果](placement/one-seat/README.md) · [协议](placement/one-seat/PREREG.md) |
| 旧位换入时，其他共同 MASK 会更早达到最终交付的基线 token 吗？ | [同一前态两模型、四步位置观察与逐题配对](placement/one-seat/beneficiary/README.md)。LLaDA Q4 定向只省 2 个位置等待轮次；Gemma Q32 的第 0 轮提前没有维持到第 3 轮。两模型均未净增正确题数。 |
| 同一预测席位，保留它的 KV 与使用它的新采样分数，各自能否改变后续输出？ | [Gemma Q32 四组等席位对照、逐题差值与图](placement/one-seat/kv-factorial/README.md) |
| 每一轮都只刷新新增旧位 KV，而不采样这些旧位，有没有端到端收益？ | [Gemma Q16/32/64/128 与原触发、零预留、IBS/OBS 对照](refresh/README.md) |
| 风险证据较弱时把席位还给 MASK，能少牺牲多少推进位置？ | [原生逐题留出回放](placement/HAZARD-GATE.md) · [独立 24 题拟合、同一参数在线测试](calibrated/README.md) |
| 记住近四轮可见的写入和编辑，是否更容易选中旧位 T2T？ | [同预算逐题回放及区间](calibrated/history/README.md)：Q4/8/16 实际预留席位分别少命中 1/5/10 次；不引入这组历史特征。 |
| 如果同时估计旧位改写和被替换 MASK 的解码机会，真实运行有没有收益？ | [LLaDA 三预算、两种交换价格、逐题分账与图](exchange/README.md) |
| 同一门控改为均匀挑选旧位，原预测还能赢吗？缓存过期时间能否区分稀疏编辑风险？ | [Q4/Q8 两条随机执行、结果图、分叉审计与单步估计](exchange/placebo/README.md)。Q4 原预测 26 题对均匀的 23/25 题；Q8 为 22 题对 27/28 题。 |
| 固定物理 Q 宽度，不丢本轮能解码的已选 MASK，最多能找回多少旧位编辑？ | [LLaDA 三档预算的透视上界，以及固定门控后换入/换出两端的消融](exchange/ceiling/README.md)。原生同状态回放，不是在线性能。 |
| 只换候选排序、保留原门控与换出位置，已有位置预测器能否接近透视？ | [同前态实测](exchange/ceiling/README.md#candidate-only-positional-reranking)：Q4/8/16 命中 2/13/48，原规则 3/14/41，同条件透视 17/64/138；没有在多档同时改善。 |
| Gemma 的高 KV 漂移旧行能否用上轮可见信息预测？换回填和换 MASK 的效果能否分开？ | [按整题五折的漂移名单](placement/native-kv/shortlist/README.md) · [单次条件换位的轨迹重建，不是新运行](placement/one-seat/conditional/README.md) |

**双边换位最新实测：** 同预算零预留之外，只在旧位的预计改写足以补偿被换位置时动一个或两个物理席位。LLaDA Q4 编辑价值四倍为 26/30、TPF 3.424、效率 19.87%，对照为 25/30、3.338、19.97%。Q8 同一规则反而少对四题，Q16 逐题判分不变但 TPF 较低。不要把 Q4 的描述性收益外推到较高预算。

十档判定只统计已完成、无 logits、硬刷新、非随机的单块部署策略，不含原生轨迹透视、随机安慰剂或一次性诊断；只在**同一条配置**上要求答对数严格增加且宏平均 TPF 不降。若同一预算有多个满足者，示例按质量、TPF、总物理 Q 顺序选一个，是重复使用 frozen30 后的描述，不是独立验证。零预留仍有中性旧位回填。双块与单块 Q64 是不同调度，详见双块报告。复算：`python -m tools.budget_index`。

**LLaDA Q16 在线结论：** 单独用 24 题校准的无 logits 排序器，固定预留旧位为 27/30、TPF 4.693，5% 门控为 26/30、4.720；零预留为 27/30、4.890。门控比固定预留少挤占 91.8% 的 MASK，但没有提高端到端质量，实际总 Q 也没有减少。[带数值的图、分账与逐题结果](calibrated/README.md)。

为区分点估计和可重复的优势，[配对区间表](budget-paired-intervals.csv)按数据集分层，各从同一配对的 10 题有放回取 10 题，重复 10,000 次，固定种子 20261126。表列出相对同预算零预留的答对题数差、宏平均 TPF 差、合计 token efficiency 的百分点差，以及百分位 95% 区间。LLaDA Q4 几何空位策略、Gemma Q32 触发策略的 TPF 差分别为 +0.066（区间 −0.020 至 +0.180）、+0.131（−0.137 至 +0.408）。这是对这批题的敏感性分析，不是随机种子复测、独立外部验证、非劣检验或多重比较校正；区间包含零不能证明等价。

下列是较早的**固定 fresh 配额、允许使用曾观测置信度**的对照研究；不可把成绩直接当作以上无 logits 策略的结果。

## 历史实验回答了什么

在相同 30 题和种子上，两模型各测试 13 个真实稀疏配置，并各自与全宽原生对照。下面是本题单上的描述性结果，不是对新题的准确率保证。完整配置、实际 Q 和配对结果见[在线总表](online.md)。

- **LLaDA 的非 fresh 改写可以预测，但收益随预算变化。** 单个选位轮的名义 Q16，固定 4 fresh、8 MASK、最多 4 个非 fresh edit 席位时，有条件地选上次预测与已写 token 不同、且当时没有过编辑阈值的位置。定向策略 26/30、TPF 5.028、效率 17.64%；按同一触发规则但随机选位置为 25/30、4.803、17.08%。两者在各自轨迹上实际命中 48/590 和 32/1,078 个预留席位。相对随机组，定向组多答对 2 题、少答对 1 题。若改为每轮固定留出 edit 席位、只按不确定性排序，Q16（4/8/4）为 26/30、5.050、18.07%；不是门控在所有指标上都更优。固定席位到了 Q8（2/4/2）则定向 24/30，落后于随机的 27/30。
- **Gemma 先区分 fresh 与非 fresh。** 在原生轨迹上，10 次严格 T2T 全在上一轮 fresh 中。在 Q128（32 fresh、80 MASK、16 edit）下，定向、随机和零独立 edit 三组均为 30/30；定向 TPF 15.97，低于随机的 17.05 和零 edit 的 16.77。单独改变 fresh 排序更有信息量：Q128（4/124/0）低置信度优先为 29/30、TPF 16.30、效率 9.40%，随机 fresh 为 27/30、14.98、8.93%；Q64（4/60/0）则都是 28/30，TPF 10.74 对 10.70。此处的 4 是预留席位；MASK 不足时，实际计算的 fresh 数可以超过 4。观察到更多改写并不保证多答对题。
- **少算 Q 没有自动换来更高的原生 TPF。** 原生 LLaDA 为 26/30、5.728、15.55%；Gemma 为 28/30、24.284、8.42%。有些稀疏配置提高了 token efficiency，却需要更多 forward 才完成请求。例如 LLaDA Q16 门控定向组的 72,190 行生成 Q 中，选位轮占 23,102 行，全块初始化、验证、提交合计 49,088 行；F 从原生 2,575 增至 2,982。本实验能核实这项成本，不能单独归因给某一个阶段，也没有测量墙钟加速。
- **设计取舍。** LLaDA 保留可开关的非 fresh edit 门控；Gemma 优先比较 fresh 刷新、默认不要求单独分配非 fresh edit。不建议两个模型共用固定 edit 比例，也不建议把“命中改写次数”当成最终优化目标。下一步如需验证可迁移性，应锁定配置，用未参与设计的题和多组种子再测。

**门控在 Q8 的反例。** LLaDA Q8（2/4/2）门控定向组为 27/30、TPF 4.571、效率 19.33%；门控随机组为 25/30、4.666、19.54%。定向预留的席位仅命中 9/709，随机为 25/1,128。也就是说，Q16 的命中优势没有迁移到 Q8；同样不能把 Q8 多答对的两题归因为“命中了更多编辑”。两个模型共 26 个稀疏配置、780 个逐题结果，没有任何配置在这批题上的 TPF 超过本模型原生。

## 原生路径与离线预测

固定 GSM8K、MATH-500、HumanEval 各 10 题，两个模型均 **30/30 与各自原生归档一致**。下表只讨论下一次去噪时会不会改写一个已经接受的 token，不把初次解码、草稿变动或提交 KV 算成改写。

| 模型 | 去噪 forward（不含提交/重编码） | 出现改写的请求 | 严格 T2T | 其中在上一轮 fresh 池之外 | 撤销接受 | 相邻轮 argmax 草稿变化 |
|---|---:|---:|---:|---:|---:|---:|
| LLaDA 2.1-mini B32 | 2,144 | 29/30 | 754 | 311 | 0 | 8,920 |
| DiffusionGemma 26B-A4B B256 | 431 | 8/30 | 10 | 0 | 186 | 17,085 |

排名只使用上一次 forward 已知的量。LLaDA 用 top-1 概率，Gemma 用 top-1 与 top-2 的 logit 差，均取最小的 4 个；随机对照在每一步使用相同的候选池和相同席位数。`fresh` 是上一轮刚接受或刚改写的位置。下表按事件汇总，不是逐题命中率。

| 模型、候选池 | 30 题命中 / 正例 | 随机期望命中 | 每个实际选择席位的命中率 | 留出 9 题命中 / 正例（随机期望） |
|---|---:|---:|---:|---:|
| LLaDA，全部已解码 | 408/754（54.1%） | 133.31 | 408/6,765（6.03%） | 145/274（49.67） |
| LLaDA，排除 fresh | 201/311（64.6%） | 61.02 | 201/5,044（3.98%） | 81/124（26.67） |
| Gemma，全部已接受 | 6/10（60%） | 0.92 | 6/1,512（0.40%） | 1/2（0.03） |
| Gemma，排除 fresh | 0/0（召回率未定义） | 0 | 0/1,300（0%） | 0/0（0） |

LLaDA 有额外 edit 席位可覆盖的目标，但即使在原生稠密轨迹上，排除 fresh 后也只有约每 25 个选中席位命中 1 次。是否值得牺牲 MASK 席位要看真实稀疏执行。Gemma 的 10 次严格改写全在 fresh 池里，不能用 17,085 次草稿波动论证值得设立独立 edit 池；也不能由 30 题断言模型永远不需要它。以上均非质量、TPF 或墙钟提速结果。

逐题计数：[LLaDA](llada21-paths.csv) · [Gemma](gemma-paths.csv)。完整 K=1/2/4/8 排名：[LLaDA](llada21-rankings.csv) · [Gemma](gemma-rankings.csv)。

[LLaDA 改写位置](llada21-positions.csv)与 [Gemma 改写及重新接受的位置](gemma-positions.csv)按请求、去噪轮次、原生块和输出偏移列出，不含 token 内容。`pre_eos` 是最终返回 EOS 之前的位置，`eos` 是最终 EOS 所在的位置，`after_return` 参与了计算但没有交付给用户。Gemma 的 `readmit_different` 是被撤销后以不同 ID 重新接受，不计入连续接受的严格 T2T。

[查看两个模型的原生 T2T 位置图](figures/native-positions.svg)：每行是一道题，横轴是块内偏移，点的面积正比于同一题、同一偏移及同一类别上的事件数（可跨多个原生块）。LLaDA 的 754 次严格改写中，733 次位于最终 EOS 前、2 次在最终 EOS 位置、19 次在最终返回范围外；Gemma 分别是 9、1、0 次。Gemma 另有 12 次撤销后换 ID 重新接受，均位于最终 EOS 前，不并入严格 T2T。

**LLaDA，接近改写阈值的已解码位置。** 上一轮 top-1 不等于当前位置上的 token、且上轮 top-1 置信度未超过原生编辑阈值 0.5，记为 `near_miss`。优先最近一次去噪中真正发生改写的位置的邻居，再按上一轮置信度升序。只考察 fresh 之外的位置。限制每步最多两个 edit 席位；没有信号时将席位留给其他用途。这个策略在全 30 题中以 595 个席位命中 135/311 次，逐步等席位随机期望是 19.78 次。留出 9 题命中 **62/124**，使用 **248** 席位（25.0% 精确率），随机期望是 **8.82** 次。若每步固定填满两个席位，留出集命中 73/124，但需要 1,021 个席位。稀疏执行不能免费观察未选行的 top-1，这些数值不是部署保证。

Native LLaDA 2.1-mini B32 and DiffusionGemma 26B-A4B B256, same frozen 30
prompts and seeds as the published baselines. No FlowBlock stopping changes.

1. **Observe native paths.** Record each denoise step's block, clock, input and
   resulting IDs, resolution/acceptance, draft and confidence proxy, on the
   cluster only. Preserve the original model calls and sampling. Batch one
   matches both archived native runs. Gate on every request's output IDs,
   logical and physical work, and per-step counts where available.
2. **Forecast from history.** At step t, compare rankings using only information
   through t-1. For LLaDA an edit changes a non-prompt previously resolved ID.
   For Gemma the primary label requires acceptance on consecutive steps and
   a changed accepted ID. Re-noising/revocation, re-admission, and argmax draft
   churn are not interchangeable with this label. Include the entire charged
   canvas; values are not limited to the delivered pre-EOS prefix.
3. **Allocate real Q.** Reserve configurable slots for newly resolved/edited
   IDs (their new token has not yet been encoded), unresolved positions, and
   predicted edits outside the refresh set. De-duplicate before executing;
   account for closure/re-encoding, real query rows and padded rows. Compare
   with no-targeted-edits and equal-cost random selection, at identical nominal
   budgets. Teacher-forced coverage is not an online speed or quality result.

## Protocol

`tools/trace_native.py` takes `--model llada21|gemma --checkpoint DIR --samples
FILE --reference DIR --output NEW_DIR`. Use separate pinned model environments
as for the baseline results. Run with the repository on `PYTHONPATH`. No raw
traces, prompts, answers, weights or cluster paths are published.

`tools/analyze_t2t.py --model MODEL --raw VERIFIED_TRACE_DIR --out TABLE_DIR`
evaluates last-step low confidence (native top-1 probability on LLaDA; top-2
logit gap on Gemma), youngest token, distance to another changed position,
and youngest then least confident. Positions break ties. Expected uniform
sampling uses the same number of seats in each individual step. Report micro
precision and recall at 1/2/4/8 seats both over all resolved positions and
outside the mandatory-refresh pool. No extra current-step logits as features.
Compare all 30 descriptively; within each dataset the first seven manifest
indices are development and the last three held out for positional scoring.
These 30 prompts were used in earlier research; not a new independent quality
confirmation set. A dense-trace ranking does not prove stale sparse features
work online. The CPU allocator in `bdlm_serving/methods/t2t/planner.py` is not
a sparsified GPU forward.

## 实际 Q 执行

[在线报告与同预算差值](online.md) · [LLaDA 对比图](figures/llada21-online.svg) · [Gemma 对比图](figures/gemma-online.svg)。[Gemma 数值](gemma-online-summary.csv)与 [LLaDA 数值](llada21-online-summary.csv)采用同一口径。[LLaDA 逐阶段编辑](llada21-online-edits.csv) · [Gemma 逐阶段编辑](gemma-online-edits.csv)。[LLaDA 配对判题](llada21-online-paired-quality.csv)与 [Gemma 配对判题](gemma-online-paired-quality.csv)把“多答对”和“少答对”的题数分开，避免总正确数相同时误认为两种策略在同一批题上表现完全一致。所有在线编辑数是本策略在自身轨迹上观察到的事件，不是对未选位置的召回率。

Gemma 的补充对照把独立 edit 席位固定为 0，`noedit` 在 fresh 内按最近实测的低 logit 差优先，`fresh_random` 只把 fresh 的顺序换成随机；MASK 与剩余非 fresh 的回填规则相同。补充配置 Q64、Q128 都只预留 4 个 fresh 席位。离线同样限定在 `fresh_only` 池，K4 实际命中 6/10，逐轮等席位随机期望 3.85；留出 9 题为 1/2 对 0.50。全原生轨迹的改写数很少，故不把这个信号当成可靠的质量提升保证。

完整跑满相同 30 题的配置才进入结果表。LLaDA：[三项指标及实际 Q/F](llada21-online-summary.csv) · [配对逐题数据](llada21-online-requests.csv) · [按执行阶段分解 F、Q](llada21-online-phases.csv)。表中的 `nominal_q` 是选位轮预算；`q_per_f` 是含所有生成阶段的 `ΣQ/ΣF`，**不是**预算上限。停止时做全块验证；LLaDA 的 KV 提交及 Gemma 的块间重编码均收费。TPF 是逐题 `N/F` 的均值，不是 `ΣN/ΣF` 或耗时；`token_efficiency` 为 `ΣN/ΣQ`。质量是同一批题的 `correct/requests`，截断不删除。

各模型共用 `tools/run_t2t.py --model MODEL --checkpoint DIR --samples FILE --reference DIR --output NEW_DIR`。默认运行两个预算、两种比例及各自配对的三项策略。可重复给 `--arm Q:fresh:mask:selection`；edit 配额为余数。`targeted` 使用已计算位置上最后一次观测到的低置信度；`random` 只随机化非 fresh 的 edit 候选顺序；`noedit` 将预留 edit 席位归还 MASK（已入选的行仍可自然触发 T2T）。不足时回填，不重复计算同一行。LLaDA 的 `near_miss` 和 `gate_random` 增加门控，随机组在**自己的**轨迹上保留相同的席位分配规则，而非强行对齐已经分叉的两条轨迹。运行每种配置前的资格检查按运行目录复用：全部 30 题的全宽输出 IDs 及实测工作量必须与原生归档匹配。

两个模型都先收集 Q 行，再执行模型，不是先做完整 forward 再遮 logits。未选行保留上次测得的块内 KV。Gemma 的神经网络和词表头只收集所选行，但原生全画布采样器仍操作完整画布，未选行使用旧 raw logits，其采样结果被丢弃；因此这些指标不代表墙钟提速。复评分在基线评分环境中执行 `tools/t2t_results.py --samples FILE --run RUN --out experiments/t2t`；可重复 `--run` 合并同一模型的不重复配置。`--only-complete` 只允许发布已有 30 个逐题结果的整组，不允许缩小同组样本数。

从已发布 CSV 重画图和表（无需权重或 GPU）：`python tools/render_t2t.py --input experiments/t2t --out experiments/t2t/online.md --fig-dir experiments/t2t/figures`。绘图依赖 matplotlib。对已完成的门控组运行 `python tools/t2t_gate_summary.py --run RUN` 可核对实际计算的逐层 Q、预留席位、命中和释放席位。

在上述绘图命令末尾加 `--native-positions experiments/t2t` 可一并重画原生位置图。[在线门控诊断](llada21-online-gates.json)按上次实测距当前选位轮的间隔（上一轮、前两轮、更旧、未知）分别列席位及命中数；无席位的命中率为 `null`。它不能用来估计未计算位置的召回。

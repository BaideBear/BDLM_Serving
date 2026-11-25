# Predicting native token edits

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

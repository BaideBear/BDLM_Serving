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

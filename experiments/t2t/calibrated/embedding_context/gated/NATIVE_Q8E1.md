# Native Q8/E1 gate accounting

**同前态实测（固定 30 题）。** 三种规则的 1,170 个选位状态完全相同：
共 12,243 个物理 Q，含 7,932 个必刷行，存在 191 次非 fresh 旧位严格编辑。
零预留在这些状态未命中旧位编辑，选中的 MASK 中有 1,616 个会在当轮解码。

| 选择规则 | 换入旧位次数 | 换入行严格编辑 | 相同门控和换出行的随机期望 | 换出 MASK | 其中当轮本会解码 |
|---|---:|---:|---:|---:|---:|
| 固定最多一席 | 705 | 30 | 9.25 | 520 | 178 |
| 只用旧位回填席 | 185 | 8 | 2.50 | 0 | 0 |
| 有回填就换，否则 5% 门槛 | 305 | 24 | 6.17 | 120 | 45 |

门槛规则保留固定预留 24/30 的旧位命中，同时将当步原生接纳损失从 178 降到 45。
与只用回填相比，额外换入 120 行、多命中 16 次旧位编辑，少覆盖 45 次原生 MASK
接纳。一次改写的重要性不等于一次接纳，不能把两列相减预测质量或速度。
这里的随机是逐步精确期望，**不是**真实随机稀疏运行的结果。

Run on **n164**, from the repository root. Defaults select the original
`llada21` native traces, `context-fixed-q4q8-v1` qualification metadata,
`context-fixed-private/calibration24.json`, and `/data0/ysy/models/LLaDA2.1-mini`.

```sh
cd /data1/project_data/gx-erqw-20261123/runtime/t2t-experiments
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ../venv/bin/python -B -m experiments.t2t.calibrated.embedding_context.native_gates
```

Outputs in this directory: `native_q8e1_requests.csv` (30 requests x 3 rules,
including zero-state requests) and `native_q8e1_summary.csv` (3 sums, not
request averages). Only the manifest ordinal and dataset identify a request.
No online request records, logits, drafts, confidence, KV, or proposals are
inputs. Raw accepted IDs and projected rows stay in process memory.

The replay skips prompt rows, block bootstrap and unmasked verification, and
resets write history per block. The base keeps every fresh row, then earliest
MASKs, then earliest old rows, up to Q8 or mandatory overflow. Only an eligible
old row changing from one resolved ID to a different resolved ID on the current
native transition counts as an old target. Edits on fresh rows are not targets.
`ContextRanker` ranks all nonfresh old rows from pre-step IDs and positional
history; ties go to the lower block position. A top row already in the base
means no swap, even if the runner-up is outside. In slack, all three rules
replace the last backfilled old row with the outside winner. When crowded,
`fixed` may replace the last base MASK, `slack_only` never does, and `gate5pct`
requires a top score at least `log(.05/.95)`. At least two base MASKs must fit
for any MASK displacement. The threshold is a logit cutoff, not a claim of
calibrated live probability.

`physical_swaps` counts changed sets, not nominal reservations. The exact uniform
comparison keeps the ranked rule's swap/no-swap and victim **on each state**.
For a swap, expected added hits = eligible old targets outside base / eligible
old rows outside base; otherwise zero. Fractions are summed before CSV conversion.
Old hits lost, displaced MASKs and imminent native admissions lost are shared by
both placements. `displaced_nonadmitting_masks` counts MASK displacements with
zero immediate admission loss. Selected hits = base hits + added hits - lost
hits. `fixed_rankable_*` and `top1_old_hits` audit the historical ranking spaces;
they are not physical swap counts.

Checks before output: private coefficient digest matches the pinned Q8 digest
and qualified run; 30/30 B32 qualification and native verification; calibration
and frozen request identities disjoint; checkpoint/tokenizer identity; trace
identity, B32 continuity, token ranges and acceptance parity. Per-request base
work, targets and admissions must match the archived Q8 zero-reserve counters;
top-1 and nominal fixed ranking denominators and hits must match the earlier
embedding study. No online selector is used. All comparisons describe the same
native transitions, not what a sparse trajectory would produce.

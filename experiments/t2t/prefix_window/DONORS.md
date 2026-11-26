# Which existing query should give up its seat?

Registered before online donor-arm scores. Gemma B256, frozen90, original seeds, native scoring and full-width qualification. Q128/K32 random is the primary cell; Q32/K8 right is a secondary contrast. These cells were chosen using prior results on these 90 tasks: this is an exploratory mechanism test, not independent validation.

At each selected step, build the existing zero-add plan. Let P be accepted generation rows in the current physical block before the first unaccepted position, minus mandatory fresh rows. B denotes selected ordinary accepted backfill rows, outside P. M denotes selected unaccepted rows other than the protected first one. Strict T2T is not a selection feature. Unselected KV stays visible. No extra forwards, window changes or hard cap on mandatory work.

| Arm | Count on that state | Donor from the zero plan |
|---|---|---|
| Zero (archived, same nominal Q) | 0 | None |
| Backfill-only | min(K, \|P\|, \|B\|) | Rightmost B; preserve every selected MASK |
| Paired-backfill | min(K, \|P\|, \|B\|, \|M\|) | Rightmost B |
| Paired-MASK | Same paired count | Rightmost M |

The two paired arms use exactly the same recipient ordering and physical width on a common state. If there is no donor of either kind, both do nothing. Neither consumes slack: this isolates changing the composition at a fixed forward width. The existing slack-only arm is a separate cost reference. Across diverged trajectories do not assume equal total Q, number of opportunities, draft, KV or token identities.

**Primary comparison:** paired-backfill vs paired-MASK at Q128; report original30, added60 and all90 paired quality, rescues/losses, macro TPF, physical Q and token efficiency. If protecting MASK does not improve correctness or if TPF goes down, do not describe it as a joint gain. Compare backfill-only to zero for a deployable alternative not gated by the presence of a second MASK. Report Q32 identically, without pooling budgets. Check donor class, recipients, zero and actual widths on all selected steps; compare pre-canvas and pre-draft digests up to the first intervention. Afterward compare whole-policy outcomes, not individual-row causal effects. No 90-task selection/filtering or adjustment to the independent decode seed.

On the archived zero trajectories the paired intervention could have exchanged 5,766 rows over 462 selected forwards in 86/90 requests at Q128; Q32: 413 rows, 98 forwards, 35/90 requests. These counts measure opportunity, not quality. Over the same trajectories backfill-only could use 9,389 and 598 rows respectively. Each new online arm uses all 90 cases regardless of opportunity.

Run-list generator: `python -m experiments.t2t.prefix_window.grid gemma 128 donor` (or `32`). The registered runner separately charges initialization, verification and re-encoding.

From the repository root, `python -m experiments.t2t.prefix_window.donor_audit --data "$PRIVATE_PREFIX90" --q 128 --out "$PRIVATE_REPORT"` audits planned recipients and donors across both trajectories, reconciles first interventions to opportunities on the frozen zero trajectory, and scores the direct pair. Use `--opportunities-only` for the preregistered zero-trajectory diagnostic. [Q128 opportunity counts](donors/gemma-donor-opportunity-128.csv) · [Q32 opportunity counts](donors/gemma-donor-opportunity-32.csv).

## Q128：随机选前缀，上限 K32

每组均为相同的 90 题。共 86 题触发配对换位（原30中28题、新60中58题）。两臂直到首次换位的已有记录一致；首个换位时的 canvas、draft 摘要、前缀席位、实际宽度一致。其余 4 题两臂全轨迹相同。每步审计均通过。表中为全90的宏平均 TPF 和合计输出 / 生成物理 Q。

| 策略 | 原30 / 新60 / 全90正确 | TPF | Token efficiency | 实际换出 |
|---|---:|---:|---:|---|
| 零注入 | 28 / 56 / 84 | 17.437 | 11.18% | 0 |
| 只补空位（已有） | 30 / 56 / 86 | 17.819 | 10.93% | 0；额外算 7,635 行次 |
| 只换普通已解码回填 | 30 / 55 / 85 | 17.137 | 11.17% | 9,849 普通，0 MASK |
| 配对：换普通已解码回填 | 29 / 56 / 85 | 17.112 | 11.02% | 6,660 普通，0 MASK |
| 配对：换非首个 MASK | 29 / 55 / 84 | 13.608 | 8.92% | 0 普通，20,587 MASK |

换普通回填相对换 MASK：多答对 1 题（新增60救回1、丢0），质量差配对 95% 区间为 [−7.14, +9.58] 个百分点；TPF 差为 **+3.504**，配对 95% 区间 **[+2.459, +4.550]**。两条路径自首次分叉后机会与轮次可不同，不能把 6,660 和 20,587 说成同样的全程剂量。前者总计 2,025 个生成 forward、296,358 个物理 Q，后者为 2,790、356,819。结果支持在这档预算下优先保留 MASK；两种等宽策略的 TPF 都未超过零组，尚不能推荐用固定前缀配额替换原有查询。

[数值图](donors/figures/gemma-q128.png) · [直接配对和逐数据集区间](donors/direct/gemma-donor-direct-128.csv) · [实际物理 Q、让位计数](donors/exposure-gemma-donor-128.csv) · [与历史基线的原30配对](historical30.csv)。重画命令：`python -m experiments.t2t.prefix_window.plot --model gemma --q 128 --input-dir experiments/t2t/prefix_window/donors --output-dir experiments/t2t/prefix_window/donors/figures`。

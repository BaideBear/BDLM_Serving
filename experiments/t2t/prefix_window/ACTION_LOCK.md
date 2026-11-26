# Do accepted prefix rows need to be rewritten?

Registered before either action-lock run is scored. Use the same frozen90, seed and full-width parity gate. Two previously observed Gemma arms: Q32/K8 right and Q128/K32 random. No test-set or position filtering.

* **Normal**: repeat the original arm, adding counters for proposed strict token changes and revocations on the newly added old prefix rows. Require all 90 outputs, work and old event fields to reproduce the archived run.
* **Lock**: same planner, model, physical attention/LM-head queries, KV writes, sampler, draft, raw logits, self-conditioning scores, and stopping rule on a common state. Only after this step's sampler and stopping decision, restore the input token and accepted flag on added rows where an old accepted token would change or be revoked. Never suppress a fresh/MASK row or the full verification step. After divergence, replan normally, and charge all work. Different trajectories are not required to have equal total Q.
* **Zero**: existing same-budget no-add control. The above comparison is within each Q; the two Q values differ in K and selector and are not a placement head-to-head.

Report original30/added60/all90 paired correctness, TPF, efficiency, counts of exact proposals, affected requests and per-question rescues/losses against both controls. Check identical planned selections, sampled proposals and outputs up to the first suppression and full trajectory identity for requests never directly suppressed. If lock preserves the improvements over zero without direct prefix actions, the sample does not establish that those actions are necessary. This does **not** isolate KV: attention, scores (including draft and self-conditioning) and RNG/shape effects remain. This is not a new holdout or a test of T2T recall.

## 配对实测

四个运行臂各完成固定 90 题，B256 全宽参考 90/90 对齐。两档的正常组重跑均逐题精确复现历史结果：完整输出、计费、旧版逐步事件均相同。锁定只作用于新加入的旧前缀行，不禁止其他位置、整块验证或未来轮次编辑。

| Q、选法 | 策略 | 原30 / 新60 / 全90正确 | 宏平均 TPF（全90） | Token efficiency（全90） |
|---|---|---:|---:|---:|
| 32，右端 K8 | 零注入 | 27 / 56 / 83 | 8.463 | 13.73% |
| | 正常 | 29 / 56 / 85 | 8.522 | 13.71% |
| | 锁定 | 29 / 56 / 85 | 8.522 | 13.71% |
| 128，随机 K32 | 零注入 | 28 / 56 / 84 | 17.437 | 11.18% |
| | 正常 | 30 / 56 / 86 | 17.819 | 10.93% |
| | 锁定 | 30 / 56 / 86 | 17.717 | 10.90% |

* **Q32**：正常组真实加回 1,640 行次，提出 0 次严格改写、0 次撤销。锁开与关，90/90 的输出、物理工作和逐步观测完全一致。相对零注入，均救回原 30 中两题，未丢题。
* **Q128**：正常组加回 7,635 行次，在两个不同请求分别提出 1 次严格改写和 1 次撤销。锁定后，首处分叉前逐步一致；两请求合计拦下 1 次改写、2 次撤销提议（路径分叉后多提出一次）。另 88 题全轨迹一致。全 90 的逐题对错均与正常组相同，相对零组仍救回原 30 中的同两题。一道从未触发锁，一道触发了锁但仍然答对。锁定相对正常组的 TPF 差为 −0.103，配对 95% 区间 [−0.303, +0.098]。
* **含义**：这两组在这 90 题上的正确数改善不需要新算前缀行直接改写或撤销自身 token。但一个位置被真实重算会同时影响多种状态，未证明改善由持久化 KV 单独引起，也不能推出其他预算或模型不需要 T2T。

[Q32 配对数值和区间](actions/action-lock-q32-summary.csv) · [Q128 配对数值和区间](actions/action-lock-q128-summary.csv) · [Q32 数值图](actions/figures/gemma-q32.png) · [Q128 数值图](actions/figures/gemma-q128.png)。在仓库根目录以 `W=experiments/t2t/prefix_window`，运行 `python -m experiments.t2t.prefix_window.plot --action-lock --input-dir "$W/actions" --output-dir "$W/actions/figures"` 可重绘。

# 单次干预的条件策略：精确请求拼接

**性质**：这是对已测 `base` / `rank` / `random` 三条完整轨迹的**事后拼接**（post-hoc splice），不是重跑，不是每步决策，也不是独立证据。条件策略本身没有单独执行；只有第 2 节的可拼接条件全部成立时，拼接才等于条件策略的精确结果。规则比较不是新的独立实验。

## 1. 策略定义

单次干预只看**首个合法候选**：`victim` 是被换出的零预留席位，`candidate` 是换入的定向旧位。

| 策略 | 首个合法候选的 victim | 动作 |
|---|---|---|
| `noedit` | — | 不干预；`base` 臂，逐请求精确复现独立 noedit 运行 |
| `uncond_rank` | 任意（slack 或 mask） | 无条件换入定向候选 |
| `cond_backfill_rank` | 旧回填行（`stratum=slack`） | 换入；否则**永久放弃**，不延迟到下一次 |
| `cond_mask_rank` | MASK（`stratum=mask`） | 换入；否则**永久放弃**，不延迟到下一次 |
| `uncond_random` | 任意 | 换入随机旧位（对照） |
| `cond_backfill_random` | 旧回填行 | 换入随机旧位；否则永久放弃（对照） |
| `cond_mask_random` | MASK | 换入随机旧位；否则永久放弃（对照） |

`slack` 表示被换出的是一个旧回填行，`mask` 表示被换出的是一个 MASK；换入的 `candidate` 始终是定向旧位，不是 mask/backfill。

## 2. 可拼接条件与证明

**命题**：对每个请求，条件策略的完整轨迹精确等于以下拼接：若首个合法候选存在且 `stratum` 匹配，则等于 `rank` 臂；否则等于 `base` 臂。因此每请求 F/Q/N/判分可由已测轨迹精确拼接，无需重跑。

**检查（全部通过，见 `conditional-checks.csv`）**：

1. **固定题单**：30 题，10 gsm8k + 10 math500 + 10 humaneval；`index/task_id/seed` 与 `benchmarks/frozen30/manifest.json` 及已发布 one-seat 表一致。
2. **base 精确复现 noedit**：30/30；所有非 `policy` 字段、noedit 事件字段相等。`base` 只允许额外带 `intervention`、`trajectory_sha256`、`accepted_sha256` 诊断字段。
3. **同一前置状态**：锚点前事件完全一致；锚点、计划、`pre_sha256`、`post_rng_sha256` 一致。LLaDA 28 锚点 + 2 none；Gemma 26 锚点 + 4 none。
4. **单次合法交换**：rank/random 各 28/28、26/26，恰好一个 `victim→candidate`；`victim` 不在 `fresh`，`candidate` 不在 `fresh∪mask`；`victim∈mask ⇔ stratum=mask`。
5. **指标恒等**：`tpf=N/F`、`token_efficiency=N/Q` 在每行成立。
6. **复现校验**：无条件 rank/random 汇总与已发布 `*-online-summary.csv` 一致；重建的 `rank_vs_base`、`random_vs_base`、`rank_vs_random` 与 `*-causal-requests.csv` 一致。
7. **拼接恒等**：每个策略行都是实测 `base`/`rank`/`random` 行对象本身，不是重算；匹配→rank，不匹配→base，none→base。

**证明**：
1. `base` 精确复现独立 noedit，所以 `base` 就是零预留策略的完整轨迹。
2. 三臂在锚点前状态一致，锚点、计划和 victim 类型由共同前置状态唯一决定，与是否执行条件策略无关。
3. `rank` 在锚点执行恰好一次 `victim→candidate`，随后恢复零预留；条件匹配时，条件策略执行同一交换、同一后续策略，轨迹与 `rank` 完全一致。
4. 条件不匹配时，条件策略在锚点不交换，后续完全按零预留策略决策；`base` 就是零预留且此前状态相同，轨迹与 `base` 完全一致。**永久放弃**保证不会在后续步骤重新寻找候选，不会出现“延迟到下一次”的额外交换。
5. 没有合法候选（`none`）时三臂完全一致，任何条件策略都等于 `base`。
6. 因此拼接精确。□

**前提**：同一固定 30 题、模板、种子、检查点；`base` 与 noedit 的确定性复现；三臂共享锚点前状态和锚点计划；条件策略的定义就是“匹配时执行 rank 的同一交换，不匹配时执行 base 的零预留策略”。

## 3. 结果

### LLaDA 2.1，名义 Q4

| 策略 | 干预/30 | 对题/30 | TPF | Token efficiency | F | Q | 实际 Q/F | 相对 noedit 净差 | 赢/输 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `noedit` | 0 | 25 | 3.338 | 19.97% | 5,993 | 72,667 | 12.13 | — | — |
| `uncond_rank` | 28 | 25 | 3.334 | 19.92% | 5,965 | 72,363 | 12.13 | 0 | 0/0 |
| `cond_backfill_rank` | 28 | 25 | 3.334 | 19.92% | 5,965 | 72,363 | 12.13 | 0 | 0/0 |
| `cond_mask_rank` | 0 | 25 | 3.338 | 19.97% | 5,993 | 72,667 | 12.13 | 0 | 0/0 |

28 个锚点**全部是 slack**，没有 mask 锚点；因此 `cond_mask_rank` 从不干预，精确等于 `noedit`；`cond_backfill_rank` 精确等于 `uncond_rank`。2 题无机会（`none`）。无条件/回填条件有 2 题输出 token 序列改变（math500/test/number_theory/1090、HumanEval/74），判分不变，0 赢 0 输。

### DiffusionGemma，名义 Q32

| 策略 | 干预/30 | 对题/30 | TPF | Token efficiency | F | Q | 实际 Q/F | 相对 noedit 净差 | 赢/输 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `noedit` | 0 | 28 | 7.980 | 13.00% | 1,389 | 81,858 | 58.93 | — | — |
| `uncond_rank` | 26 | 28 | 8.165 | 13.05% | 1,321 | 79,380 | 60.09 | 0 | 1/1 |
| `cond_backfill_rank` | 17 | 27 | 8.015 | 12.91% | 1,363 | 81,109 | 59.51 | −1 | 0/1 |
| `cond_mask_rank` | 9 | 29 | 8.130 | 13.15% | 1,347 | 80,129 | 59.49 | +1 | 1/0 |

26 个锚点 = 17 slack + 9 mask，4 题 `none`。`cond_mask_rank` 只在 9 个 mask 锚点换入，29/30（+1），1 赢 0 输；`cond_backfill_rank` 只在 17 个 slack 锚点换入，27/30（−1），0 赢 1 输；`uncond_rank` 两者都做，28/30，1 赢 1 输。净差完全来自两道题：

- **赢**：HumanEval/74（mask 锚点，noedit 0 → rank 1；F −18，Q −494，N 0，TPF +2.5）。
- **输**：math500/test/precalculus/807（slack 锚点，noedit 1 → rank 0；F −18，Q −528，N −21，TPF +1.675）。

**分层机制（Gemma）**：

| 策略 | 分层 | 请求/锚点 | noedit 对题 | 该策略对题 | Δ对题 | TPF noedit→策略 | ΔToken efficiency | ΔQ |
|---|---|---:|---:|---:|---:|---|---:|---:|
| `cond_mask_rank` | mask | 9/9 | 8 | 9 | +1 | 8.111→8.613 | +0.00481 | −1,729 |
| `cond_mask_rank` | slack | 17/17 | 16 | 16（base） | 0 | 8.022→8.022 | 0 | 0 |
| `cond_mask_rank` | none | 4/0 | 4 | 4（base） | 0 | 7.504→7.504 | 0 | 0 |
| `cond_backfill_rank` | mask | 9/9 | 8 | 8（base） | 0 | 8.111→8.111 | 0 | 0 |
| `cond_backfill_rank` | slack | 17/17 | 16 | 15 | −1 | 8.022→8.084 | −0.00147 | −749 |
| `cond_backfill_rank` | none | 4/0 | 4 | 4（base） | 0 | 7.504→7.504 | 0 | 0 |

mask 锚点上 rank 的 9/9 输出都改变，8/9→9/9，Q −1,729（这 9 题上 −6.67%），TPF 8.111→8.613，token efficiency +0.00481；slack 锚点上 16/17→15/17，Q −749，token efficiency −0.00147。**这是分层关联，不是随机分配**：mask/slack 由前置状态决定，不能据此推断“应多挤占 MASK”。

### 随机候选对照（Gemma）

| 策略 | 干预/30 | 对题/30 | TPF | Token efficiency | F | Q | 实际 Q/F | 相对 noedit 净差 | 赢/输 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `uncond_random` | 26 | 28 | 7.986 | 13.02% | 1,350 | 80,132 | 59.36 | 0 | 0/0 |
| `cond_backfill_random` | 17 | 28 | 8.018 | 13.08% | 1,372 | 81,069 | 59.09 | 0 | 0/0 |
| `cond_mask_random` | 9 | 28 | 7.948 | 12.95% | 1,367 | 80,921 | 59.20 | 0 | 0/0 |

三个随机策略都是 28/30、0 净差；在这组同种子、同题单的配对里，随机候选没有复现同样的分数翻转，因此不能把 +1/−1 归因于“只换 mask/backfill”这一条件本身。这只是本次同种子对照的观察，不据此下普遍因果结论。LLaDA 随机对照同样 0 赢 0 输，见 CSV。

## 4. 相对 noedit 的净差与赢输

| 模型 | 策略 | Δ对题 | 赢/输 | ΔF | ΔQ | ΔTPF | ΔToken efficiency |
|---|---|---:|---:|---:|---:|---:|---:|
| LLaDA Q4 | `uncond_rank` | 0 | 0/0 | −28（−0.47%） | −304（−0.42%） | −0.0035（−0.11%） | −0.00046（−0.23%） |
| LLaDA Q4 | `cond_backfill_rank` | 0 | 0/0 | −28（−0.47%） | −304（−0.42%） | −0.0035（−0.11%） | −0.00046（−0.23%） |
| LLaDA Q4 | `cond_mask_rank` | 0 | 0/0 | 0 | 0 | 0 | 0 |
| Gemma Q32 | `uncond_rank` | 0 | 1/1 | −68（−4.90%） | −2,478（−3.03%） | +0.186（+2.33%） | +0.00049（+0.38%） |
| Gemma Q32 | `cond_backfill_rank` | −1 | 0/1 | −26（−1.87%） | −749（−0.91%） | +0.035（+0.44%） | −0.00091（−0.70%） |
| Gemma Q32 | `cond_mask_rank` | +1 | 1/0 | −42（−3.02%） | −1,729（−2.11%） | +0.151（+1.89%） | +0.00141（+1.08%） |

## 5. 限制

- **事后拼接，不是重跑**：条件策略没有单独执行，因此不是独立证据；只有第 2 节前提成立时，拼接才等于条件策略的精确结果。
- **不是每步决策**：只在首个合法候选做一次判断；不匹配就永久放弃，不会在后续步骤重新寻找机会。
- **分层不是随机分配**：mask/slack 由前置状态决定；条件之间的差异是关联，不是因果效应，不能据此推断“应多挤占 MASK”。
- **固定 30 题**：固定模板、顺序、种子、检查点；不能外推到其他题单或在线部署。
- **样本很小**：Gemma 的 +1 只是一道 HumanEval 题；mask 只有 9 个锚点，slack 17 个。
- **完整请求的账**：F/Q 是整条请求的物理工作，不是单步 CUDA 加速；TPF 改善来自完整请求的 N/F，不是单步延迟。

## 6. 文件

- `conditional-requests.csv`：每模型 × 策略 × 请求的拼接指标、来源臂（`base`/`arm`/`noedit`）、是否干预、相对 noedit 的逐题差。
- `conditional-summary.csv`：每模型 × 策略 × 数据集（all/gsm8k/math500/humaneval）的汇总。
- `conditional-vs-noedit.csv`：相对 noedit 的净差、赢/输、绝对与百分比变化。
- `conditional-anchor.csv`：每请求锚点证据（stratum、victim、candidate、base/targeted/random 位置、fresh/masks、pre/post SHA、各臂 executed 与 after SHA、首个 canvas/accepted 差异）。
- `conditional-by-stratum.csv`：按锚点类型（all/slack/mask/none）的汇总。
- `conditional-checks.csv`：可拼接性检查（24 项全部通过）。

## 7. 复算

```sh
cd /data1/project_data/gx-erqw-20261123/runtime/t2t-experiments
/data1/project_data/gx-erqw-20261123/runtime/venv/bin/python -m tools.conditional_seat
/data1/project_data/gx-erqw-20261123/runtime/venv/bin/python -m pytest tests/test_conditional_seat.py -q
```

默认读取 n164 上的 `one-seat-v1-llada21`、`one-seat-v2-gemma`、`hard-v1-llada21`、`hard-v1-gemma`，输出到本目录。

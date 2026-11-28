# 单请求预算实验：共识与交接

工作分支：`BaideBear/BDLM_Serving:T2T_experiments`。从本页恢复，不从某次聊天、临时工作树或占用率截图推断实验状态。前四节是持续有效的约定；末节是带时间戳的交接，不是实时看板。

## 1. 用户定下的目标与边界

| 模型 | 物理块宽 B | 正式实验：整条请求保持不变的 decoder Q 档位 |
|---|---:|---|
| LLaDA 2.1 | 32 | 4、8、16、32、64 |
| DiffusionGemma | 256 | 32、64、128、256、512 |

- 先解决**一个请求在给定 Q 下怎样解码**，同时看质量和 TPF。先对齐同档外部基线，再看增加预算是否带来收益；不能把高 TPF、低质量称为胜出。这个阶段不做 Serving 排布，不把混合预算结果冒充固定档性能。
- 统一的是决策逻辑与约束，不是两模型用相同超参、走相同轨迹或选相同位置。允许对两模型分别搜索。先试每模型一套跨五档的常数；有可重复证据再引入随 `q/B` 变化的简单函数。交付应是一套可解释的算法，不是把十个格子的冠军拼起来。
- “前缀”是当前正在处理的**严格物理块内**、首个未解位置之前的连续已解部分。没被选为 Q 不等于删除其驻留 KV。不以预测 T2T 为目标，不假设上一轮有全位置 logits；考察的是重访哪一段前缀、花多少行、是否值得挤占 MASK 的位置。
- 固定档完成后再探索无损切档和随状态变化的 `Q → TPF` 曲线。外层给预算，单请求不能预知下一档。先大后小只是待测路径，不能要求 Gemma 从 Q256 起步。Triton/FA 的墙钟对齐不是当前验收目标。

## 2. 当前候选与实验顺序（待验证）

普通轮先按预算偿还仍有效的 token 写入或 T2T 造成的 KV 欠账，再给未解位置和已解前缀分 Q。前缀份额 `α` 是**剩余预算**的上限，不是总 Q 的实测比例。块内双向；前块不能读后块，后块可读前块。当前块充分建立且未解比例降到 `τ` 时可给相邻后块分配位置；`Q=2B` 可以从首轮覆盖两块。允许空位，不为填满预算重复投影干净位置。切换 Q 不丢欠账、不重置 KV、不追加隐形 forward。模型必要的验证、重建和 encoder 要如实记账。[精确执行顺序与模型差异](shared-request-planner.md) · [选行源码](../bdlm_serving/methods/t2t/unified_rows.py)。

1. 先核实物理选行、因果方向、终止与计费，再建立两模型的共同参照点 `α=1/8, τ=1/4`。这是工程对照，不是最优结论。
2. 各模型独立搜索 `α, τ ∈ {0, 1/8, 1/4, 1/2}`，一次只变一项。比较基础规则（欠账 + MASK）、只加前缀、只加后块、两者都加；在 `2B` 要另设真正的单块对照：`τ=0` 不关闭 `2B` 的后块。记录欠账、前缀、MASK 的实际份额，后块哪些工作进入最终输出、哪些作废。[搜索登记](../experiments/t2t/serving_slots/controls/shared-request/search-space.json)。
3. 候选值得继续迭代时，冻结源码、两套参数、权重、题单、种子、评分器、2048 生成上限。每次应有 **2 模型 × 5 档 × 90 题 = 900 条**。同一题单、不换难例、不删超时题、不拼不同注册号。算法不收敛记错、保留 F/Q；基础设施、选行或计费错误不计成绩，修复后复验同一输入。原 30 与新增 60 单列；用来调过参的题不是独立泛化证据。

## 3. 怎样读结果

| 指标 | 本轮统一口径 |
|---|---|
| 质量 | 冻结 90 题答对数；同时列原 30、新 60、与同题对照相比的改对/改错。 |
| TPF | 逐请求 `N/F` 的算术平均。`N` 为最终输出的非特殊 token；`F` 为生成阶段实际 forward，包括验证、重建及 Gemma 的生成期 encoder，不含纯 prompt prefill。T2T 不直接多算一个最终 token，其开销不能免计。 |
| Token efficiency | `ΣN / Σ实际选中读取历史 KV 的 decoder Q`，百分数展示。分母不是 `B×F`，不是名义槽位、padding 或注意力 FLOP；encoder 行另报。 |

优先比较同实算上限的 Optimus IBS；原生在 `B`、FlowBlock W1/W2 在对应物理预算作为参照。原版未限实算 Q 的 IBS、30 题旧档和其它名义窗口可以展示，但必须标明口径。图中标具体数值、基线虚线；逐档指出胜、负或取舍。先看实际 N、F、Q，再解释 TPF 上涨是少算了 forward，还是输出变长。

保留的对照：[Live Profile v1 十档数值、实现与参数](live-profile-v1.md) · [冻结 90 题外部基线与差值](../experiments/t2t/serving_slots/controls/fixed-request-comparison/README.md) · [原 30 题基线](../experiments/baselines/README.md) · [FlowBlock PrefixCache](../experiments/flowblock-prefix/README.md) · [Optimus OBS 跨块掩码](../experiments/optimus-masked-obs/README.md)。Live Profile 是可复现的历史方法，不是本候选的成绩。

## 4. 集群交接与发布

| 位置 | 允许事项与红线 |
|---|---|
| n164 H800 物理 0、4 | 实验结束或异常退出后恢复**同一张卡**的 holder 并验证。交接前核对物理 UUID、holder 归属/PID、进程启动身份和设备实际进程；发现其他进程不停止、不挤占。使用 `/data1/project_data/gx-megrez-20261122/gpu-holders/gpu-hold` 的单卡 `status / stop / start`。不要 `stop all`。 |
| A800 物理 0、1、2、3 | 只跑真任务，**不占卡**；结束后释放，不碰其他四张。`/gs` 持久盘仅权重、源码和必要依赖。日志、回执直接流到 n164；需临时文件时用私有、有界 tmpfs，n164 验收后清理。 |
| n164 持久盘 | `/data1/project_data/gx-erqw-20261123` 是结果源。A800 旧数据在 `archive/a800-gs-20261128/{snapshot,delta-final}`。搬迁先验证目标文件哈希，再清理属于本项目且不在用的源文件。 |
| Mac / Git | Mac 不留权重、原始输出、大日志；Git 只归档必要源码、配置、脱敏指标、数值图与绘图脚本。密码、私钥、访问令牌不入库。 |

正式比较用 **2H + 4A 共六卡**，每张固定 15 题（每数据集 5 题，其中原 30 合计 5 题），各档、各对照复用相同分配。支持真正合批时按每批实算 Q 约 512 选 batch，显存不够降低；batch-one 执行器不要报成已经合批。禁止 CPU 模型推理。约 20 分钟检查一次 PID、递增记录、退出状态与回执；不把 holder 的 util 当作正在推理。发现异常只处置本任务，保留现场和已付成本。

正式启动和验收分别见[注册与 worker 命令](../experiments/t2t/serving_slots/controls/shared-request/README.md)、[六卡分片和存储规则](cluster-storage.md)、[计费核对](../experiments/t2t/serving_slots/shared_point_metrics.py)。正常归档继续推送 `T2T_experiments`；**仅**忽略 `Verify frozen results` 失败引发的自动重试、修复及相应提交/推送，不扩大成停掉正常维护。多模型讨论用 GLM-5.4、DeepSeek Flash、GPT-6.2 Sol；设备归属先协调，设计建议要靠同题实验验证。

## 5. 交接记录（北京时间，不代替实时检查）

| 核实时间 | 阶段 | 证据及不可外推的边界 |
|---|---|---|
| 2026-11-28 15:30 | 已封存，旧方案 | LLaDA `covered` 在 Q4/8/16/32/64 的质量相对 Live 为 `72→64、72→65、75→67、75→63、75→63`（各 90 题）。虽然 TPF 上升，质量失败，不替换 Live。[完整记录](../experiments/t2t/serving_slots/controls/coverage-span/README.md) |
| 2026-11-28 15:30 | 资格复验，新方案 | H4 同一题 `gsm8k/167`、seed 17000：EOS 修复后 Q32 答对（N/F/Q=265/45/1181），Q4 跑完但答错（233/199/785）。n164 `runtime/h4-q32-dcaa10b/results-qualify/`。不能当成正式点。 |
| 2026-11-28 15:30 | 参数筛选，新方案 | H0 Gemma 有 60 条回执：3 道题 × 5 档 × 4 个前缀份额，固定 `τ=1/4`。n164 `results/shared-h0-screen-1128-1521/receipts.ndjson` 及 `returned`；是筛选，**不是 60 道独立题**。此时新候选没有已封存的 90 题档。 |
| 2026-11-28 15:38 | 正式参照已启动，尚无封存点 | 冻结[双模型参照配置与源码指纹](../experiments/t2t/serving_slots/controls/shared-request/formal90-anchor.json)，执行源码 `7a38ffc`，注册号 `a18f108aef1dbc607f3b21b2a4d169b04f5ae5dab01be400d015d07249d406f3`。n164 `results/formal90-7a38ffc/H0/gemma.ndjson` 已有 4 条，第一条为 Q32、`complete`。源目录 `runtime/formal90-7a38ffc/src`，A800 代码在 `/gs/code/formal90-7a38ffc/src`。继续按同一注册号核对两边回执。 |
| 2026-11-28 15:45 | 六卡实际出数 | `tools.shared_anchor_report` 核验同一注册的 224/900 条，当时六卡分别为 H0 79、H4 15、A0 57、A1 11、A2 50、A3 12。H0 的 Gemma 15 题 × 5 档全部落盘，不等于 Gemma 全 90 完成。 |
| 2026-11-28 | 下一步：独立筛选 `α` | H0 三题小样本 Q32：`α=0→1/8`，质量均 3/3，TPF `3.60→4.46`，TE `11.17%→14.24%`，纯还账 forward 占 decoder 轮次 `41.1%→4.2%`。改变前缀份额也会改变 MASK 接受的节奏，因此不能把差异全归因于新鲜 KV。冻结[18题四份额面板](../experiments/t2t/serving_slots/controls/shared-request/screen18-prefix.json)，同模型跨五档比较，参照构建不变，单独存至 n164 `results/screen18-prefix-2404af23`；不得将旧三题账本拼入本轮。 |

当前计算授权窗口为 **2026-11-28 15:20 至 2026-11-29 09:20（UTC+8）**，是本轮资源保护时间，不是宣布研究完成的期限。恢复时：先读本页和设计协议，核对远端分支、注册 ID 与 n164 账本，再查两边实际进程，按缺失的 `(版本, 模型, Q, task_id)` 续跑。后续登记、资源变化和封存点要附时间与证据，不覆盖旧的否决结论。

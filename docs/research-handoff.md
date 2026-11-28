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

正式启动和验收分别见[注册与 worker 命令](../experiments/t2t/serving_slots/controls/shared-request/README.md)、[六卡分片和存储规则](cluster-storage.md)、[计费核对](../experiments/t2t/serving_slots/shared_point_metrics.py)。正常归档继续推送 `T2T_experiments`；**仅**忽略 `Verify frozen results` 失败引发的自动重试、修复及相应提交/推送，不扩大成停掉正常维护。按 11 月 28 日修订的目标，多模型设计评议分别使用 `glm-5.3-flash`、`deepseek-flash`、`gpt-6.2-sol`，独立回答同一个问题。原 GLM-5.4 的集群交接不算新模型评议；不要因变更评议模型而重启 GPU。建议需用同题实验验证。

## 5. 交接记录（北京时间，不代替实时检查）

| 核实时间 | 阶段 | 证据及不可外推的边界 |
|---|---|---|
| 2026-11-28 15:30 | 已封存，旧方案 | LLaDA `covered` 在 Q4/8/16/32/64 的质量相对 Live 为 `72→64、72→65、75→67、75→63、75→63`（各 90 题）。虽然 TPF 上升，质量失败，不替换 Live。[完整记录](../experiments/t2t/serving_slots/controls/coverage-span/README.md) |
| 2026-11-28 15:30 | 资格复验，新方案 | H4 同一题 `gsm8k/167`、seed 17000：EOS 修复后 Q32 答对（N/F/Q=265/45/1181），Q4 跑完但答错（233/199/785）。n164 `runtime/h4-q32-dcaa10b/results-qualify/`。不能当成正式点。 |
| 2026-11-28 15:30 | 参数筛选，新方案 | H0 Gemma 有 60 条回执：3 道题 × 5 档 × 4 个前缀份额，固定 `τ=1/4`。n164 `results/shared-h0-screen-1128-1521/receipts.ndjson` 及 `returned`；是筛选，**不是 60 道独立题**。此时新候选没有已封存的 90 题档。 |
| 2026-11-28 15:38 | 正式参照已启动，尚无封存点 | 冻结[双模型参照配置与源码指纹](../experiments/t2t/serving_slots/controls/shared-request/formal90-anchor.json)，执行源码 `7a38ffc`，注册号 `a18f108aef1dbc607f3b21b2a4d169b04f5ae5dab01be400d015d07249d406f3`。n164 `results/formal90-7a38ffc/H0/gemma.ndjson` 已有 4 条，第一条为 Q32、`complete`。源目录 `runtime/formal90-7a38ffc/src`，A800 代码在 `/gs/code/formal90-7a38ffc/src`。继续按同一注册号核对两边回执。 |
| 2026-11-28 15:45 | 六卡实际出数 | `tools.shared_anchor_report` 核验同一注册的 224/900 条，当时六卡分别为 H0 79、H4 15、A0 57、A1 11、A2 50、A3 12。H0 的 Gemma 15 题 × 5 档全部落盘，不等于 Gemma 全 90 完成。 |
| 2026-11-28 | 下一步：独立筛选 `α` | H0 三题小样本 Q32：`α=0→1/8`，质量均 3/3，TPF `3.60→4.46`，TE `11.17%→14.24%`，纯还账 forward 占 decoder 轮次 `41.1%→4.2%`。改变前缀份额也会改变 MASK 接受的节奏，因此不能把差异全归因于新鲜 KV。冻结[18题四份额面板](../experiments/t2t/serving_slots/controls/shared-request/screen18-prefix.json)，同模型跨五档比较，参照构建不变，单独存至 n164 `results/screen18-prefix-2404af23`；不得将旧三题账本拼入本轮。 |
| 2026-11-28 16:02 | 正式参照进行中 | n164 逐条审计 **460/900**：H0/H4/A0/A1/A2/A3 分别 `127/53/96/42/89/53`。LLaDA Q4/8/16/32/64 为 `48/47/47/46/47` 条；Gemma 五档各 45 条。没有完成的 90 题档位，不能给出胜负。 |
| 2026-11-28 16:13 | 正式与筛选分开计数 | 正式参照 **676/900**，H0、H4 各 150 条齐全。独立份额筛选的首个 H0 Gemma 分片 60 条经新审计器验收（3 题 × 5 档 × 4 份额）。H0 两模型机制消融已在集群通过不加载模型的预检；须等本卡两模型份额筛选各 60 条及归还凭证通过，才用同卡运行资格测试，不是正式结果。 |
| 2026-11-28 16:22 | H0 紧预算机制消融开始产生配对结果 | 只读 n164 `results/mechanism18-c6275852-H0/`。Gemma 首题 `gsm8k/167`、seed `17000` 三臂均答对；按全 MASK / 留空 / 计算可编辑前缀排序，Q32 的 F 为 `98/97/86`，Q64 为 `59/50/56`。只是一个输入，两档方向不一致；须继续同题配对，不能据此选参或称是 KV 新鲜度的因果收益。 |
| 2026-11-28 16:41 | 双模型固定档参照封存 | 同注册 900/900，[十档表、图、90/原30/新增60与18/72分表](../experiments/t2t/serving_slots/controls/shared-request/formal90/README.md)。共同参照两模型均未同时达到同档 IBS/Flow 的质量与 TPF，不能升为推荐版。LLaDA Q4 前缀 Q=0；与限额 IBS 比改对 1、改错 49。Gemma Q256 为 83/90、13.055，原生 85/90、25.510。 |
| 2026-11-28 16:42 | 2B 资格通过，开始紧预算 MASK 顺序检验 | n164 `results/successor18-785aead4/H0/qualified.json` 验收 12/12，6 个保持原算法的请求均复现正式账单，H0 已按 PID/UUID 归还。随后 H0 接[新登记](../experiments/t2t/serving_slots/controls/shared-request/mask-order18.json)，`results/mask-order18-2a754078-H0/`；只换物理块内 MASK 排序，不修改原生采样、写回、开块和前缀份额。继续逐项验证，不转混合档。 |

当前计算授权窗口为 **2026-11-28 15:20 至 2026-11-29 09:20（UTC+8）**，是本轮资源保护时间，不是宣布研究完成的期限。恢复时：先读本页和设计协议，核对远端分支、注册 ID 与 n164 账本，再查两边实际进程，按缺失的 `(版本, 模型, Q, task_id)` 续跑。后续登记、资源变化和封存点要附时间与证据，不覆盖旧的否决结论。

## 6. 压缩上下文后怎样接手

1. **先查账本，不从零启动。** 仓库首页链接本页，`T2T_experiments` 是项目分支；聊天所在的 `BlockKernel` 不是实验源码。下面是本轮参照的只读核验命令，不会领取 GPU。若更换注册，先查该注册对应的源码和账本，不混用。最终比较还需提供同版本的[基线 CSV](../experiments/t2t/serving_slots/controls/fixed-request-comparison/metrics.csv)；不得因干净部署缺少 CSV 而省略对照。

   ```bash
   ssh n164 'cd /data1/project_data/gx-erqw-20261123/runtime/formal90-7a38ffc && env PYTHONPATH=$PWD/src ../venv/bin/python shared_anchor_report.py ../../results/formal90-7a38ffc registration.json'
   ```

   同目录已备好 `metrics.csv`（SHA-256 `4f5bb4dfdb9065e60a4173551dc33cad20ab380222c94577e239266272fc6568`）；上条命令加 `--baselines metrics.csv` 可核对完整档位的差值。份额筛选从该目录运行 `env PYTHONPATH=$PWD/src ../venv/bin/python -Bm tools.shared_screen_report ../../results/screen18-prefix-2404af23 registration.json screen18-prefix.json`，未满 360 条的模型只显示计数。

2. **再查逐卡交接。** 这轮主任务核对 H0 和总账；H4、A0/A1、A2/A3 分别有独立负责人。正式分片每卡两模型各 75 条，接续筛选须核验这 150 条身份齐全。进程可能正在等待同卡锁；不能因某模型一段时间计数不动就再派一份。先确认谁在运行或排队、账本是否增长、是否仍在授权窗口，再做逐卡交接。H800 结束时恢复 holder，A800 结束时必须释放。
3. **按顺序推进，不让探索回执混入正式结果。** `screen18-prefix` 注册号 `2404af2390fb2fc96e6348f584691c071c77be8c4ee18da3016beec46e51d602`。每模型须收齐 18 题 × 5 档 × 4 份额 = 360 个唯一身份，核对物理 Q 计费后按[预先写定的规则](../experiments/t2t/serving_slots/controls/shared-request/README.md#paired-calibration-separate-from-formal90)选一个跨档 `α`，再登记 `τ` 筛选。筛选用的 18 题与其余 72 题分开报告；其余 72 题也不能自动视为从未参与过历史选模的盲测集。
4. **待验证的机制，不是已证实的结论。** 紧预算先分清两件事：是少选 MASK、减少下一轮还账有用，还是将腾出的座位真的用于计算前缀有用？[三组配对消融协议](../experiments/t2t/serving_slots/controls/shared-request/prefix-mechanism.md)只改变这一项；LLaDA Q4 因取整应完全不受干预，是负对照。协议归档不等于已部署或已取得成绩，不抢占正式参照和已登记筛选。后续再单独验证尾部跨块以及 2B 后块计算的有效产出；固定档结论具备后才开展预算切换和响应曲线实验。

## 7. 正式决策：淘汰本轮参照，不晋升

2026-11-28 16:50（UTC+8）。上述第 2、6 节记录的是原计划。正式参照 `7a38ffc` 已完整测完。与 Live Profile 同档相比，**十档 TPF 和 TE 全部下降，质量九降一平**。不再沿这版启动新一轮全面份额或开块阈值搜索，不以三题资格结果为它翻案。已经登记、在跑的消融保留为归因材料；不和下一版拼数据。

- LLaDA Q4：本次前缀 Q 为零，仍比限额 IBS 净少 48 题。[逐题配对汇总](../experiments/t2t/serving_slots/controls/shared-request/formal90/llada-q4-pairs.json)。优先查最左连续未解窗口、何时允许已投影的行采样、首次与基线发生分歧的那一步。不要把未证明的因果归给跨块或 T2T。
- Gemma Q256：90 题共 2,646 F，其中采样 2,031、纯写回 293、验证 169、重建 74、生成期 encoder 79；原生合计 1,397 F，输出 N 为 31,911 对 31,352。采样轮数本身已经超过原生总 F。即使把纯写回视为免费的，也解释不了差额。该分解是实测分类，不是删除必要工作的反事实性能。
- 下一版从强对照出发：小预算以同实算上限 IBS 的连续前沿为零改动对照，B 先复现原生完整块的一轮采样，2B 对齐 Flow W2。三者是检验点，不是按预算挑三种算法。拟验证的一条共用规则是：**先保留当前解码前沿及必需写回；只把不挤掉这些行的余量用于其他位置**。模型保留原生采样和停止语义。同一外层预算下先做零改动配对（输出 ID、N/F/Q、停止原因）；每次只加前缀重访或后块其中一项。先完成同题小组逐步回放，值得迭代再锁定双模型各五档、完整 90 题。没有这组证据不晋升、不转混合预算。

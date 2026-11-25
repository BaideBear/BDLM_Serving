# 只在值得时分配旧 token 复查席位

同一固定 30 题、同题同种子，真实稀疏 forward。选择器和原生采样器分离；无 logits 指**选位置**不读 logits，不是模型生成 token 不用 logits。所有实际执行的生成阶段都计费，不剔除截断。名义 Q 是选位轮的目标，不是完整请求 `ΣQ/ΣF` 的上限。

| 政策 | 何时计算额外旧行 | 是否可部署 |
|---|---|---|
| `hard_noedit` | 不专门预留；fresh 和 MASK 选完后的空位按位置回填 | 是；同预算内部基线 |
| `hard_slack` | 只对 fresh 和所有 MASK 后的空位按历史邻近写入排序 | 是；不挤占当前 MASK |
| `hard_trigger` | 同上；没有空位时，上轮严格 T2T 周围四格内若有未消化该写入的旧行，最多换入一个 | 是；保留首个 MASK |
| `hard_oracle1` | 在同一前置状态先联合筛，再按真实 Q 的单席换位验证；确认会发生严格 T2T 才换入 | 仅诊断；筛、验两种 forward 全部收费 |

**LLaDA Q16 透视结果。** 联合筛选累计检查 28,648 个未入选旧位置，652 次在**联合宽度**下呈正例；每轮最多取一个再按实际宽度验证，最终 79 次换入全部在真实 forward 上发生严格编辑。这不是对全部潜在换位的穷举，也不能从 652 与 79 直接推算校验通过率。探针的状态、输出都不写回实际请求。

| 同 30 题，Q16 | 质量 | 实测 TPF | 实测 token efficiency | 生成 F | 实际 Q |
|---|---:|---:|---:|---:|---:|
| 无额外预留 | 27/30 | 4.890 | 17.53% | 2,892 | 69,322 |
| 透视并确认一次换位 | 25/30 | 3.719 | 8.92% | 6,766 | 164,584 |

透视组没有独自答对的题，有 2 题只有对照答对。探针本身占 2,569 F 和 69,752 Q；即便将它们假想为免费，保持该策略自己的输出不变，TPF/效率也只有 **4.885 / 15.48%**，不是可部署的观测值。两组实际输出 N 分别为 12,154 与 14,678；正确率差不能仅据总 F 或命中数解释。本结果反驳“只要额外位置真的会编辑就必然提升这 30 题质量”，不排除其他换位、阶段或预算有效。

**读表及复算。** [LLaDA 指标](llada21-online-summary.csv) · [逐请求](llada21-online-requests.csv) · [工作量与内部配对](llada21-paired-cost.csv)。与原始 IBS/OBS 比较时，除了名义窗口须同时查看双方实际 Q/F；不要把免费探针的诊断值与基线性能作对比。[首轮固定分配、随机和原始基线](../hard/README.md) · [离线机会与探针协议](../FRONTIER.md)。

每个策略只有全部 30 题运行完、身份和工作量均通过校验才公开计分。运行参数例：`--arm 16:0:14:hard_slack --arm 16:0:14:hard_trigger --arm 16:0:15:hard_oracle1`（最后一个仅支持 LLaDA）。`tools/t2t_results.py` 从新运行目录生成计分表；再运行 `python -m tools.selection_effect --model llada21 --run RUN --scores experiments/t2t/adaptive/llada21-online-requests.csv --baseline experiments/t2t/hard/llada21-online-requests.csv --out experiments/t2t/adaptive/llada21-paired-cost.csv`。

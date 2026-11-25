# 相同席位，换一批旧位置是否也有效？

本轮比较事先指定的四个名义 Q：LLaDA **4** 与 Gemma **32** 是上一轮筛出的正例，另外两组 LLaDA **16** 与 Gemma **64** 是反例。使用原有 30 题、生成种子、native 资格校验和生成成本口径，原始模型输出与权重留在集群；不以这批题证明泛化。

- `hard_slack_placebo`、`hard_trigger_placebo` 先根据本请求的状态计算原策略会预留几个旧位置，再从**所有**非 fresh 已解析位置里无放回均匀抽取同样数量；生成采样器、MASK 选择、强制刷新和中性回填保持不变。选择器有独立、确定性的 RNG。触发成功时，安慰剂的行可以不在前一轮 T2T 的邻近区域。
- 当两个选择器看到同一个状态时，fresh、实际选位行数、MASK 集合、额外复查席位数严格一致；不要求两条已经分叉的轨迹每轮仍具有相同状态或同样数目的席位。800 个构造状态的合同测试分别核对两种策略。
- 完整分组可以用共享入口重跑，例如 `python -m tools.run_t2t --model gemma --checkpoint MODEL --samples SAMPLES --reference NATIVE --output NEW_RUN --arm 32:0:28:hard_trigger_placebo`。用 `tools/t2t_results.py` 做原有固定判分。

### 物理工作量

[LLaDA 成本对照](llada21-work-envelope.csv) · [Gemma 成本对照](gemma-work-envelope.csv)。以此前两个正例为对象，逐个与原始 IBS/OBS 的每个已归档窗口比较。名义窗口、总物理 Q、`ΣQ/ΣF`、逐请求实际 Q 不超过对方的题数分别报告。不插值，不把不同窗口当成同窗口，也不将行数当成实测耗时。这不是强行给 Q 加上限的实验。

同时满足质量严格更高、TPF 不降、总 Q 和平均 Q/F 都不增加的比较中，LLaDA Q4 空位组仅对原始 IBS16 达成这一条件，而且只有 11/30 题的总 Q 没有增加；Gemma Q32 触发组对任何已测 IBS/OBS 均未满足四项条件。两种方法相对各自**同名义窗口**的改进不能直接外推为相同工作量的收益。

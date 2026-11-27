# Frozen90 图包：块内全算、选行读历史

打开 [LLaDA 2.1](figures/block-history90-llada.png) 和 [DiffusionGemma](figures/block-history90-gemma.png)。横轴是单请求的固定预算 C，三个面板分别是正确题数 / 90、逐题平均 TPF、token efficiency（%）。

**Token efficiency = 100 × 实际解码输出的 token 数之和 / 选中计算前缀 KV 的 Q 行数之和。** 每一轮每个选中的位置计一行，与 KV 长度、层数、注意力头数、物理投影行数无关。只做块内计算或复用上轮历史结果的行不计；同一位置在另一轮重新选中则再计一次。全读收尾、commit 和 IBS 超过名义 C 的轮次都按实际选中数计；不计 prompt prefill 或 Gemma encoder 重编码。分子沿用原评测的有效输出 token 数，T2T 改写不算新输出。先对同一组请求分别求和再相除，不平均逐题百分比。

| 模型 | Optimus IBS 与新方法的固定 C | 原生推理（正确 / 90，宏 TPF，效率） |
|---|---|---|
| LLaDA 2.1 | 4、8、16、32、64 | B32：75，5.677，15.56% |
| DiffusionGemma | 32、64、128、256、512 | B256：85，25.510，9.31% |

| LLaDA 2.1 预算 | 新方法 | Optimus IBS | DiffusionGemma 预算 | 新方法 | Optimus IBS |
|---:|---:|---:|---:|---:|---:|
| 4 | 30.94% | 36.44% | 32 | 13.89% | 16.65% |
| 8 | 29.12% | 28.22% | 64 | 8.29% | 17.78% |
| 16 | 25.34% | 20.51% | 128 | 8.80% | 14.66% |
| 32 | 15.32% | 15.23% | 256 | 9.31% | 11.10% |
| 64 | 15.32% | 14.56% | 512 | 9.31% | 11.10% |

- 绿色实线：本轮新方法；蓝色虚线：原版 Optimus IBS，固定窗口、不硬限制实际更新行数；点线：原生推理单一实测配置，横向延展只是方便读图。各图使用独立的纵轴刻度。
- 新方法每轮都投影整个当前块；稀疏预算只限制常规轮哪些行读取历史，不是总物理 Q。Gemma C256 和 C512 走原生全读路径，和原生点线重合。LLaDA 的全读端点是分区实现，不等同于原生。两模型的两张图都不是壁钟速度图。
- 对照设备：Gemma IBS C512 用与 C256 匹配的 H800 结果；独立 A800 版本保留在 `controls/block-history90-audited.csv`。LLaDA 图的 IBS C32 是历史 H800，C64 是 A800，同一台 A800 上二者的补充对照见主实验报告。
- 例：Gemma C32 新方法的 90 题共输出 96,795 个 token，选中 696,684 个 Q 读前缀，效率 **13.89%**。原图 **1.93%** 是除以 5,009,152 个物理 Q 的结果，仍可用于描述物理开销，但不是这张图要比较的指标。质量仍是 30/90，宏 TPF 仍是 4.433。
- [完整汇总](controls/block-history90-matrix.csv) 包含 32 配置 × 3 分组。图中 22 个配置的 `token_efficiency = N / prefix_kv_Q`（CSV 中为比例）；旧指标保留为 `physical_token_efficiency = N / physical_Q`。FlowBlock 和 B3 不在本图中，未核对选行计数，两个选行口径的字段留空，不以物理 Q 替代。旧的分类 CSV 原样保留，其 `token_efficiency` 若存在仍是物理口径。`original30` 是原来的 30 题，`added60` 是新增 60 题；图只用 `all90`。宏 TPF 是逐题 `N/F` 的算术平均。
- [绘图所用分子与分母](controls/block-history90-prefix-kv.csv) 来自冻结逐请求记录；[计数脚本](audit_block_history90_prefix_q.py) 用于有原始归档的环境。汇总与原实验逐组核对了样本数、输出 token 数、forward 数及物理 Q，无需重跑推理。

装有 Matplotlib 的 Python 3.9+ 环境下，在解压目录运行 `python plot_block_history90.py` 即可重画两张图；无需模型、GPU 或原始测试输出。运行 `python build_block_history90_matrix.py` 可从随包的分类 CSV 重建总表。

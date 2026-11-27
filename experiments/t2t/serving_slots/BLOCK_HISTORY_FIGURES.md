# Frozen90 图包：块内全算、选行读历史

打开 `figures/block-history90-llada.png` 和 `figures/block-history90-gemma.png`。每张图的横轴是每个请求的固定预算 C，三个纵轴分别为正确题数 / 90、逐题平均 TPF、物理 token efficiency（%）。

| 模型 | Optimus IBS 与新方法的固定 C | 原生推理（正确 / 90，宏 TPF，效率） |
|---|---|---|
| LLaDA 2.1 | 4、8、16、32、64 | B32：75，5.677，15.56% |
| DiffusionGemma | 32、64、128、256、512 | B256：85，25.510，8.77% |

- 绿色实线：本轮新方法；蓝色虚线：原版 Optimus IBS，固定窗口、不硬限制实际更新行数；点线：原生推理单一实测配置，横向延展只是方便读图。各图使用独立的纵轴刻度。
- 新方法每轮都投影整个当前块；稀疏预算只限制常规轮哪些行读取历史，不是总物理 Q。Gemma C256 和 C512 走原生全读路径，和原生点线重合。LLaDA 的全读端点是分区实现，不等同于原生。两模型的两张图都不是壁钟速度图。
- 对照设备：Gemma IBS C512 用与 C256 匹配的 H800 结果；独立 A800 版本保留在 `controls/block-history90-audited.csv`。LLaDA 图的 IBS C32 是历史 H800，C64 是 A800，同一台 A800 上二者的补充对照见主实验报告。
- `controls/block-history90-matrix.csv` 是两模型的完整汇总，包含新方法、原生、IBS、FlowBlock W1–4 和 B3，共 32 配置 × 3 分组。`original30` 是原来的 30 题，`added60` 是新增 60 题；图只使用 `all90`。TPF 为逐题 `N/F` 的算术平均；效率为 `ΣN/Σ物理Q`，不是逐题百分比的平均。

装有 Matplotlib 的 Python 3.9+ 环境下，在解压目录运行 `python plot_block_history90.py` 即可重画两张图；无需模型、GPU 或原始测试输出。运行 `python build_block_history90_matrix.py` 可从随包的分类 CSV 重建总表。

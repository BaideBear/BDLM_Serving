# 上游与本地实现

| 上游 | 使用的提交 | 实际调用 |
|---|---|---|
| [FlowBlock](https://github.com/Red-EAD/FlowBlock/tree/8f730a2173140792a4324736efdcba27a2bdee75) | `8f730a2173140792a4324736efdcba27a2bdee75` | `dInfer/python/dinfer/decoding/{flowblock_packed,generate_uniform,utils}.py` |
| [Optimus](https://github.com/dubcyfor3/Optimus/tree/7343a37b2a69337caa452f17b59727dc6665f039) | `7343a37b2a69337caa452f17b59727dc6665f039` | `engine/lmdeploy/pytorch/strategies/dllm/{sequence,unmasking}.py` |

两个目录是 Git submodule，保留各自完整源码树和上游 `LICENSE`。未递归克隆时运行 `git submodule update --init`。本地适配器从锁定源码中装载指定类/函数，核对 [5 个文件的 SHA-256](controls.json)，不要求安装整个 dInfer 或 LMDeploy。SDAR FlowBlock 的 EOS 检查从生成段开始；LLaDA Optimus 在原选中行上做 T2T。它们是统一模型后端上的适配，不是与上游全栈逐 token 等价的声明。

B3 没有作为第三个上游仓库引入。[本地实现](../bdlm_serving/methods/b3/)是按 [B3 论文](https://arxiv.org/abs/2611.14387)推导的 B32 控制：空候选集合不开启 future；晋升清除 previous entropy，LLaDA 传递 post 计数。SDAR 是本地迁移。相关论文：[FlowBlock](https://arxiv.org/abs/2607.17652)、[Optimus](https://arxiv.org/abs/2605.24832)。

本地包与历史实验模块的关系见[源码来源表](../docs/controller-provenance.json)。其中是迁移前模块的指纹，不是重排后文件的哈希。上游源码、模型和评测数据的许可分别适用。

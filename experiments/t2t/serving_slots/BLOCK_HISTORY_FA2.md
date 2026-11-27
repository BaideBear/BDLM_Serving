# 块内全算、按 Q 行读历史：FA2 算子对照

`B` 为物理块长，`H` 为本层可见的历史长度，`C` 为每请求读历史的名义 Q 数。本测试的输入无 padding、所有给出的 KV 都可见，输入 FP16/BF16，性能表为 BF16。不是端到端模型推理；不使用 30/90 题成绩判断内核快慢。

| 实现 | 当前块（每头每请求） | 历史（每头每请求） |
|---|---:|---:|
| 官方 `flash_attn_func`，`causal=False` | `B²` | `B×H` |
| 本算子，`C<B` | `B²` | `C×H` |
| 本算子，`C≥B` | `B²` | `B×H` |

FA2 侧把历史与当前块的 K、V 预先拼好，没有读槽位、稀疏掩码、窗口裁剪或旧 attention 状态复用。两边同一测试点的 batch、head_dim、GQA 和输入相同。`C<B` 时快过 FA2 只说明这项稀疏工作负载可能省时；判断相同注意力计算量的性能看 `C≥B`。即便注意力边数相等，本算子仍额外保存历史状态，属于本方法必须承担的成本。`C=2B` 单块饱和，并非两块同时解码。每档 batch=`512/C`，所以不同档位的 FA2 时间不必相等。

测量：[脚本](../../../tools/bench_block_history_fa2.py)，预热和 CUDA Graph 捕获之后，交错执行 15 组、每组 30 次，以 CUDA events 测组均值的中位数。计入本地、历史、FP32 状态写回与合并；预先分配历史缓存、初始化、选行、布局转换和 QKV 投影不计入。测试仅代表热缓存、已打包输入，不代表冷启动、滑窗、混长 batch 或整个模型的延迟。`D=512` 不填 FA2 比值，所测官方 CUDA FA2 接口不支持该头宽。

数值：[Triton 源码](../../../bdlm_serving/methods/t2t/history_attention.py) 对两段和旧缓存使用 FP32 `m/l/u`，FP32 重标定、合并、归一化；权重在乘 V 前转模型 dtype，Tensor Core 点积 FP32 累加，最终输出恢复模型 dtype。参考官方 [kernel traits](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/csrc/flash_attn/src/kernel_traits.h)、[softmax](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/csrc/flash_attn/src/softmax.h)、[forward](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/csrc/flash_attn/src/flash_fwd_kernel.h)。精度策略相同不保证逐位相同；复用旧历史是另外一项算法近似。[GPU 检查](../../../tests/check_history_attention_gpu.py) 对照 FP32 参考，检查缓存值、未选行保持不变、GQA、空分区、变化的 Q 和逐行 KV 范围。

### 实测：全读 C=B

2026-11-27，单位 µs。H800 PCIe（FA2 2.8.3.post2）；A800 SXM4（FA2 2.5.7）。同卡内比较。全读路径先写块内状态，再于历史内核内写历史缓存并融合最终合并；对显式逐行边界保留掩码。

| 模型形状 | batch | 历史 | H800 本算子 / FA2 | A800 本算子 / FA2 |
|---|---:|---:|---:|---:|
| LLaDA B32, D128 | 16 | 128 | 20.61 / 21.93 | 32.63 / 31.33 |
| LLaDA B32, D128 | 16 | 512 | 40.64 / 51.57 | 63.93 / 67.82 |
| Gemma B256, D256 | 2 | 128 | 57.92 / 44.70 | 81.00 / 66.56 |
| Gemma B256, D256 | 2 | 512 | 94.18 / 60.71 | 118.78 / 87.69 |

其中 LLaDA 在上述范围接近或快于 FA2；Gemma D256 仍落后。不能据此宣称所有预算、不同 batch 或 Gemma 全模型达到 FA2 性能。原始逐档 JSONL 在 n164 的 `runtime/block-history-1127/{bench-h800-fused.jsonl,a800-evidence/bench-a800-fused.jsonl}`，包括 `C=2B` 和仅作绝对计时的 D512。

# 单步改变预算，随后回到同一档：首批 GPU 实测

这是[响应预测协议](../../../../../docs/request-q-response.md)的 **Live strict 迁移对照**，不是全块 admission 新版本，不是已部署的预测结果。这里固定公开同一题的前两个登记状态；不给同一题的十条轨迹十票。

LLaDA 2.1，A800，`gsm8k/167`，seed 17000。先用 Q16 重放到指定真实状态，仅改一次预算，然后五臂都恢复 Q16 直至 EOS。每个起点都比较 token 画布、所有层的 KV、驻留集合、控制状态、随机数、历史与不可撤销输出摘要；五臂相同。五档均答对，未截断。第一步实际 decoder Q 和**随后整段**有效 TPF 是不同的量。

| 决策前状态 | Q4 | Q8 | Q16 | Q32 | Q64 |
|---|---:|---:|---:|---:|---:|
| 状态 A：26 个 MASK、6 行欠刷新、16 行驻留；后缀 TPF | 3.616 | 3.049 | 3.158 | 3.370 | 1.633 |
| A：本步实际 Q | 4 | 8 | 16 | 22 | 32 |
| A：后缀 F | 73 | 102 | 95 | 81 | 218 |
| 状态 B：12 个 MASK、1 行欠刷新、27 行驻留；后缀 TPF | 1.955 | 3.573 | 3.573 | 3.181 | 3.986 |
| B：本步实际 Q | 4 | 6 | 6 | 6 | 32 |
| B：后缀 F | 132 | 75 | 75 | 94 | 69 |

A 在原轨迹的第 5 步后，已不可撤销交付 6 个有效 token；B 在第 25 步后，已交付 38 个。这两个步数只用于重放定位，不进入预测特征。Q16 保持不变时两个分叉终点都是相同的 306 个有效 token。Q16→Q64 相对保持 Q16，在 A 降低后缀 TPF，在 B 提高后缀 TPF。它说明状态可能改变预算排序；不能从中单独归因给 MASK、欠账或驻留行数，不能证明 Q64 一般更好或更坏，也不能用它评估在其他题上的预测误差。

[紧凑的实测输入](llada-first-two.jsonl)包含条件、每个分叉的实际 Q/F、终点有效长度、得分和内容摘要；不含题目、生成文本或原始 KV。重现：在 n164 的 `results/llada-response-forks-a1/migrated-live-strict-frontier32/attempt-20261128T155055Z-live/`，用 `python -m tools.llada_response_seals worker.ndjson --limit 2` 提取。`python -m tools.request_response_eval experiments/t2t/serving_slots/controls/request-q-response/llada-first-two.jsonl` 给出 2 个决策点、1 道独立题、0 个有训练支持的留题预测；此时报告 MAE 或“胜率”没有意义。

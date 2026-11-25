# LLaDA 2.1：交换是否需要预测出的候选？

本轮在 Q4/1（主预算）和 Q8/1（预定负对照）上比较历史 4x 交换 `hard_exchange_edit4`（ranked）与两个新的候选随机化 placebo。两个 placebo 保持 4x gate/victim、换位数和必刷 fresh 不变，只把换入候选改为从合格旧位中均匀随机抽取；它们是两次独立的 30 题执行，不能合并成 60 题。零预留为同预算对照，来源逐臂列在下表。所有臂使用同一 frozen30（GSM8K、MATH-500、HumanEval 各 10 题）。

## 状态

完整：2 个预算 × 4 个臂 × 30 题 = 240 条请求明细；每个臂的 30 个身份与 frozen30 一致，requests 与 summary 表交叉核对通过。以下原始事件检查不在本报告计算，由独立原始事件审计负责：全宽 parity、gate/victim 匹配、首次物理 Q 分叉、换位类型、新增行的严格改写。

## 实测臂

| Q | 臂 | policy | 来源 | 答对/30 | 失败 | 截断 | macro TPF | 物理 Q | 物理 efficiency | token efficiency |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | zero reserve | `t2t_q4_f0_m4_hard_noedit` | hard | 25 | 0 | 1 | 3.338 | 72,667 | 19.97% | 19.97% |
| 4 | ranked edit4 | `t2t_q4_f0_m3_hard_exchange_edit4` | ranked | 26 | 0 | 1 | 3.424 | 70,171 | 19.87% | 19.87% |
| 4 | placebo a | `t2t_q4_f0_m3_hard_exchange_random_a` | measured | 23 | 0 | 2 | 3.385 | 72,783 | 19.76% | 19.76% |
| 4 | placebo b | `t2t_q4_f0_m3_hard_exchange_random_b` | measured | 25 | 0 | 1 | 3.379 | 67,022 | 19.64% | 19.64% |
| 8 | zero reserve | `t2t_q8_f0_m8_hard_noedit` | hard | 26 | 0 | 1 | 4.201 | 66,515 | 19.27% | 19.27% |
| 8 | ranked edit4 | `t2t_q8_f0_m7_hard_exchange_edit4` | ranked | 22 | 0 | 3 | 4.075 | 86,265 | 18.13% | 18.13% |
| 8 | placebo a | `t2t_q8_f0_m7_hard_exchange_random_a` | measured | 27 | 0 | 1 | 4.141 | 79,612 | 18.26% | 18.26% |
| 8 | placebo b | `t2t_q8_f0_m7_hard_exchange_random_b` | measured | 28 | 0 | 1 | 4.213 | 72,488 | 18.52% | 18.52% |

macro TPF 是逐请求 `N/F` 的均值；物理 efficiency 是 `ΣN/Σ物理 Q`；token efficiency 是 `ΣN/ΣQ`。失败与截断保留在分母中。

## 配对：ranked − 每个参照

| Q | 参照 | 只 ranked 对 | 只参照对 | 都对 | 都错 | Δ答对 | ΔTPF | Δ物理 Q | Δ物理 efficiency |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | zero reserve | 1 | 0 | 25 | 4 | +1 | +0.087 | -2,496 | -0.10 pp |
| 4 | placebo a | 3 | 0 | 23 | 4 | +3 | +0.039 | -2,612 | +0.10 pp |
| 4 | placebo b | 1 | 0 | 25 | 4 | +1 | +0.045 | +3,149 | +0.23 pp |
| 8 | zero reserve | 0 | 4 | 22 | 4 | -4 | -0.126 | +19,750 | -1.14 pp |
| 8 | placebo a | 0 | 5 | 22 | 3 | -5 | -0.066 | +6,653 | -0.13 pp |
| 8 | placebo b | 0 | 6 | 22 | 2 | -6 | -0.137 | +13,777 | -0.40 pp |

ΔTPF 是逐请求 `N/F` 差值的均值；Δ物理 Q 是 30 题 `physical Q` 总和的差。两个 placebo 是独立执行，分别列出，不做合并。

## 上下文：原固定窗口 IBS/OBS

| Q | policy | 答对/30 | macro TPF | 物理 Q | 物理 efficiency |
|---:|---|---:|---:|---:|---:|
| 4 | `ibs_4` | 18 | 1.434 | 43,554 | 36.02% |
| 4 | `obs_4` | 22 | 1.541 | 48,838 | 35.98% |
| 8 | `ibs_8` | 23 | 2.293 | 61,239 | 26.99% |
| 8 | `obs_8` | 21 | 2.337 | 60,808 | 26.84% |

IBS/OBS 使用原固定窗口 gate 和原 seed，不是本轮匹配的 4x gate/victim；它们只用于定位差距，不进入任何配对比较。配对只比较同一 frozen30 上、同一预算的交换臂。

## 范围与限制

- 两个 placebo 是两次独立的 30 题执行，不能当作 60 个独立请求。
- 本报告不重算评分，不读原始生成事件；失败与截断保留在分母中。
- 全宽 parity、gate/victim 匹配、首次分叉、换位类型和新增行严格改写由原始事件审计负责，本报告不声称这些检查已经完成。

## 文件

- `exchange-placebo-requests.csv`：逐请求明细（4 臂 × 2 预算 × 30 题）。
- `exchange-placebo-summary.csv`：逐臂汇总。
- `exchange-placebo-paired.csv`：ranked 对每个参照的配对汇总。
- `exchange-placebo-paired-requests.csv`：逐请求配对差值。
- `exchange-placebo-context.csv`：原固定窗口 IBS/OBS 上下文（不配对）。
- `exchange-placebo-checks.csv`：数据完整性与分离检查。
- `exchange-placebo-study.svg`：四臂与 IBS/OBS 的数值图。
- `exchange-placebo-delta.svg`：ranked − 参照的配对差值图。
- `exchange-placebo-report.md`：本文件。

## 复现

```bash
python -m tools.exchange_placebo_report --measured <measured-dir> --ranked <ranked-dir> --out <out-dir>
```

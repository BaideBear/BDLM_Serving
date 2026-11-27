# LLaDA 2.1: head coverage, successor admission and borrowed rows

Frozen task/seed, identical prompt digest and **same physical device per
pair**, no sample-specific exceptions. The scope is the `llada-cross-race-v2`
snapshot with retirement `all_resident_no_masks_no_token_debt`. Not the warm-head
resampling experiment, not native generation, not the older progress80 policy.
The [aggregate JSON](paired.json) publishes metrics only when every request
in a cohort exists; unfinished cohorts report counts only.

| Fixed offer and one change, original 30 | Correct | Macro TPF | Selected-Q efficiency |
|---|---:|---:|---:|
| Q8, borrow off → on | 11 → 10 | 1.635 → 1.577 | 16.28% → 16.27% |
| Q16, borrow off → on | 14 → 20 | 2.495 → 2.861 | 12.06% → 16.75% |
| Q64, no successor, sparse → full head | 24 → 25 | 4.187 → 4.948 | 15.14% → 11.75% |
| Q64, full head, off → open immediately | 25 → 17 | 4.948 → 5.494 | 11.75% → 5.94% |
| Q64, full head, off → open at half | 25 → 19 | 4.948 → 5.293 | 11.75% → 5.48% |
| Q64, full head, off → open at quarter | 25 → 24 | 4.948 → 5.643 | 11.75% → 8.64% |

Small-offer borrowing only projects the successor's rows, without sampling
them on that borrowed call. It rescued 7, lost 1 at Q16; at Q8 it rescued
5 and lost 6. The Q16 result remains behind historical IBS (24/30, 3.173).
At Q64 the quarter gate has the highest TPF of the four full-head gates on
these 30 cases, but still trails historical W2 (23/30, 9.721) on TPF and
native B32 (26/30, 5.728) on both objectives. External baselines come from
[`block-history90-matrix.csv`](../block-history90-matrix.csv), not the paired
GPU execution path. Historical W2 has no comparable TE in that table.

Rebuild on the GPU host with `llada_gate_cohorts.py /gs/results/llada-hfull-paired
benchmarks/frozen90/manifest.json`. The script validates identity, all 20
observed projection widths, head/tail/read-only row totals and per-pair device.
Only its aggregate JSON should leave the host. No model execution is needed.

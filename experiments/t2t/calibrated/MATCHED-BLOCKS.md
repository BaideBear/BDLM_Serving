# LLaDA 2.1 Q16 matched-block F diagnostic

**Date:** 2026-11-26
**Execution:** n164 CPU only (`CUDA_VISIBLE_DEVICES=`); no new inference.
**Control:** `hard-v1-llada21` / `t2t_q16_f0_m16_hard_noedit`
**Arms:** `fixed` = `hazard24-v1-llada21` / `t2t_q16_f0_m14_hard_hazard_fixed`, `gate` = `hazard24-v1-llada21` / `t2t_q16_f0_m14_hard_hazard_gate`, `geometry` = `geometry-v1-llada21` / `t2t_q16_f0_m14_hard_geometry`, `slack_geometry` = `geometry-v1-llada21` / `t2t_q16_f0_m14_hard_slack_geometry`

Descriptive only: a matched block ordinal is the same position in the same prompt, not the same model state. Policies can produce different text; no comparison here is causal.

## Trace semantics

- Actual `prompt_length` comes from the frozen native traces (`*.trace.jsonl.gz`), not from `prefill_rows`.
- `prefill_rows == 32 * floor(prompt_length/32)`. The prompt-mixed first block starts at `prefill_rows` and is excluded.
- A valid full output block has `q_start >= prompt_length` and `q_end <= prompt_length + len(output_ids)`. A final partial block is excluded; a full block ending exactly at EOS remains valid. `K = min(K_control, K_arm)` per pair.
- All 30 prompts have `prompt_length % 32 != 0`; every request has exactly one prompt-mixed first block, which is excluded.
- `output_ids` includes generated positions including EOS; `output_tokens` excludes special tokens.
- Bootstrap and commit are one forward per block; at fixed K their matched delta is exactly zero and is checked per pair.

## Results

| arm | matched blocks | excess / 30 | progress_only | both | length_only | equal | arm_lower | total dF | matched dF | remainder dF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | 344 | 21 | 8 | 9 | 4 | 3 | 6 | +266 | +84 | +182 |
| gate | 345 | 18 | 7 | 7 | 4 | 6 | 6 | +1085 | +168 | +917 |
| geometry | 344 | 20 | 8 | 7 | 5 | 3 | 7 | +502 | +126 | +376 |
| slack_geometry | 343 | 19 | 6 | 11 | 2 | 5 | 6 | +3290 | +204 | +3086 |

| arm | phase | total dF | matched dF | remainder dF | excess-only total dF | excess-only matched dF | excess-only remainder dF |
|---|---|---:|---:|---:|---:|---:|---:|
| fixed | bootstrap | +25 | +0 | +25 | +24 | +0 | +24 |
| fixed | selected | +198 | +97 | +101 | +342 | +238 | +104 |
| fixed | verify | +18 | -13 | +31 | +47 | +13 | +34 |
| fixed | commit | +25 | +0 | +25 | +24 | +0 | +24 |
| gate | bootstrap | +89 | +0 | +89 | +95 | +0 | +95 |
| gate | selected | +784 | +171 | +613 | +841 | +194 | +647 |
| gate | verify | +123 | -3 | +126 | +147 | +10 | +137 |
| gate | commit | +89 | +0 | +89 | +95 | +0 | +95 |
| geometry | bootstrap | +41 | +0 | +41 | +50 | +0 | +50 |
| geometry | selected | +339 | +86 | +253 | +507 | +221 | +286 |
| geometry | verify | +81 | +40 | +41 | +89 | +33 | +56 |
| geometry | commit | +41 | +0 | +41 | +50 | +0 | +50 |
| slack_geometry | bootstrap | +172 | +0 | +172 | +182 | +0 | +182 |
| slack_geometry | selected | +2683 | +153 | +2530 | +2831 | +268 | +2563 |
| slack_geometry | verify | +263 | +51 | +212 | +282 | +57 | +225 |
| slack_geometry | commit | +172 | +0 | +172 | +182 | +0 | +182 |

| arm | selected gt/eq/lt | verify gt/eq/lt | matched selected ctrl->arm | matched verify ctrl->arm | same-token blocks |
|---|---|---|---|---|---:|
| fixed | 119/125/100 | 53/237/54 | 1397->1494 | 464->451 | 85/344 |
| gate | 97/149/99 | 51/241/53 | 1405->1576 | 465->462 | 116/345 |
| geometry | 111/129/104 | 63/222/59 | 1397->1483 | 464->504 | 86/344 |
| slack_geometry | 99/158/86 | 62/232/49 | 1395->1548 | 462->513 | 125/343 |

## Checks

- 30/30 request traces per arm; all `status=success`; native prompt traces cover all 30 requests.
- Every event has `F=1`, `actual_q=physical_q`, uniform `layer_physical_q`, `q_tokens=actual_q=len(absolute_positions)`, strictly increasing positions; one bootstrap and one commit per block.
- Aggregate identities hold: `generation_forwards`, `forwards_by_phase`, `q_by_phase`, `prefill_rows`, `generation_query_rows`, and `prefill_rows == 32*floor(prompt_length/32)`.
- Valid output blocks are contiguous and start at the first prompt-free block; the terminal partial block is excluded.
- All 30 prompts have a non-multiple-of-32 length; exactly one prompt-mixed block per request is excluded.
- Control request 17 is the only exact EOS boundary case (prompt 87, 1161 generated positions, end 1248 = 39*32); its final block is full and valid. All other requests have a partial terminal block.
- Matched bootstrap/commit deltas are zero for all 120 pair rows; matched + remainder equals excess for all rows.
- Matched pairs share `(index, dataset, task_id)`, `prefill_rows`, `seed`, model and `max_q`; every matched block has identical absolute `q_start/q_end`.
- Input SHA-256 values are recorded in `llada21-matched-block-summary.csv`.
- Unit tests: `6 tests passed (unittest discover -s tests -p test_matched_work.py)`
- All measurements reuse the archived GPU runs.

## Files

- `experiments/t2t/calibrated/llada21-matched-block-requests.csv`
- `experiments/t2t/calibrated/llada21-matched-block-blocks.csv`
- `experiments/t2t/calibrated/llada21-matched-block-summary.csv`
- `experiments/t2t/calibrated/llada21-matched-block-phases.csv`
- `experiments/t2t/calibrated/MATCHED-BLOCKS.md`

## Limitations

- Descriptive only, not causal. A shared block ordinal is a shared prompt position, not a shared model state; text diverges in most pairs.
- Q16 LLaDA 2.1-mini, 30 frozen requests, one seed; no quality, TPF or wall-clock claim.
- The remainder contains extra complete blocks and the partial terminal block; it is length-related but not a pure token-length effect.
- `output_tokens` excludes EOS; block validity uses positions including EOS. A full block ending exactly at EOS remains valid.

# Where can full first admission change the strict Gemma parent?

Fresh requests, 2048-token cap, same registered 90 cases and seeds at each endpoint. This is an observation of the strict parent (`dense_stop_joint_verified_eos`, source `7f160c54fa1fcd005aa92f9420c700f5df4cb47f`). The probe stops after observing the **first real projection into the second physical canvas**, or after the parent terminates. It does not measure subsequent admissions, generation quality, or whole-request TPF.

| Fixed offer | Board | Cohort | No second canvas projected | Partial first projection | All 256 projected |
|---:|---|---|---:|---:|---:|
| 32 | A0 | original 30 | 11 | 19 | 0 |
| 32 | A0 | additional 60 | 24 | 36 | 0 |
| 32 | A0 | all 90 | 35 | 55 | 0 |
| 512 | A2 | original 30 | 13 | 0 | 17 |
| 512 | A2 | additional 60 | 18 | 0 | 42 |
| 512 | A2 | all 90 | 31 | 0 | 59 |

All 55 partial first projections at Q32 computed exactly 32 new-canvas rows. Adding the rest changes this first call by 224 rows for an exposed request, **not** necessarily its net Q cost to termination. On Q512 no observed first successor lacked any rows. Check final output, encoder re-entry, and subsequent admissions on a real same-device candidate before claiming equivalence or reusing a score.

Both GPUs used Torch 2.6.0+cu124, Transformers 5.12.0, bf16, SDPA, `grouped_mm`, and checked actual Q/K/V projection widths against decoder positions. Both the prespecified pilot (#7/13/25) and the exact 87-case complement exited cleanly on each board. Setup attempts with missing manifests failed before measuring any case; they contribute nothing. Only code and immutable inputs were staged in temporary RAM; raw records reside on n164 under `results/gemma-parent-first-projection-7f160c5/`. The compact summaries preserve exact attempt names and record hashes.

[Q32](q32.json) / [Q512](q512.json) / [GPU probe](../../../../../tools/gemma_parent_first_projection.py) / [checked aggregator](../../../../../tools/gemma_parent_exposure_summary.py).

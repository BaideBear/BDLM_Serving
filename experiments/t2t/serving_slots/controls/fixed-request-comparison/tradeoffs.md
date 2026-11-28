# Fixed-budget decisions against baselines

Each entry compares the published live request to one measured baseline on the same frozen tasks. Use hard-capped IBS below 2B, plus native and Flow at their actual block-size endpoints. Uncapped nominal windows are shown separately in the [full table](README.md).

The deltas are live minus baseline, never vs a composite of different baselines. Correct deltas are whole questions (not percent); TE deltas are percentage points. 'Both at least as high' is a point estimate, not a noninferiority test. The original and additional tasks are disjoint, not two new independent replications.

| Model | Q | Baseline | Correct delta (30 / 60 / 90) | Macro TPF delta (30 / 60 / 90) | TE delta (90, pp) | Both at least as high (30 / 60 / 90) |
|---|---:|---|---:|---:|---:|---|
| llada21 | 4 | IBS capped | +2 / +1 / +3 | +0.107 / +0.109 / +0.109 | +2.02 | yes / yes / yes |
| llada21 | 8 | IBS capped | -1 / +1 / +0 | -0.055 / -0.069 / -0.064 | +0.58 | no / no / no |
| llada21 | 16 | IBS capped | +2 / +1 / +3 | -0.006 / -0.025 / -0.019 | +5.06 | no / no / no |
| llada21 | 32 | IBS capped | +1 / +3 / +4 | +1.366 / +1.264 / +1.298 | +2.82 | yes / yes / yes |
| llada21 | 32 | Native | +0 / +0 / +0 | -0.091 / +0.219 / +0.116 | +2.50 | no / yes / yes |
| llada21 | 32 | Flow W1 | +0 / +0 / +0 | -1.406 / -1.275 / -1.319 | +0.76 | no / no / no |
| llada21 | 64 | Flow W2 | +5 / -4 / +1 | -2.450 / -2.276 / -2.334 | -0.73 | no / no / no |
| gemma | 32 | IBS capped | +1 / +5 / +6 | -1.912 / -1.986 / -1.961 | -3.70 | no / no / no |
| gemma | 64 | IBS capped | +1 / +1 / +2 | -2.888 / -2.867 / -2.874 | -3.69 | no / no / no |
| gemma | 128 | IBS capped | +5 / +4 / +9 | -4.046 / -3.019 / -3.361 | -2.72 | no / no / no |
| gemma | 256 | IBS capped | +1 / +3 / +4 | +0.213 / +0.788 / +0.596 | -2.41 | yes / yes / yes |
| gemma | 256 | Native | +0 / -2 / -2 | -1.025 / -1.374 / -1.258 | -0.62 | no / no / no |
| gemma | 256 | Flow W1 | +0 / -2 / -2 | -1.025 / -1.374 / -1.258 | -0.62 | no / no / no |
| gemma | 512 | Flow W2 | -1 / +1 / +0 | +1.030 / -0.207 / +0.206 | +2.91 | no / no / yes |

Same offer does not mean the same number of rows over the entire request. Cross-device results are descriptive. See [metrics.csv](metrics.csv) for N, F, selected Q (when verified), source receipts, and the exact series.

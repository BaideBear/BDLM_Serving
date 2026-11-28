# Fixed offers: one executable profile versus measured baselines

Every point is an independent constant-budget request from prompt to final output. Only the published profile is a candidate here. No per-metric winner selection. For each model, Native/W1 are measured at B and W2 at 2B only. [Main figure](frozen90.png) | [Differences against each baseline](tradeoffs.md) | [Experimental variants (separate)](ablations.md).

**Live profile** = `quarter_window_repeat64_paid16_frontier32` for LLaDA, `dense_stop_joint_verified_eos` for Gemma. Values are from the independent fixed90 runs implementing each exact path, not 90 reruns through the switching API. LLaDA uses Active window at Q4/8, Paid MASK at Q16, Paid frontier at Q32, and Paid repeat at Q64. [Switching qualification](../strict-handle/README.md).

Each cell: **correct / macro TPF / token efficiency (%)**. Missed or unavailable metrics are `n/a`. The Q in the first column is a per-request decoder offer, not mean Q/F. Native and Flow are shown only at their actual block-size endpoints.

TPF averages per-request final nonspecial N / all post-prefill F (including Gemma encoder). TE pools N / selected decoder rows reading history KV, not N / decoder-plus-encoder rows. Flow TE is omitted where this selection is not separately verified; it is not zero. For LLaDA W1, the [90-case decoder-call ledger](llada-flow-w1-calls90.json) supplies the measured decoder rows with prefill excluded; the fixed budget is not used as a multiplier. For Gemma W1, [recorded call events](gemma-flow-w1-events90.json) separate the generation decoder rows from the generation encoder rows, with the ordered 90 raw digests matched to the scored cases. W2 uses the [audited decoder ledgers](../llada-flow-active-w2/README.md), with generation encoder F included in TPF and encoder rows excluded from TE. Nominal IBS is a fixed window, not necessarily a per-forward hard cap. At 2B only the native fixed-window IBS run is shown; it saturates at a single physical block. Contrasts with native/IBS/Flow are descriptive on the same frozen cases, sometimes on different GPUs. Differences are not claims of statistical significance.

## frozen90

| Model | Q | Live profile | IBS capped | IBS nominal | Native | Flow W1 | Flow W2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| llada21 | 4 | 72/90 / 1.504 / 37.68% | 69/90 / 1.395 / 35.66% | 59/90 / 1.477 / 36.44% | - | - | - |
| llada21 | 8 | 72/90 / 2.300 / 29.94% | 72/90 / 2.364 / 29.36% | 69/90 / 2.341 / 28.22% | - | - | - |
| llada21 | 16 | 75/90 / 3.356 / 25.60% | 72/90 / 3.375 / 20.54% | 73/90 / 3.371 / 20.51% | - | - | - |
| llada21 | 32 | 75/90 / 5.793 / 18.06% | 71/90 / 4.495 / 15.23% | 71/90 / 4.495 / 15.23% | 75/90 / 5.677 / 15.56% | 75/90 / 7.111 / 17.30% | - |
| llada21 | 64 | 75/90 / 7.672 / 10.87% | - | 72/90 / 4.397 / 14.56% | - | - | 74/90 / 10.006 / 11.61% |
| gemma | 32 | 85/90 / 4.879 / 15.63% | 79/90 / 6.840 / 19.33% | 78/90 / 6.743 / 16.65% | - | - | - |
| gemma | 64 | 83/90 / 8.176 / 13.96% | 81/90 / 11.050 / 17.65% | 80/90 / 11.116 / 17.78% | - | - | - |
| gemma | 128 | 88/90 / 13.047 / 11.97% | 79/90 / 16.408 / 14.69% | 79/90 / 16.418 / 14.66% | - | - | - |
| gemma | 256 | 83/90 / 24.252 / 8.69% | 79/90 / 23.656 / 11.10% | 79/90 / 23.656 / 11.10% | 85/90 / 25.510 / 9.31% | 85/90 / 25.510 / 9.31% | - |
| gemma | 512 | 84/90 / 25.596 / 7.42% | - | 79/90 / 23.656 / 11.10% | - | - | 84/90 / 25.390 / 4.51% |

## original30

| Model | Q | Live profile | IBS capped | IBS nominal | Native | Flow W1 | Flow W2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| llada21 | 4 | 25/30 / 1.489 / 37.45% | 23/30 / 1.382 / 33.42% | 18/30 / 1.434 / 36.02% | - | - | - |
| llada21 | 8 | 24/30 / 2.210 / 29.32% | 25/30 / 2.265 / 28.36% | 23/30 / 2.293 / 26.99% | - | - | - |
| llada21 | 16 | 25/30 / 3.183 / 23.00% | 23/30 / 3.189 / 19.32% | 24/30 / 3.173 / 19.90% | - | - | - |
| llada21 | 32 | 26/30 / 5.637 / 17.65% | 25/30 / 4.271 / 13.34% | 25/30 / 4.271 / 13.34% | 26/30 / 5.728 / 15.55% | 26/30 / 7.042 / 17.87% | - |
| llada21 | 64 | 28/30 / 7.271 / 10.83% | - | 24/30 / 4.228 / 13.74% | - | - | 23/30 / 9.721 / 11.81% |
| gemma | 32 | 29/30 / 4.832 / 15.34% | 28/30 / 6.745 / 20.20% | 27/30 / 6.872 / 20.06% | - | - | - |
| gemma | 64 | 29/30 / 8.030 / 13.62% | 28/30 / 10.918 / 15.55% | 29/30 / 11.007 / 17.24% | - | - | - |
| gemma | 128 | 30/30 / 12.424 / 11.44% | 25/30 / 16.470 / 14.33% | 25/30 / 16.471 / 14.28% | - | - | - |
| gemma | 256 | 28/30 / 23.259 / 8.06% | 27/30 / 23.045 / 9.74% | 27/30 / 23.045 / 9.74% | 28/30 / 24.284 / 8.87% | 28/30 / 24.284 / 8.87% | - |
| gemma | 512 | 28/30 / 24.923 / 7.36% | - | 27/30 / 23.045 / 9.74% | - | - | 29/30 / 23.893 / 4.33% |

## new60

| Model | Q | Live profile | IBS capped | IBS nominal | Native | Flow W1 | Flow W2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| llada21 | 4 | 47/60 / 1.511 / 37.79% | 46/60 / 1.402 / 36.70% | 41/60 / 1.499 / 36.65% | - | - | - |
| llada21 | 8 | 48/60 / 2.345 / 30.26% | 47/60 / 2.414 / 29.87% | 46/60 / 2.366 / 28.96% | - | - | - |
| llada21 | 16 | 50/60 / 3.443 / 27.21% | 49/60 / 3.468 / 21.17% | 49/60 / 3.470 / 20.83% | - | - | - |
| llada21 | 32 | 49/60 / 5.871 / 18.27% | 46/60 / 4.607 / 16.48% | 46/60 / 4.607 / 16.48% | 49/60 / 5.652 / 15.56% | 49/60 / 7.146 / 17.04% | - |
| llada21 | 64 | 47/60 / 7.873 / 10.89% | - | 48/60 / 4.481 / 15.02% | - | - | 51/60 / 10.149 / 11.51% |
| gemma | 32 | 56/60 / 4.902 / 15.77% | 51/60 / 6.888 / 19.02% | 51/60 / 6.678 / 15.46% | - | - | - |
| gemma | 64 | 54/60 / 8.249 / 14.13% | 53/60 / 11.116 / 19.07% | 51/60 / 11.170 / 18.05% | - | - | - |
| gemma | 128 | 58/60 / 13.359 / 12.23% | 54/60 / 16.377 / 14.88% | 54/60 / 16.392 / 14.85% | - | - | - |
| gemma | 256 | 55/60 / 24.749 / 9.01% | 52/60 / 23.961 / 11.85% | 52/60 / 23.961 / 11.85% | 57/60 / 26.123 / 9.52% | 57/60 / 26.123 / 9.52% | - |
| gemma | 512 | 56/60 / 25.932 / 7.44% | - | 52/60 / 23.961 / 11.85% | - | - | 55/60 / 26.139 / 4.60% |

[Same-offer deltas vs each baseline, including original30 and new60](tradeoffs.md). [Budget curves with numeric readouts](frozen90.png). [Numeric data and source paths](metrics.csv). [Lower-offer receipt manifest](llada-lower.json). Regenerate this table from the repository root with `python -m experiments.t2t.serving_slots.fixed_request_comparison`. Refresh only complete LLaDA lower-offer cohorts on the receipt host with `llada_fixed_lower_cohorts.py --q4 Q4_ROOT --cross CROSS_ROOT --manifest MANIFEST`.

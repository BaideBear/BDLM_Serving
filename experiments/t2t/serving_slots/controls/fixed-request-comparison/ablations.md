# LLaDA fixed-offer ablations

The earlier `Request` is quarter_edit_warm. `Active window` changes selected head rows; `Paid MASK` samples projected waiting MASKs; `Paid repeat` accepts a repeated measured proposal; `Paid frontier` accepts a high-confidence contiguous run. The latter dropped Q64 accuracy by seven against paid repeat. The paired H800 Q8 control is 71/90, not the A800 window row (72/90). Compare subgroup and same-device receipts before attributing differences. [Ablation figure](ablations.png). [Main comparison](README.md).

**Live profile** = `quarter_window_repeat64_paid16_frontier32` for LLaDA, `dense_stop_joint_verified_eos` for Gemma. Values are from the independent fixed90 runs implementing each exact path, not 90 reruns through the switching API. LLaDA uses Active window at Q4/8, Paid MASK at Q16, Paid frontier at Q32, and Paid repeat at Q64. [Switching qualification](../strict-handle/README.md).

Each cell: **correct / macro TPF / token efficiency (%)**. Missed or unavailable metrics are `n/a`. The Q in the first column is a per-request decoder offer, not mean Q/F. Native and Flow are shown only at their actual block-size endpoints.

TPF averages per-request final nonspecial N / all post-prefill F (including Gemma encoder). TE pools N / selected decoder rows reading history KV, not N / decoder-plus-encoder rows. Flow TE is omitted where this selection is not separately verified; it is not zero. For LLaDA W1, the [90-case decoder-call ledger](llada-flow-w1-calls90.json) supplies the measured decoder rows with prefill excluded; the fixed budget is not used as a multiplier. Nominal IBS is a fixed window, not necessarily a per-forward hard cap. At 2B only the native fixed-window IBS run is shown; it saturates at a single physical block. Contrasts with native/IBS/Flow are descriptive on the same frozen cases, sometimes on different GPUs. Differences are not claims of statistical significance.

## frozen90

| Model | Q | Live profile | Request | Active window | Paid MASK | Paid repeat | Paid frontier |
|---|---:|---:|---:|---:|---:|---:|---:|
| llada21 | 4 | 72/90 / 1.504 / 37.68% | 18/90 / 1.182 / 27.22% | 72/90 / 1.504 / 37.68% | - | - | - |
| llada21 | 8 | 72/90 / 2.300 / 29.94% | 29/90 / 1.677 / 16.22% | 72/90 / 2.300 / 29.94% | 70/90 / 2.424 / 32.08% | - | - |
| llada21 | 16 | 75/90 / 3.356 / 25.60% | 62/90 / 2.887 / 17.66% | 70/90 / 2.933 / 21.05% | 75/90 / 3.356 / 25.60% | - | - |
| llada21 | 32 | 75/90 / 5.793 / 18.06% | 74/90 / 4.937 / 16.25% | - | - | - | 75/90 / 5.793 / 18.06% |
| llada21 | 64 | 75/90 / 7.672 / 10.87% | 72/90 / 6.979 / 9.94% | - | - | 75/90 / 7.672 / 10.87% | 68/90 / 7.966 / 11.94% |

## original30

| Model | Q | Live profile | Request | Active window | Paid MASK | Paid repeat | Paid frontier |
|---|---:|---:|---:|---:|---:|---:|---:|
| llada21 | 4 | 25/30 / 1.489 / 37.45% | 8/30 / 1.196 / 27.19% | 25/30 / 1.489 / 37.45% | - | - | - |
| llada21 | 8 | 24/30 / 2.210 / 29.32% | 9/30 / 1.528 / 15.70% | 24/30 / 2.210 / 29.32% | 24/30 / 2.326 / 30.93% | - | - |
| llada21 | 16 | 25/30 / 3.183 / 23.00% | 20/30 / 2.769 / 17.31% | 21/30 / 2.799 / 20.47% | 25/30 / 3.183 / 23.00% | - | - |
| llada21 | 32 | 26/30 / 5.637 / 17.65% | 26/30 / 4.811 / 15.31% | - | - | - | 26/30 / 5.637 / 17.65% |
| llada21 | 64 | 28/30 / 7.271 / 10.83% | 25/30 / 6.387 / 9.09% | - | - | 28/30 / 7.271 / 10.83% | 24/30 / 7.672 / 11.99% |

## new60

| Model | Q | Live profile | Request | Active window | Paid MASK | Paid repeat | Paid frontier |
|---|---:|---:|---:|---:|---:|---:|---:|
| llada21 | 4 | 47/60 / 1.511 / 37.79% | 10/60 / 1.176 / 27.23% | 47/60 / 1.511 / 37.79% | - | - | - |
| llada21 | 8 | 48/60 / 2.345 / 30.26% | 20/60 / 1.751 / 16.58% | 48/60 / 2.345 / 30.26% | 46/60 / 2.474 / 32.74% | - | - |
| llada21 | 16 | 50/60 / 3.443 / 27.21% | 42/60 / 2.946 / 17.84% | 49/60 / 2.999 / 21.34% | 50/60 / 3.443 / 27.21% | - | - |
| llada21 | 32 | 49/60 / 5.871 / 18.27% | 48/60 / 4.999 / 16.77% | - | - | - | 49/60 / 5.871 / 18.27% |
| llada21 | 64 | 47/60 / 7.873 / 10.89% | 47/60 / 7.275 / 10.40% | - | - | 47/60 / 7.873 / 10.89% | 44/60 / 8.113 / 11.91% |

[Same-offer deltas vs each baseline, including original30 and new60](tradeoffs.md). [Budget curves with numeric readouts](frozen90.png). [Numeric data and source paths](metrics.csv). [Lower-offer receipt manifest](llada-lower.json). Regenerate this table from the repository root with `python -m experiments.t2t.serving_slots.fixed_request_comparison`. Refresh only complete LLaDA lower-offer cohorts on the receipt host with `llada_fixed_lower_cohorts.py --q4 Q4_ROOT --cross CROSS_ROOT --manifest MANIFEST`.

# Shared fixed-budget planner: formal90 anchor

Formal registration `a18f108aef1dbc607f3b21b2a4d169b04f5ae5dab01be400d015d07249d406f3`; execution revision `7a38ffc86a2bec165de1be6448bb5be2901e7c11`.
Engineering reference, not a tuned winner. All rows are independent, fixed-offer requests.
Each metric cell is **correct / macro TPF / pooled token efficiency (%)**.
A dash marks an untested budget, `n/a` an unaudited efficiency denominator.
[Figure](frozen90.png) | [All source values](metrics.csv) | [Deltas, physical Q and calibration split](summary.json).

| Model | Q | Shared planner | Live profile | IBS capped | IBS nominal | Native | Flow W1 | Flow W2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llada21 | 4 | 21/90 / 1.309 / 28.71% | 72/90 / 1.504 / 37.68% | 69/90 / 1.395 / 35.66% | 59/90 / 1.477 / 36.44% | - | - | - |
| llada21 | 8 | 43/90 / 2.033 / 19.30% | 72/90 / 2.300 / 29.94% | 72/90 / 2.364 / 29.36% | 69/90 / 2.341 / 28.22% | - | - | - |
| llada21 | 16 | 42/90 / 2.961 / 11.71% | 75/90 / 3.356 / 25.60% | 72/90 / 3.375 / 20.54% | 73/90 / 3.371 / 20.51% | - | - | - |
| llada21 | 32 | 64/90 / 5.258 / 14.34% | 75/90 / 5.793 / 18.06% | 71/90 / 4.495 / 15.23% | 71/90 / 4.495 / 15.23% | 75/90 / 5.677 / 15.56% | 75/90 / 7.111 / n/a | - |
| llada21 | 64 | 59/90 / 6.206 / 10.21% | 75/90 / 7.672 / 10.87% | - | 72/90 / 4.397 / 14.56% | - | - | 74/90 / 10.006 / n/a |
| gemma | 32 | 75/90 / 4.563 / 14.26% | 85/90 / 4.879 / 15.63% | 79/90 / 6.840 / 19.33% | 78/90 / 6.743 / 16.65% | - | - | - |
| gemma | 64 | 79/90 / 6.937 / 10.70% | 83/90 / 8.176 / 13.96% | 81/90 / 11.050 / 17.65% | 80/90 / 11.116 / 17.78% | - | - | - |
| gemma | 128 | 79/90 / 10.355 / 8.26% | 88/90 / 13.047 / 11.97% | 79/90 / 16.408 / 14.69% | 79/90 / 16.418 / 14.66% | - | - | - |
| gemma | 256 | 83/90 / 13.055 / 6.28% | 83/90 / 24.252 / 8.69% | 79/90 / 23.656 / 11.10% | 79/90 / 23.656 / 11.10% | 85/90 / 25.510 / 9.31% | 85/90 / 25.510 / n/a | - |
| gemma | 512 | 80/90 / 14.082 / 4.66% | 84/90 / 25.596 / 7.42% | - | 79/90 / 23.656 / 11.10% | - | - | 84/90 / 25.390 / n/a |

At 2B, nominal IBS can saturate at one block. Flow is shown at B and 2B only.
TPF counts actual post-prefill forwards; efficiency pools final nonspecial tokens over selected history-reading decoder rows.

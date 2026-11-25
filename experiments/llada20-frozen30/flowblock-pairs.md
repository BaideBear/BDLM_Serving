# FlowBlock: dense vs per-block PrefixCache

Every row uses the same 30 requests. Request Q excludes padding; physical Q includes it. These are costs summed across requests, not wall-clock speed.

## llada20

| W | Method | Correct / 30 | Mean TPF | Pooled TPF | Efficiency | Physical efficiency | Forwards | Request Q | Physical Q |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 2 | flow | 28 | 3.964 | 2.745 | 4.29% | 4.29% | 5,086 | 325,504 | 325,504 |
| 2 | prefix | 25 | 3.865 | 2.280 | 4.48% | 3.71% | 7,337 | 373,278 | 451,349 |
| 3 | flow | 27 | 4.029 | 2.474 | 2.58% | 2.58% | 5,738 | 550,848 | 550,848 |
| 3 | prefix | 25 | 3.982 | 2.195 | 2.64% | 2.34% | 7,870 | 655,576 | 736,761 |
| 4 | flow | 26 | 4.061 | 2.522 | 1.97% | 1.97% | 6,291 | 805,248 | 805,248 |
| 4 | prefix | 24 | 4.044 | 2.376 | 2.07% | 1.90% | 6,799 | 781,041 | 848,386 |

## llada21

| W | Method | Correct / 30 | Mean TPF | Pooled TPF | Efficiency | Physical efficiency | Forwards | Request Q | Physical Q |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 2 | flow | 27 | 9.262 | 6.626 | 10.35% | 5.94% | 1,942 | 124,288 | 216,672 |
| 2 | prefix | 23 | 7.300 | 4.102 | 8.00% | 3.25% | 3,387 | 173,719 | 426,948 |
| 3 | flow | 23 | 10.342 | 6.269 | 6.53% | 5.23% | 2,369 | 227,424 | 284,137 |
| 3 | prefix | 21 | 8.068 | 4.799 | 5.78% | 3.78% | 2,963 | 245,881 | 376,502 |
| 4 | flow | 25 | 9.472 | 6.921 | 5.41% | 5.41% | 1,948 | 249,344 | 249,344 |
| 4 | prefix | 22 | 8.118 | 4.824 | 4.18% | 3.82% | 2,923 | 337,390 | 369,054 |

Mean TPF is the equally weighted mean of the 30 individual N/F ratios; pooled TPF is total N / total F. Quality includes all 30 responses, even length-capped outputs. The 2.1 reference for these six rows is the paired PrefixCache run, including its dense controls.

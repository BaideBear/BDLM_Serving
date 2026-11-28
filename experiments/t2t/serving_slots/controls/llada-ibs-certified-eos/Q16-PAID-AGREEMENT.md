# LLaDA2.1 strict IBS Q16: paid agreement six-case gate

All three arms use the same certified frozen-prefix EOS close and include
any actual debt-only forward in their totals. The single intervention in
`apply` is an extra acceptance after native M2T/T2T, based on measured MASK
rows only: same absolute row and physical block, adjacent physical decoder
ticks, same argmax, FP32 probability strictly above 0.5 twice, no special
IDs, and not already accepted by the native sampler. The tracker has no
cross-request state. It does not add query rows or alter the planner or
T2T rules. An extra write follows ordinary strict IBS writeback.

The six preregistered requests ran all three arms on `n164:H0`,
`GPU-ceadb3cd-e63f-1446-a2d5-a291144f8efb`, on matching prompts and seeds
with rotated arm order. The paid stock arm matches every historical
ordinary prefix and output ID, and tracked-noop matches paid stock
exactly. The first fork on each case preserves the physical input, rows,
native writes, T2T, and incoming debt; only the agreement writes are added.
The audit also replays the subsequent writes and owed input IDs, including
the terminal debt-only payment when present.

| Case | Stock / tracked F | Apply F | Stock / tracked Q | Apply Q | Extra writes | First tick (zero-based) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 73 | 65 | 943 | 856 | 31 | 2 |
| 13 | 77 | 85 | 1027 | 1156 | 30 | 7 |
| 25 | 39 | 34 | 553 | 479 | 6 | 20 |
| 33 | 69 | 66 | 945 | 871 | 40 | 2 |
| 55 | 121 | 102 | 1485 | 1387 | 43 | 2 |
| 71 | 20 | 19 | 276 | 261 | 2 | 7 |
| **Total** | **399** | **371** | **5229** | **5010** | **152** | |

The six-case benefit gate passed: 28 fewer F (7.02%) and 219 fewer Q
(4.19%), with 6/6 correct and zero failures in all arms. Case 13 regressed
by 8 F / 129 Q. Stock and tracked paid one terminal debt-only call of six
rows each (case 7); apply paid one of one row (case 33). Delivered N was
1379 / 1379 / 1384. Apply changes output IDs on five cases (only 71 is
exact); these are measured trajectory changes, not ID-equivalent
acceleration or a latency result. No expansion was run.

Results and all raw pairs:
`n164:/data1/project_data/gx-erqw-20261123/results/llada-ibs-q16-paid-agreement-h0-v1/`.
The same basename under `runtime/` identifies the frozen source overlay
and supervisor log. The six raw hashes, fork receipts and bill are in
`receipts.ndjson` and `complete-six.json`. `raw-audit.json` passed without
loading the model or running a scorer. Six no-model unit tests and the
remote source-and-registration preflight passed before launch.

| Receipt | SHA-256 |
| --- | --- |
| `q16-source.sha256` | `d0f79d66c03c0930f83de0b49a603549e65212177a5d6dd81bebdc8dbdce62a7` |
| `complete-six.json` | `07f0505a314e65dd108d30577117c33595537e177332a6c72b8d62b9a45f42db` |
| `receipts.ndjson` | `2cdf8e9b347e924b9d2cdee8ef10f08575a685f0dfeb8711bbffd726c70a2a2e` |
| `raw-audit.json` | `1d1e61371e7931242eccc5ef14475967f5598e53e07be05f280ef538fcc60da6` |
| `returned.identity` | `38ab8975396afc6bf3feef14e66efd7269eb80937ccbee1c2282c6053957f97d` |

At `2026-11-28T19:32:52+08:00`, worker `2125784` was gone, the original
physical H0 lock (`66305:200303112`) was acquirable, and holder `2145802`
(UID `1018`, start ticks `2041177602`) was the sole H0 compute process.

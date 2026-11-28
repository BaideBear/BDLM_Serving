# LLaDA Q16 paid-agreement allowance: H0 six-case screen

**Result: passed paired and offline audits on all six cases.** Capping optional
v1 admissions at 0.5 uncertainty per block decision reduced generation forwards
by 25 (6.7%) and physical query rows by 395 (7.9%) versus full v1. All four arms
scored 6/6, with no generation failures. This is a fixed-panel result, not a
frozen90 or latency claim; no expansion was run.

All arms used the same strict IBS Q16 and 2048-token ceiling, with actual
certificate debt paid before frozen-prefix EOS delivery. The capped arm
replaces only the request-local ledger factory. It consumes the same current
physically projected MASK observations as v1, never borrows logits, and leaves
native M2T/T2T, special-ID exclusions, row selection and debt accounting intact.
At the first fork in each exposed case, the physical forward (including input
IDs), incoming debt, native decisions and native T2T matched full v1; the capped
arm removed exactly the rejected extras from same-tick writes. Stock, observer,
and full v1 reproduced their complete archived model and decision records.

| Arm | Correct | N | F | Q | N/F | N/Q | Extra writes | Paid D/Q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Paid stock | 6/6 | 1379 | 399 | 5229 | 3.456 | 0.264 | 0 | 1/6 |
| Observer, no-op | 6/6 | 1379 | 399 | 5229 | 3.456 | 0.264 | 0 | 1/6 |
| Full v1 | 6/6 | 1384 | 371 | 5010 | 3.730 | 0.276 | 152 | 1/1 |
| Capped v1 | 6/6 | 1401 | 346 | 4615 | 4.049 | 0.304 | 84 | 1/2 |

N is reported output tokens, F is actual generation forwards, and Q is
physical generation projection rows verified at all layers (prefill excluded,
certificate debt included). Ratios are pooled. Capped output IDs matched v1
on only cases 25 and 71. Case 71 had no refusal, and its entire run was
identical. Capped F rose on 7, 25, and 55, and Q rose on 7 and 25; the pooled
gain is not uniform.

| Case | Stock N/F/Q | Full v1 N/F/Q | Capped N/F/Q | First fork (block, tick, rejected absolute rows) | Frozen refusals (observer/full v1) |
|---:|---:|---:|---:|---|---:|
| 7 | 211/73/943 | 247/65/856 | 247/67/908 | 2, 5, [91] | 9/11 |
| 13 | 289/77/1027 | 333/85/1156 | 288/66/856 | 2, 7, [82, 84] | 35/14 |
| 25 | 206/39/553 | 206/34/479 | 206/36/499 | 9, 20, [316] | 3/3 |
| 33 | 263/69/945 | 214/66/871 | 227/55/721 | 2, 2, [83, 84, 85, 86, 89] | 20/25 |
| 55 | 313/121/1485 | 287/102/1387 | 336/103/1370 | 1, 2, [54] | 39/24 |
| 71 | 97/20/276 | 97/19/261 | 97/19/261 | None | 0/0 |

Frozen refusals are preflight counterfactuals on archived tapes, not live
refusal counts after divergence. The shared class matched the independent
oracle on those tapes; five no-model tests also passed. The live first-fork
coordinates and exclusions matched the predicted first refusals.

## Receipts

- Source commit `8299c87ca1c0630f503cb78d635dd139f65a9a2f`; shared class supplied in
  `84d6388ce9d3a5ed858967c854897309eb585225`.
- Runtime: `n164:/data1/project_data/gx-erqw-20261123/runtime/llada-ibs-q16-risk-cap-h0-six-v1/`.
  Manifest SHA-256 `78140ae108cd1f30c210deefb0d5b34a1efad0c3e51a5245d62b29768ab73eb4`;
  `exposure.json` SHA-256 `4a178824411370007a7f3f6213fe345566ea42c87e278b54a36781ca3876d849`.
- Results: `n164:/data1/project_data/gx-erqw-20261123/results/llada-ibs-q16-risk-cap-h0-six-v1/`.
  Full raw pairs are in `pairs/`; `offline-audit.json` independently reconciles
  them against the archive, per-case receipts, paid certificates, and totals.
  Audit SHA-256 `2b2f58d7df93fbe406e15b9c20bcaf39237f530829379573bb75d59f75a8eee6`.
- H0 `GPU-ceadb3cd-e63f-1446-a2d5-a291144f8efb`, lock inode `66305:200303112`.
  Worker `2848090`, start ticks `2042090577`, UID `1018`. Returned holder
  `2869852`, start ticks `2042115815`, UID `1018`. Post-run verification found the
  worker absent, the lock acquirable, and the new holder the sole GPU process.
  `returned.identity` SHA-256
  `f00d9e022f2d13626814864f4ea099d740ee6ab22c52038fe4f82365a850aa3d`.

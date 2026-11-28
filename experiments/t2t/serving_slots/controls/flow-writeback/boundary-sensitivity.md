# What could one earlier successor call buy?

This is arithmetic on the original measured strict-IBS trajectories, **not an inference result**. On every eligible clean promotion, optimistically delete one whole successor call without changing the output, paying additional Q, or changing any other forward. For request `i` this changes `N_i/F_i` to `N_i/(F_i-c_i)`, where `c_i` counts the first mask-free undersized head call per block followed by a promotion without a head edit. Average over all 90, including requests with no opportunities. We do **not** delete the head's payment. The preceding [address audit](README.md) found no case where the spare seats actually cover the whole first call. Gemma would also have to rebuild across its encoder boundary. A policy that changes future tokens or saves multiple subsequent calls is outside this sensitivity calculation.

| Model | Fixed Q | Stock TPF | TPF if one whole call were free at each clean boundary | Observed unresolved row revisits | Admitted on those visits |
|---|---:|---:|---:|---:|---:|
| LLaDA | 4 | 1.395 | 1.441 | 15,309 | 9,273 |
| LLaDA | 8 | 2.364 | 2.500 | 38,674 | 14,740 |
| LLaDA | 16 | 3.375 | 3.696 | 84,273 | 20,367 |
| LLaDA | 32 | 4.495 | 5.190 | 133,386 | 22,908 |
| Gemma | 32 | 6.840 | 6.918 | 58,615 | 21,026 |
| Gemma | 64 | 11.050 | 11.276 | 60,895 | 23,640 |
| Gemma | 128 | 16.408 | 16.874 | 67,084 | 26,745 |
| Gemma | 256 | 23.656 | 24.716 | 112,753 | 30,757 |

These are the historical device/stop arms; the columns are not a comparison to the newer paid-EOS control on another board. “Revisit” means an unresolved row that was physically projected before in that block. It does not establish two successive agreeing logits or a safe admission, and re-noised Gemma inputs can change between visits. The [aggregate receipt](one-call-sensitivity.json) partitions decoder Q into previous writes requiring payment, first measured unresolved rows, revisited unresolved rows, and all other resolved rows. Each row and each decoder forward is counted exactly once. Prompt prefill and generation encoder are not decoder work; the latter **is** included in the measured macro TPF. Reproduce on the archive host with `python3 tools/ibs_boundary_headroom.py PREFIX90`.

## Next registered screen: two paid observations on an unresolved row

Test the same [admission rule](../../../../../bdlm_serving/methods/t2t/paid_agreement.py) first at LLaDA Q16 and Gemma Q128 (both half a block). Control is strict capped IBS with the **same** certified paid-EOS stop in all arms. Compare stock, observation only, and observation plus acceptance on the fixed six indices `7,13,25,33,55,71`; these are development screens, not new held-out tasks. Prior LLaDA agreement tests used a different executor and should not be pooled.

Only rows already selected by the original policy are eligible. Apply the original sampler first. If it rejects a row, allow an additional acceptance only if the same absolute unresolved position in the same block was measured and rejected on the immediately preceding decoder call, the argmax ID agrees, each current-measurement FP32 probability exceeded 0.5, and the ID is not special. Maintain one previous vote per measured row. A skipped call, block change, differing ID, or weak measurement breaks agreement. No extra query rows, no read of unmeasured logits, no change to the control's RNG draw count or native T2T decisions at the first differing call. Actual added writes must be reprojected before retirement; capture proposals, acceptance, next projection and the physical bill.

**Gate:** validate stock vs observer equality on all six including IDs and F/Q; reconcile the first new admission and its actual next payment with the unchanged prefix, rows, current proposals, original acceptance and physical cap. If no extra admissions occur, report no exposure and do not escalate. If the extra admissions change later quality, fail or increase work, record the full pairs; do not tune thresholds against those errors. A six-case improvement is not a ten-tier result. No claim about native Gemma T2T follows from an IBS screen.

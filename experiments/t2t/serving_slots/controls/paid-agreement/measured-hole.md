# Do not advance optional writes past a measured unresolved hole

Registered after the four-slot checkpoint; it does not modify that release.
On the same-A2 Q64 panel the first extra admission followed a still-rejected
measured position on five of seven losses, but also on many unchanged cases.
That observation selected this hypothesis; the 90 questions are not a fresh
holdout. The proposed mechanism is to delay early IDs whose left context is
still changing. Neither correctness nor stale KV is observable from that
witness alone.

**One change from uncapped repeat.** Let `M` be the ordinary sampler's
physically measured unresolved rows in the current block and step, `A` its
native accepts and `E` the original repeat ledger's eligible extras.
Only keep extras strictly before `min(M minus (A union E))`; keep all `E`
if this difference is empty. Compare absolute positions, return the surviving
IDs in their original sampler order. Eligibility is determined *before*
filtering, so an earlier eligible extra does not itself block a later one.
Real measurements remain valid votes for the next tick, even when the extra
was not accepted. A gap in selected positions alone is not a hole. This
does not certify an unmeasured prefix, change any native M2T/T2T, add
attention masking, choose new Q rows, or spend/accumulate an allowance.
[Shared implementation](../../../../../bdlm_serving/methods/t2t/frontier_agreement.py).

**Screen.** First split extra row-events on archived GPU observer trajectories
into retained and blocked, without consulting labels or model logits at other
positions. Preserve all six indices `7,13,25,33,55,71`, including cases with
no eligible extras. These counts have no counterfactual score, F or Q.
Prioritize Gemma paid IBS Q64 as the measured failure point, comparing
same-device stock, uncapped v1 and measured-hole; no second intervention.
At the first *actual* fork require identical preceding state, rows, proposals
and native accepts; audit real payments and T2T support of the executor.
If the filter preserves every v1 decision or suppresses all optional writes
on the unchanged stock path, report that degeneracy instead of presenting it
as a new sampler. No 90-case promotion unless on six it matches or exceeds
paid IBS on correct count and macro TPF (one strictly better), does not
increase both total F and Q, and is not strictly dominated by v1 on the two
primary metrics. Continue to compare independently to the quality-oriented
Live control. Test feasibility on its T2T-capable sampler separately; an IBS
screen does not answer that question.

## Replay on unchanged stock trajectories

| Fixed offer (paid IBS) | Requests | v1 optional row-events | Retained | Blocked | Steps with >=1 blocked / v1-positive steps |
|---|---:|---:|---:|---:|---:|
| LLaDA 2.1 Q16 | 6 | 221 | 131 | 90 | 45/109 |
| DiffusionGemma Q128 | 6 | 81 | 23 | 58 | 15/24 |

Five of six requests in each model had a blocked proposal, though not the
same set of accepted rows or starting baselines. Gemma #71 had no v1 event;
it remains in the six. LLaDA stores all ticks, so the shared ledger could be
replayed directly; Gemma stores v1-positive observer events and their
measured rows, so this is a filter of the *recorded* extras, not an
independent reconstruction of their earlier vote. These are row-events,
not unique positions, changes to final output, forwards saved, or T2T counts.
[Per-request counts and source hashes](measured-hole-exposure.json).
From a checkout on n164 regenerate with `python -m tools.audit_measured_hole
/data1/project_data/gx-erqw-20261123/results`. No model is loaded.

## A2: actual fixed Q64 requests (paid IBS)

Using the same six indices and seeds, not just replaying votes:

| Arm | Correct | Macro TPF | Pooled TE | Delivered N | Total F | Actual decoder Q |
|---|---:|---:|---:|---:|---:|---:|
| Paid IBS | 5/6 | 14.4550 | 24.18% | 1,256 | 88 | 5,194 |
| Uncapped repeat | 5/6 | 14.1731 | 24.21% | 1,337 | 97 | 5,522 |
| Repeat before measured hole | 5/6 | 13.6569 | 23.24% | 1,350 | 102 | 5,809 |

**Do not expand this arm to 90.** Same-board stock and uncapped repeat
reproduced their sealed receipts. Three treatment trajectories diverged
from uncapped repeat (#7, #13, #25), first at ticks 5, 18 and 6 respectively.
At each fork the paid inputs and native decision were unchanged; the new
set of extra admissions was strictly smaller. All three affected requests
used more F and Q than uncapped repeat; the other three were unchanged.
The common wrong answer (#71) remains in the table. This does not establish
what the rule would do to the seven losses in the earlier 90, or to a
T2T-capable Live sampler. [Per-request results and first differences](gemma-q64-measured-hole-six.json).
Reproduction source and receipts on n164:
`results/gemma-q64-frontier-run-20261128T222416/gemma_q64_frontier`.

## A3: observe the native Live sampler, Q128

On the same six and the verified-EOS Live parent, the actual multinomial
draws showed 138 qualifying repeats in 28 calls across five requests. Only
28 row-events in 18 calls (five requests) lie before the first measured
hole. All 138 rows were already dirty on the stock trajectory and projected
again on the next step. None of the repeated IDs equalled the input ID
after stock re-noising. An admission therefore cannot be treated as an
unchanged cached ID or a proven saving in Q. The observer reproduced the
six stock receipts, including output IDs and measured work; it produced
no treatment scores. Full reference, raw log and audited `summary-verified.json`
on n164: `results/gemma-q128-live-agreement-a3-QM86IUzQ`.

## Same six questions: end-to-end results across executors

Each arrow is a same-device contrast. For paid IBS the comparator here is
the **uncapped** repeat arm; for T2T-capable Live it is the unchanged native
Live sampler. The A3 comparison therefore tests adding **filtered repeat**
to Live, not the difference between filtered and unfiltered repeat on Live.
None of these results is a 90-question confirmation.

| Model, fixed Q, comparator | Correct / 6 | Macro TPF | Pooled TE | Total F | Actual decoder Q |
|---|---:|---:|---:|---:|---:|
| [LLaDA Q8, paid IBS repeat](llada-q8-measured-hole-six.json) | 6 → 5 | 2.782 → 2.610 | 37.32% → 31.43% | 535 → 643 | 3,885 → 4,756 |
| [LLaDA Q16, paid IBS repeat](llada-q16-measured-hole-six.json) | 6 → 6 | 4.156 → 4.327 | 27.62% → 27.89% | 371 → 388 | 5,010 → 5,227 |
| [Gemma Q64, paid IBS repeat](gemma-q64-measured-hole-six.json) | 5 → 5 | 14.173 → 13.657 | 24.21% → 23.24% | 97 → 102 | 5,522 → 5,809 |
| [Gemma Q128, native Live](gemma-live-q128-measured-hole-six.json) | 6 → 6 | 12.160 → 11.982 | 11.40% → 11.24% | 120 → 121 | 12,451 → 12,544 |

LLaDA Q16 spent more work than repeat and delivered 74 more tokens; the
gain in macro TPF is not a reduction in F. Native Live Q128 kept the same
RNG procedure, T2T and encoder bookkeeping; both arms paid three generation
encoder calls and peaked at 128 decoder Q. The earlier zero same-ID result
does not mean the filtered admissions can be applied for free. **Do not
adopt the measured-hole rule across tiers.**

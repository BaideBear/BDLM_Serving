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

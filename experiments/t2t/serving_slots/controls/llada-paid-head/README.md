# LLaDA Q64: projected but unsampled unresolved head positions

Same 17 requests common to the four gate arms at the snapshot: indices 0–15
and 20. This partial cohort is **not** a quality or TPF comparison. All arms
project every valid head row at Q64. When only successor rows are cold, the
reference sampler can select those cold rows to the exclusion of already
projected, warm, non-dirty MASKs in the head.

| Successor gate | Affected requests | Affected forwards | Paid, unsampled warm head MASK rows |
|---|---:|---:|---:|
| Off | 0/17 | 0 | 0 |
| Immediate | 17/17 | 171 | 2,577 |
| Half | 17/17 | 204 | 2,519 |
| Quarter | 17/17 | 143 | 741 |

Counts are opportunities, not tokens saved or fixes to wrong answers. The
single-factor follow-up holds the row plan fixed at the first changed call
and tests whether also sampling exactly those head MASKs improves the final
trajectory. Aggregate CSV/JSON contain counts only.

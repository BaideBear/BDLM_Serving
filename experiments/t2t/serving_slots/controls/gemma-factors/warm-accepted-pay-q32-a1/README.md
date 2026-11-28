# Fixed Gemma Q32: keep warm accepted-debt rows eligible for sampling

**Decision: do not promote payment-only.** All 90 matched Q32 pairs completed
on physical A800 GPU1. Every off receipt exactly replays historical
verified-EOS Q32, including the per-tick tape, outputs, scorer and bill.
No missing, failed, repeated, or over-budget pairs. The new rule really
changed the sampler on 88/90 cases but did not establish a TPF improvement
at comparable quality.

| Cohort | Correct off/on | Macro TPF off/on | Post-prefill F off/on | Selected decoder Q off/on | Canonical TE off/on |
|---|---:|---:|---:|---:|---:|
| Original 30 | 29 / 27 | 4.832 / 4.884 | 2,247 / 2,136 | 68,412 / 65,131 | 15.34% / 15.68% |
| New 60 | 56 / 56 | 4.902 / 4.936 | 4,627 / 4,692 | 141,277 / 143,745 | 15.77% / 15.88% |
| Frozen 90 | 85 / 83 | 4.879 / 4.919 | 6,874 / 6,828 | 209,689 / 208,876 | 15.63% / 15.82% |

Delta macro TPF +0.040, paired within-dataset descriptive bootstrap 95%
interval [-0.044, +0.118]. Correct delta -2.22 percentage points
[-7.78, +3.33]; this is not a noninferiority qualification. Two rescues
(#41, #50), four losses (#10, #16, #51, #68). There were 44 TPF wins,
14 ties, 32 losses; only 17/90 delivered ID sequences were identical.
The six prespecified anchors (0/10/20, 30/50/70) all passed the *screen*
(5 correct per arm, mean TPF 5.611/5.638, F 381/380, decoder Q 11,644/11,658).
The 84 other cases alone scored 80/78 with TPF 4.826/4.867.

**What changed.** In a sparse call after all 256 head rows have KV and SC,
with some unresolved rows being sampled, the candidate projects the same
plan but does not feed already-accepted dirty rows to the sampler on that
payment. Dirty unresolved rows and accepted clean-prefix rows keep their
eligibility. The first actual divergence on all 88 exposed requests was
at zero-based decoder tick 17, head 0, sampler age 9. Positions, all
fresh logits at those positions, and the previous trajectory matched.
Both sides paid 32 physical rows there; on consumed 2 to 28 fewer rows.
No divergence and identical complete receipts on #75 and #76. The six
anchor first drops were 23, 26, 28, 28, 14, 28 rows, respectively.

**Why fewer warm forwards did not justify replacement.**

| Full90 measured work | Off | On |
|---|---:|---:|
| Non-special tokens N | 32,774 | 33,041 |
| Decoder F | 6,794 | 6,746 |
| Cold / warm / closing drain / ordered verify F | 2,550 / 3,001 / 165 / 1,078 | 2,580 / 2,913 / 169 / 1,084 |
| Warm sampled / warm pure drain F | 2,780 / 221 | 2,731 / 182 |
| Warm samples with no admissions / nonpositive net admissions | 40 / 102 | 7 / 11 |
| First-head warm F only | 1,783 | 1,695 |
| Heads; required post-prefill encoder F / Q | 170; 80 / 20,480 | 172; 82 / 20,992 |
| Mixed warm calls with eligible rows; eligible row-visits | 2,530; 28,468 | 2,546; 28,310 |
| Eligible visits actually dropped from sampling | 0 | 28,310 |
| Edits / revocations on eligible visits | 143 / 427 | 0 / 0 |
| Actual dirty writes / paid dirty rows | 97,711 / 97,711 | 97,139 / 97,139 |
| Maximum physical decoder width | 32 | 32 |

Suppressing those warm revisits is not free: 570 eligible control visits
really changed a previously accepted token or revoked its acceptance.
Removing rows also changes subsequent selected-set entropy admission,
RNG draws, SC and trajectories. This is not equivalent to IBS or to the
native sampler. The recorded savings of 88 warm F, 46 total F and 813
decoder Q are trajectory results, not contractual reductions per call.
Cold, verify and encoder costs increased on the resulting requests; on
the new 60 alone total F rose by 65. Ordered argmax verification, EOS
guard, required encoder, and every actual dirty write stayed protected.
No full-width saving follows just from not sampling a row already projected.

Macro TPF is mean per-request N/(decoder F + post-prefill encoder F);
TE is pooled N/selected history-reading decoder Q, excluding encoder Q.
Neither is a wall-clock speed measurement. Capped IBS's 6.840 TPF at 79/90
is a diagnostic on different trajectories, not a third arm; see [IBS stage
accounting and prior negatives](diagnosis.md).

**Evidence and reproduction.** The local checkout has only
[paired scalars and fingerprints](pairs.csv), [cohorts](cohorts.csv),
[provenance](provenance.json), [preregistration](registration.md),
[launcher](launch.sh), and the independent source/test files.
The three compact data files matched their A800 SHA-256. Six no-model tests
passed both locally and on the server. The model worker PID 382341 exited
normally; `complete.json` was sealed 2026-11-28 14:02:20 +08. Physical A1
had no compute PID and 0 MiB at 14:02:36 +08. No successor or holder.

Raw results: `/gs/results/warm-accepted-pay-q32-a1-1128`. Regenerate the
compact on A800 without any inference or allocation:

```sh
bash /gs/runtime/warm-accepted-pay-q32-a1-1128/report.sh
```

The independent checkout files are `experiments/t2t/serving_slots/`
`gemma_warm_accepted_pay{,_eval,_report}.py`, `tests/test_gemma_warm_accepted_pay.py`,
and this control directory. No public profile, central table or old launcher edit.

# LLaDA 2.1: more accepted MASKs do not imply a better request

Same-device paired Q64 run, full current head, successor admission at ≤1/4
remaining head MASK, one fixed offer for the entire request. The M2T cutoff
is the only intervention. Both arms use the same T2T cutoff (0.5), per-block
fallback and planner. At the first divergence each request sees the same
selected Q rows and measured proposals; extra admissions must be explained by
the changed cutoff. Subsequent writes, edits, output lengths and forwards are
allowed to differ and are billed in full.

| Fixed configuration, original 30 | Correct | Macro TPF | Pooled TE | F | N | History Q |
|---|---:|---:|---:|---:|---:|---:|
| Global 0.9 control | 24 | 5.643 | 8.64% | 3,598 | 14,095 | 163,115 |
| Global 0.5 trial | 19 | 8.510 | 12.00% | 2,439 | 13,684 | 114,022 |
| Historical Optimus IBS64 | 24 | 4.228 | 13.74% | 5,078 | 14,469 | 105,283 |
| Historical Flow W2 | 23 | 9.721 | unavailable | 1,768 | 13,359 | unavailable |
| Historical native B32 (different offer) | 26 | 5.728 | 15.55% | 2,575 | 12,814 | 82,400 |

All 30 pairs reached a measured difference; no failures. Lowering the cutoff
rescued three answers and lost eight. T2T events increased 633→1,001,
while per-block fallback admissions fell 2,544→1,181. This is a quality/TPF
tradeoff, not an equivalent-work speedup. The three qualification tasks
alone misleadingly favored the lower cutoff on quality (1→3).
See the [registered receipt](c64.json) and the
[same-cohort baselines](../block-history90-matrix.csv). Only complete
cohorts have metrics; the qualification tasks also belong to the original 30.

The separate `warm_head` variant permits 0.5 only on MASKs of the current
physical head once its valid generated rows had already been projected
**before** this tick; first visits and successor rows stay at 0.9. Each new
control must replay the prior high-cutoff result on the same device (including
the complete output digest, F and Q). This variant is not the published
0.5 result and is not enabled in the default policy.

## Completed Q64 comparison

The [receipt](c64.json) now includes all 90 pairs. High vs low cutoff on
the new 60: 46 vs 40 correct, 6.596 vs 9.721 macro TPF, 9.42% vs 12.54%
pooled TE. Across all 90: **70 vs 59 correct**, 6.279 vs 9.317 macro TPF,
9.14% vs 12.36% pooled TE. Six rescues, 17 losses and zero failures. Low
cutoff reduces total forwards from 10,020 to 7,182 but changes the decoding
trajectory and is rejected as a quality-preserving optimization.

## Head-only Q16

At fixed Q16 on n164, the [same-device registered comparison](c16.json)
on the original 30 also loses scored answers (three rescues, five losses).
Here even the total physical work rises despite the higher macro TPF:

| Original 30 | Correct | Macro TPF | Pooled TE | F | N | History Q |
|---|---:|---:|---:|---:|---:|---:|
| 0.9 | 18 | 2.494 | 12.78% | 11,175 | 16,978 | 132,839 |
| 0.5 | 16 | 3.220 | 13.20% | 11,883 | 18,693 | 141,602 |
| Historical IBS16 | 24 | 3.173 | 19.90% | 5,549 | 14,548 | 73,098 |

These are unweighted per-request N/F means and pooled N/Q efficiencies.
Do not use the other device's historical head-only row as the paired
control. The same-device qualification replays indices 0, 10 and 20;
it is not a cross-device equivalence test on all 30 outputs.

The [completed Q16 run](c16.json) also rejects global relaxation on frozen90.
At 0.9 vs 0.5, the new60 scored 37 vs 26; all90 **55 vs 42** (8 rescues,
21 losses). Total F 30,317 vs 29,248, N 48,518 vs 51,632, history Q 360,843 vs
346,033, macro TPF 2.723 vs 3.181, pooled TE 13.45% vs 14.92%. Zero failures
in both arms. Unlike the original30 subcohort, the full cohort used fewer total
forwards under 0.5. The loss of quality is not a result of missing samples.

## Full cohort: Q32 and scoped Q64

The Q32 comparison uses the sparse head-only control, not the later editable
full-head close. The Q64 comparison uses full-head quarter gating in both
arms; the warm-only arm lowers M2T only once the current head is fully
initialized. It is not the separate `quarter_edit_warm` public profile.
Each comparison includes the same 90 questions; all four arms had zero failures.

| Fixed offer / M2T rule | Original30 | Added60 | Frozen90 | Macro TPF | Pooled TE | F | N | Decoder Q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [Q32](c32.json), 0.9 | 23 | 49 | 72 | 4.513 | 16.62% | 13,870 | 44,068 | 265,204 |
| Q32, 0.5 globally | 16 | 35 | 51 | 5.968 | 19.15% | 13,081 | 47,923 | 250,312 |
| [Q64](c64-warm.json), 0.9 | 24 | 46 | 70 | 6.279 | 9.14% | 10,020 | 41,634 | 455,344 |
| Q64, 0.5 on the initialized head | 24 | 42 | 66 | 7.505 | 11.43% | 7,446 | 39,992 | 349,850 |

Q32: 2 rescues and 23 losses. Q64 warm-only: 7 rescues and 11 losses.
The latter lost fewer answers than the earlier global Q64 trial (59/90),
but still falls short of its own control; the tie on original30 did not hold
on added60. Do not adopt either threshold relaxation as a quality-preserving rule.

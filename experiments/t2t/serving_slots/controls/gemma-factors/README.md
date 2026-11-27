# Gemma: fixed-budget prefix selection and terminal verification

Two separate single-factor studies. Do not pair receipts across the two
source snapshots. `coverage.csv` reports each registered contrast in original
30, added 60 and all 90, including missing coverage. `paired.csv` publishes
absolute metrics on the common, fully scored cohort only. The reference is
repeated per contrast; never sum across contrasts or overlapping cohorts.

| Directory | Matched arms | Diagnostic other than quality |
|---|---|---|
| `rank/` | Within each Q32/64/128/256, keep prefix dose at 1/8. Reference is right edge. Separately change to oldest, seeded uniform, or reuse currently computed unresolved logits when the whole initialized head's debt is paid. | Actual seats differing from same-state right edge, prefix edits and revocations, paid sampling F, and changes to the complete output ID sequence. Different IDs need not mean worse quality. |
| `close/` | Within each Q32/64/128/256/512, compare unmodified head-only stopping with exact-order drain/first-verify fusion. Both sides use the same head-only route; Q512 is an offer, **not** a dual-canvas test. | Check output ID sequences, encoder work, and whether all F and Q reductions exactly match the fused work. Any parity violations remain visible. |

`token_efficiency` is `sum(N)/sum(history_Q)`, not a percentage column;
`macro_TPF` includes post-prefill encoder forwards. Rank and close use real
decoder projections, so here `history_Q == decoder_Q`. Physical Q includes
encoder Q. Rescue/loss, index lists and confidence intervals belong to the
contrast, not an individual side; deltas are trial minus reference. `[]`
means checked with no mismatches; a blank diagnostic on a no-common row is
untested. Scorer and recorded source digests are checked for every receipt
within its study. These are not a hash of the entire runtime.

From the repository root run `bash experiments/t2t/serving_slots/controls/gather_gemma_factors.sh rank`
and the same command with `close`. Optional `A800_HOST`, `A800_PORT`,
`A800_MUX` and `PYTHON` select the SSH host and local Python 3.10+. Optional
second argument selects a different output directory. Only aggregate reports
leave the GPU host. No worker is restarted and no generation runs locally.
The [90-case external controls](../cross-race-gemma/README.md) are not online
interventions on the same sampler.

## First complete slice: original 30, terminal fusion

On the original 30 tasks, all five paid-close comparisons preserved every
final output ID and encoder workload. Observed differences in F and decoder Q
match the fusion counters for each request. Negative and positive opportunities
are both included:

| Fixed decoder offer | Correct, off = on | Macro TPF, off → on | Saved F, total | Saved decoder Q, total |
|---:|---:|---:|---:|---:|
| 32 | 29/30 | 4.607 → 4.607 | 0 | 0 |
| 64 | 29/30 | 7.713 → 7.742 | 6 | 51 |
| 128 | 30/30 | 12.024 → 12.209 | 14 | 96 |
| 256 | 29/30 | 17.177 → 19.072 | 54 | 884 |
| 512 (head only) | 28/30 | 19.657 → 22.184 | 54 | 740 |

The intervention removes a duplicate projection, not final verification or
the encoder. It is easier for the original first verification slice to contain
all remaining dirty rows at a large offer. The table is a statement about
these 30 inputs only; a real two-canvas C512 finish has not been tested here.

## Complete 90: keep verification, avoid projecting the same rows twice

Each row has 90 scored pairs. Within each pair the output sequence and encoder
work matched exactly; the observed F and Q reductions equal the recorded fusions
and dirty rows on every request. TE below uses selected decoder Q; the reference
and trial use the same eligible head-only configuration and prefix share 1/8.

| Offer | Correct, both | Macro TPF, off → on | TE, off → on | F saved | Decoder Q saved |
|---:|---:|---:|---:|---:|---:|
| 32 | 85/90 | 4.666 → 4.669 | 14.98% → 14.98% | 5 | 32 |
| 64 | 83/90 | 7.842 → 7.881 | 13.42% → 13.43% | 23 | 264 |
| 128 | 85/90 | 12.209 → 12.398 | 11.69% → 11.70% | 42 | 361 |
| 256 | 83/90 | 17.742 → 19.799 | 10.17% → 10.28% | 166 | 3,310 |
| 512 (head only) | 83/90 | 20.178 → 22.940 | 8.13% → 8.21% | 171 | 3,630 |

At Q256 the improvement still falls short of native 85/90 and macro TPF 25.510.
`profile-close/` is a separately registered transfer test: Q128 uses the
candidate 1/4 share and Q512 uses the half gate, comparing fusion off/on on
the same physical device per pair. No change in tail acceptance, KV epoch,
termination or order of the verification slices. Even in the Q512 half-gate
profile fusion is forbidden while the successor is resident. Use the same
gather command with `profile-close`; qualify transfer from its own evidence,
never by adding the observed gains to a different fixed profile.

On the original 30, Q128 with the 1/4 share was 30/30 on both sides, macro
TPF 12.034 to 12.212 (12 F, 141 decoder Q saved). Q512 with the half gate
was 28/30 on both sides, 20.749 to 22.718 (29 F, 260 decoder Q saved).
All final IDs and encoder work matched on both pairs. These are not 90-case
estimates and do not establish real-time speed.

All 90 on the two live profiles are now sealed. Q128: **88/90** unchanged,
TPF **12.588 to 12.794**, TE **11.67% to 11.69%**, saving 43 F and 487
decoder Q. Q512 half gate: **84/90** unchanged, TPF **21.344 to 23.035**,
TE **7.02% to 7.04%**, saving 70 F and 931 decoder Q. All 180 pairs had
identical final IDs and encoder costs and exact per-request conservation
of the fused work. The gated Q512 endpoint still trails W2 (84/90,
25.390 TPF). These are paired algorithmic results, not a latency forecast.

## Spending a full block at the actual one-block offer

`capacity/` compares two runs that **both receive Q256**, with the same
1/8 law, no successor, first-verify fusion, and frozen inputs. In the trial,
while denoising, cover and sample the entire head when the head plus all
mandatory writes fit. This is an actual projection planner change. In a
live request with parked tail debt, Q256 is insufficient for full coverage;
the change does not displace that debt. The gate at Q512 is unaffected.
The control retains sparse selection at Q256. Both use the same strict
local sampler, not a call to native generation.

Additionally replay the historical Q256 sparse and historical nominal Q512
head-only receipts, requiring each reference's **maximum decoder width <=
256**. Compare actual output IDs, F, decoder and encoder Q, routing and
scorer; report parity mismatches, never relabel old runs as fresh Q256. Only
the same-budget arms estimate the effect. Recompute with `gather_gemma_factors.sh
capacity`.

All 90 pairs now pass the physical Q256 cap and historical endpoint checks.
Sparse vs full head was **83 vs 83 correct** (one rescue, one loss), macro TPF
**19.799 vs 22.940** (paired delta +3.141, 95% bootstrap interval +2.171 to
+4.125), selected-Q efficiency **10.28% vs 8.21%**. The extra **88,330 decoder
Q** bought **114 fewer post-prefill forwards**, not identical outputs or
identical accuracy per task. Both still trail native in TPF and correct count.
The original30 changed 29 to 28; the added60 changed 54 to 55.

[Five-budget chart, frozen90](fixed-profile/frozen90.png) ·
[original30](fixed-profile/original30.png) · [added60](fixed-profile/new60.png) ·
[numeric data and source per point](fixed-profile/metrics.csv). The line is the
strict fused profile, not best-of; the diamond shows the full-head alternative
at the same cap. Dashed lines are external controls rather than matched
interventions. Rebuild with `python -m experiments.t2t.serving_slots.plot_fixed_gemma_profile`.

Each strict profile row in `metrics.csv` has six mutually exclusive decoder-Q
counts. For the actual fused Q128 1/4 candidate, the shares of **all selected
decoder Q** are 50.08% ordinary dirty refresh, 5.04% discretionary clean
prefix, 28.70% unresolved and 16.18% verification. A row paid during fused
verification is counted under verification, not twice. For the dense Q256
alternative 30.73% are other clean rows and there is no *explicit* ranked
prefix selection; dense projections still cover clean prefixes. The six-way
breakdown for external baselines is not known and remains empty.

`density/` isolates whole-head selection **within** a fixed Q256 offer.
Compare never, only when at least half of the head remains unresolved, only
when fewer than half remain, and always. Each head uses its *pre-forward* state
without hindsight, and admissions/revocations can move the threshold either
way. All four arms retain the same sparse law, verify fusion, no successor and
no early-stop experiment. Endpoints must reproduce the previous local controls
(IDs and F/Q) per task. `gather_gemma_factors.sh density` exports totals only on
complete frozen cohorts; high/low dense F counts are physical route counts.

| Q256 fixed, original 30 only | Correct | Macro TPF | Selected-Q efficiency | Decoder Q | Total F |
|---|---:|---:|---:|---:|---:|
| Sparse throughout | 29/30 | 19.072 | 9.90% | 102,864 | 606 |
| Dense only with at least 128 unresolved | 29/30 | 21.039 | 10.69% | 93,587 | 527 |
| Dense only with fewer than 128 unresolved | 29/30 | 20.155 | 7.36% | 137,005 | 577 |
| Dense throughout | 28/30 | 22.184 | 7.69% | 132,096 | 540 |

The dense-early rule saved aggregate rows as well as forwards on these tasks:
doing more work on one tick does not imply spending more over the request.
Do not confuse *head unresolved density* with the remaining request length,
or infer equal per-task outputs from the equal score totals. Check the added
60 separately before changing the recommended profile.

**Completed fixed Q256 phase study, 360/360 scored receipts.** Counts are
correct responses, not matching token streams. The sparse and fully dense
endpoints replayed their full-output and workload controls. Subsets cover
the same frozen 30 and additional 60 in each arm.

| Q256 denoising policy | Original30 | Added60 | All90 | Macro TPF | TE | N | Total F | Decoder Q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sparse throughout | 29 | 54 | 83 | 19.799 | 10.28% | 30,992 | 1,718 | 301,558 |
| Dense when unresolved ≥128 | 29 | 54 | 83 | 21.601 | 10.90% | 32,759 | 1,720 | 300,575 |
| Dense when unresolved <128 | 29 | 55 | 84 | 20.862 | 7.89% | 31,426 | 1,685 | 398,238 |
| Dense throughout | 28 | 55 | 83 | 22.940 | 8.21% | 32,006 | 1,604 | 389,888 |

High-density only raises macro TPF on both original30 and added60; pooled
over all90 it uses 983 fewer decoder rows but **two more forwards** than
sparse. Outputs get longer; the macro improvement (+1.802, paired 95%
interval +0.765 to +2.839) is not proof of lower latency or fewer total
rounds. Two rescues and two losses leave the total score unchanged. Low-only
rescues one and loses none relative to sparse, at 96,680 extra decoder Q.
Full throughout is the fastest by TPF of these four, not the most efficient
by rows. Native (85, 25.510, 9.31%) and historical IBS256 (79, 23.656,
11.10%) prevent a claim of joint dominance. The high-density option is a
candidate for row-efficient execution at fixed B, not evidence for switching
the external budget at the tail of a request.

## Prefix location and sampling on refresh: original 30

Each entry is correct out of 30, then macro TPF. All use 1/8 and the same
head-only executor. The last column changes sampling eligibility, **not**
the rank: it uses current (not stale) logits from fully paid unresolved rows
on certain otherwise refresh-only ticks.

| Offer | Right edge | Oldest | Seeded uniform | Right edge + sampling on refresh |
|---:|---:|---:|---:|---:|
| 32 | 29, 4.607 | 28, 4.553 | 28, 4.534 | 22, 4.049 |
| 64 | 29, 7.713 | 29, 7.453 | 27, 7.402 | 20, 6.025 |
| 128 | 30, 12.024 | 28, 12.198 | 29, 12.002 | 25, 9.666 |
| 256 | 29, 17.177 | 28, 17.615 | 28, 17.417 | 28, 17.831 |

Keep the right edge for now. At Q64 the alternative sampling rule lost nine
previously correct answers without a rescue and lowered pooled token efficiency
from 13.07% to 8.14%. This is an end-to-end change to the sampling schedule;
the comparison does not isolate which later change in the trajectory caused a
wrong answer. At Q256 the outcome is a speed/quality tradeoff, not dominance.

## Native stop on fully measured consecutive heads

`fullstop/` contains a separate 366-receipt study: off/on on the same GPU per
pair, all 90 at C256 (head only) and C512 (half gate), plus 3 *qualification*
pairs at C128. No terminal fusion and no logits imputed for unselected rows.
Recompute on the original A800 runtime with `python readout.py --complete`.
`pairs.csv` contains only per-request counts and paired changes, not outputs.
Separate the original 30 and additional 60 via the `index` column.

| Offer | Correct / 90, off → on | Macro TPF | Token efficiency | Applied effect |
|---:|---:|---:|---:|---|
| 256 | 83 → 83 | 17.742 → 17.742 | 10.17% → 10.17% | 189 full head ticks in control, only one consecutive pair; zero hits |
| 512 | 84 → 83 | 21.344 → 21.785 | 7.02% → 7.22% | 42/90 had an actionable hit; total F 1,579 → 1,560 |

C512 changed 14 complete output sequences, with one additional wrong answer
(index 65) and no rescues. The paired TPF difference is +0.441, 95% interval
[0.144, 0.770]. Keep the stop *off* in the public profile. C128 had zero full
head ticks on the three qualification tasks; it is not a 90-task result.

**Fixed dense Q256, stop only.** `cross_race --suite dense-stop` repeats the
native stop vs shadow comparison at a physical 256 rows for each denoising
tick, with no successor and the same guarded terminal writeback fusion on
both arms. In contrast to the previous sparse Q256 experiment, consecutive
complete observed head samples are possible throughout denoising. Control
IDs and F/Q must replay `capacity-head-v1` at actual Q256. A stop may end
sampling but never substitutes the observer's argmax for the final measured
verify. The encoder is charged; no terminal work is hidden. Pairs stay on
A800 GPU4 in alternating order. On the cluster, `python fullstop/readout.py
--dense-stop` from this directory audits the receipts; add `--complete` to
require all 180 scored receipts. A partial readout is not a frozen90 result.

**Result: all 180 scored; same A800 per pair.** Every shadow run reproduced
the archived physical Q256 control (IDs and F/Q). Switching the native stop
on only after two consecutive complete measured head samples gave:

| Fixed dense Q256: shadow → apply | Correct | Macro TPF | Selected-Q TE | Total F | Decoder Q |
|---|---:|---:|---:|---:|---:|
| Original 30 | 28 → 28 | 22.184 → 23.259 | 7.69% → 8.06% | 540 → 516 | 132,096 → 125,952 |
| Added 60 | 55 → 55 | 23.318 → 24.749 | 8.48% → 9.01% | 1,064 → 1,004 | 257,792 → 242,432 |
| Frozen 90 | 83 → 83 | 22.940 → 24.252 | 8.21% → 8.69% | 1,604 → 1,520 | 389,888 → 368,384 |

No question changed correctness; four full ID sequences changed. Final N was
32,006 in both groups. 51 requests used fewer F, 39 tied, none increased;
paired macro TPF delta +1.312 (95% stratified bootstrap interval +0.980 to
+1.692). All 171 terminal verify calls in each arm also wrote pending inputs.
This result is for *fixed* Q256; don't transfer it to a mixed path or to the
two-canvas Q512 stop. Historical Optimus IBS256 is 79/90, 23.656, 11.10%
(nominal budget); native Q256 is 85/90, 25.510, 9.31%. The candidate beats
IBS on the two main metrics but not on selected-row efficiency, and trails
native on all three. See [cohorts](fullstop/dense-summary.jsonl) and
[paired counts](fullstop/dense-pairs.csv).

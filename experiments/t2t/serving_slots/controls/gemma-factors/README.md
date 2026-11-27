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

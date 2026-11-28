# Warm prefix refresh on hard-capped IBS

**Final decision: reject the four-row slack refresh as a default.** The [full Q32 confirmation](#fixed-q32-confirmation) failed its retention gate. The nine-case results below are the original screen, not the final decision.
Q32 rescued one of two control misses (#50, MATH) without losing a control success.
The Q32 forward reduction came mostly from a different case (#80). Q64 gained
no correct answers and consumed more decoder rows. This is one paired rescue,
not evidence that the six-point historical gap on frozen90 has been recovered.

| Fixed budget / anchors | Correct off -> on | Macro TPF off -> on | TE off -> on | Total F off -> on | Selected decoder Q off -> on |
|---|---:|---:|---:|---:|---:|
| Q32, all 9 | 7 -> 8 | 6.770 -> 7.047 | .2344 -> .2391 | 504 -> 490 | 14,636 -> 14,574 |
| Q32, original 3 | 3 -> 3 | 6.526 -> 6.613 | .2225 -> .2207 | 150 -> 148 | 4,414 -> 4,449 |
| Q32, added 6 | 4 -> 5 | 6.892 -> 7.263 | .2396 -> .2471 | 354 -> 342 | 10,222 -> 10,125 |
| Q64, all 9 | 7 -> 7 | 12.531 -> 12.566 | .1979 -> .1958 | 303 -> 310 | 17,193 -> 17,606 |
| Q64, original 3 | 2 -> 2 | 14.461 -> 14.461 | .2224 -> .2220 | 74 -> 74 | 4,352 -> 4,361 |
| Q64, added 6 | 5 -> 5 | 11.566 -> 11.618 | .1895 -> .1872 | 229 -> 236 | 12,841 -> 13,245 |

Both budgets use the same nine frozen indices, not 18 distinct tasks. There
are 18 sealed off/on pairs, 0 missing and 18 independent unmodified A1
replays (54 request executions). Frozen90 is **not** an evaluation denominator
here. Macro TPF is the mean of per-request `N/(decoder_F+postprefill_encoder_F)`;
TE is `sum(N)/sum(selected history-reading decoder Q)`, excluding encoder Q.
`N` excludes special tokens. Q32 N went 3,431 -> 3,484; Q64 3,402 -> 3,447.
Thus Q64 macro TPF slightly increasing does not mean it saved physical work.

## Control and intervention

The control is the pinned Optimus IBS adapter with a hard physical decoder-row
cap, not Live. Both use the frozen prompt and scorer. This adapter uses the
same single-user chat template with generation prompt in all three runs.
When a proposal exceeds the cap it protects admission debt, fits a first MASK,
then clean replay rows and remaining MASKs. It calls its .9 dynamic sparse
sampler on measured unresolved rows; rejected MASKs can be re-noised without
incurring new *admission* debt. It refreshes raw-logit self-conditioning for
projected rows. A fully admitted and paid B256 head retires without Live's
ordered argmax verification; the actual nonterminal B256 output is encoded
before the next head. Terminal delivery truncates at EOS. Do not mix these
results with Live's sampling, write-debt, verification or scoring receipts.
The historical Q32/Q64 per-case archive was not available for an inter-board
check; replay equality here is to the pristine pinned code on the same A1.

The on arm changes only row selection. If a call already revisits an
unresolved row, has spare physical seats and can pay the original debt, add at
most four adjacent clean, resolved, previously measured rows at the right edge
of the cached prefix. Never displace an original selection or change the
rightmost visible key. The added rows refresh KV and raw-logit self-conditioning
by the **existing IBS** path. They are neither sampled nor admitted. This can
change downstream sampling through the measured logits; it guarantees no
trajectory-level reduction in calls or rows.

## Evidence

- Q32 exposed 8/9, spending 128 actual extra prefix rows on 34 calls; Q64
  exposed 8/9, spending 265 on 67. The first exposed unresolved-logit hash
  changed in all 8 at each tier. Sampled-row/admission decisions diverged in
  4/9 and 5/9; final IDs changed in 3/9 and 2/9. Unexposed Q32 #40 and Q64
  #20 matched completely. First-fork ticks in `pairs.csv` are zero-based.
- Q32 #50 first changed the decision at tick 23, rescued the answer, and
  **increased** F from 84 to 86. Q32 #80 remained correct and fell from 72 to
  60 F, accounting for 12 of the 14 net saved F. At Q64 #70 stayed incorrect
  and grew from 39 to 45 F. The score and cost changes need not coincide.
- Peak observed decoder widths were 32 and 64 respectively on both arms.
  All 85 projection-width streams and physical position IDs were checked.
  All admissions were paid on each arm: 4,608/4,608 at Q32, 4,352/4,352 at
  Q64. Non-prefill encoder bills were unchanged: 9 calls / 2,304 rows at Q32,
  8 calls / 2,048 rows at Q64. Actual encoder inputs were checked against the
  respective trajectory, not assumed to have identical contents between arms.
- All four preregistered gates passed: exposure at both budgets, a net Q32
  rescue, Q32 macro TPF at least 98% of control, and no net Q64 score loss.
  Only a separately authorized, paired evaluation on the remaining 81 unique
  tasks could establish a frozen90 result. No pilot-selected tuning.

## Receipts and release

`registration.md` fixes the rule, cases and gate. `pairs.csv` includes input
and receipt hashes, both bills, first activation and decision change;
`cohorts.csv` separates original and added anchors. `provenance.json` pins
environment, code, checkpoint manifest, worker identity and the receipt set.
No raw traces or generations were copied locally.

Original raw directory: `/gs/results/gemma-ibs-prefix-bridge-a1-1128/`.
Original runtime: `/gs/runtime/gemma-ibs-prefix-bridge-a1-1128/`. On an
archive host with the registered directory layout, `bash report.sh` in that
runtime verifies the sealed receipts without loading a model and writes
compact outputs; `launch.sh` is **not** needed for reporting.
On 2026-11-28 the completion marker was written at 14:27:31 +08. At
14:28:18 +08 worker 417055 and its original logger had exited; physical A1
had 0 MiB in use and no compute PID. The final audit streamed only the three
compact artifacts to the local archive, without writing another A800 log.
No successor, holder, or dependency on retaining the A800 result path.

## Independent runner check on n164 H0

The [six-case registration](replay-six.json) selected the second index of each dataset within each frozen cohort, disjoint from the nine above. On Torch 2.9.0+cu128, all 12 plain/off comparisons matched in full, including output IDs, score, physical projections and encoder inputs. The off/on comparison below is **not** compared across GPU stacks or combined with the earlier nine; it is a GPU runner qualification, not frozen90. [Verified compact evidence](replay-six-summary.json).

| Fixed Q (6 cases each) | Correct off/on | Macro TPF off/on | TE off/on | Total F off/on | Decoder Q off/on |
|---|---:|---:|---:|---:|---:|
| 32 | 5/5 | 6.073/6.447 | 17.01%/22.71% | 597/471 | 17,386/13,584 |
| 64 | 5/5 | 11.749/11.749 | 21.52%/21.43% | 171/171 | 9,733/9,774 |

At Q32 all 126 saved forwards came from MATH index 51, **incorrect on both arms**, with N increasing 1,919 to 2,046. The other five saved zero F; index 31 produced one more token. At Q64 all six outputs were unchanged. No case was rescued or lost at either offer. This warrants checking concentration, not adding prefix refresh to a default policy.

The [complete Q32 paired registration](q32-frozen90.json) calls for 90 fresh triplets, disjoint reporting of the previously screened 15 and remaining 75, and a sensitivity analysis for #51 without excluding it from any primary result. The separate Q64 qualifications do not support a further standalone Q64 run. These controls cannot demonstrate that a native T2T request path is correct.

## Fixed Q32 confirmation

All 90 triplets (270 requests) were rerun on the same n164 H0. All pristine
and observation-only runs agreed. Against the historical same-stack capped
IBS, **90/90** also agreed on output IDs, events, trace and N/F/Q bill.
Observed peak was 32 in both arms. The treatment was triggered on 70/90
requests, spent 1,830 extra prefix rows at the moments it triggered, changed
33 final outputs, rescued one scored answer and lost none.

| Frozen cases | Correct off → on | Macro TPF off → on | Token efficiency off → on | Total N off → on | Total F off → on | Selected decoder Q off → on |
|---|---:|---:|---:|---:|---:|---:|
| All 90 (primary) | 79 → 80 | 6.840 → 6.821 | 19.33% → 19.43% | 36,118 → 35,754 | 6,287 → 6,148 | 186,829 → 184,058 |
| Original 30 | 28 → 28 | 6.745 → 6.676 | 20.20% → 19.91% | 9,988 → 10,222 | 1,656 → 1,718 | 49,445 → 51,336 |
| Added 60 | 51 → 52 | 6.888 → 6.894 | 19.02% → 19.24% | 26,130 → 25,532 | 4,631 → 4,430 | 137,384 → 132,722 |
| Previously screened 15 (now on H0) | 12 → 12 | 6.793 → 7.081 | 19.94% → 23.67% | 8,008 → 7,082 | 1,368 → 1,018 | 40,161 → 29,918 |
| Other 75 (primary confirmation) | 67 → 68 | 6.849 → 6.769 | 19.17% → 18.60% | 28,110 → 28,672 | 4,919 → 5,130 | 146,668 → 154,140 |
| Without #51 (sensitivity only) | 79 → 80 | 6.867 → 6.823 | 19.64% → 19.24% | 34,199 → 33,708 | 5,853 → 5,840 | 174,131 → 175,197 |

The only rescue was HumanEval #78, N 251 → 322 and F 19 → 41. The older A1
pilot rescue on #50 is not replicated on H0: both arms got it right with
identical N/F. #51 remained wrong (434 → 308 F); it accounts for 126 of
the 139 **net** saved forwards. The primary macro TPF nevertheless decreased:
each request, not each forward, receives the same weight. Among the other
75, the treatment spent 211 more F and 7,472 more decoder Q. These data do
not warrant increasing the default prefix dose or claiming a general T2T
benefit.

Descriptive paired, stratified 95% bootstrap intervals for the **all-90**
changes: quality [0, +3.33] percentage points, mean TPF [-0.200, +0.135],
pooled TE [-2.57, +3.09] percentage points. For the 75: quality [0,+4.00],
TPF [-0.266,+0.084], TE [-3.28,+2.87] pp. The other 75 were not in these
two pilots, not a globally untouched validation set.
[Verified metrics and receipt digests](q32-frozen90-summary.json) ·
[Numeric figure](q32-frozen90.png) ·
[Replot script](../../../../../../tools/plot_ibs_bridge_confirmation.py).

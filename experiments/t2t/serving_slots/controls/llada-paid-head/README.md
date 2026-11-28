# LLaDA Q64: projected but unsampled unresolved head positions

**Transfer to Q16, not the Q16 paid-waiting sampler.** [Sealed A7 fixed90](q16-agreement-frozen90.json).
Add only Q64's two-consecutive-paid-proposals acceptance to the original
Q16 active-window control. Every historical control matched. On 55 exposed
requests the rule added 328 admissions. Correct remained 70/90 (original30
21→23, new60 49→47), macro TPF 2.933→2.992, TE 21.05%→20.61%.
Total F 21,720→21,309. Do not transfer the Q64 policy to Q16 by default:
allowing the *already paid waiting MASKs* into the ordinary sampler is a
different, stronger Q16 intervention (75/90, 3.356 TPF, 25.60% TE).

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

## Single-factor warm-head sample: paired summary (Q64, half gate)

Inputs are cluster-side receipts only:

* pilot: `/gs/results/llada-hfull-single-factor-20261128-031321` (0, 10, 20)
* pinned remainder: `/gs/results/llada-warm-head-rest-pinned-1128` (87)
* excluded: `/gs/results/llada-hfull-single-factor-20261128-030853`
  (3 rows; index 10 failed before scoring because the first deployment used the
  base interpreter without `math_verify`; excluded, not replaced)

The two included runs share one protocol, checkpoint, samples SHA, Q64, half
gate, `coverage="head"`, and `old_source="run"`. The failed first run is
recorded as metadata only and is never loaded.

Common metrics: per-request TPF = `N/F` (in
`paired-warm-head-single-factor.csv`); **macro TPF = mean(N_i/F_i)** is the
primary token-rate readout; **token efficiency = pooled TE (`ΣN/ΣQ`)**.
Macro TE (`mean(N_i/Q_i)`) and pooled TPF (`ΣN/ΣF`) are different descriptive
statistics and do not replace the two primary metrics. `F` is the **self-developed executor's actual forward
count**, not an extra native-decoder measurement; `Q` is the physical decoder
row count under the **strict cap** (`q ≤ 64` and `layer_physical_q == [q]×20`
on every forward; max `Q/forward = 64`). Both arms are strict-cap on every
cohort.

| Cohort | Coverage | Old correct | New correct | Old F | New F | Old Q | New Q | ΔQ | Old macro TPF | New macro TPF | Old macro TE | New macro TE | Old pooled TPF | New pooled TPF | Old pooled TE | New pooled TE | Strict cap | Intervened | First checks (declared / independent) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| original30 | 30/30 | 19 | 21 | 5258 | 4211 | 280606 | 222430 | -58176 | 5.293 | 6.006 | 0.1051 | 0.1179 | 2.924 | 3.438 | 0.0548 | 0.0651 | yes | 30 | 30 / 30 |
| new60 | 60/60 | 42 | 41 | 7532 | 7754 | 389411 | 410258 | +20847 | 6.109 | 6.886 | 0.1288 | 0.1402 | 3.605 | 3.624 | 0.0697 | 0.0685 | yes | 58 | 58 / 58 |
| all90 | 90/90 | 61 | 62 | 12790 | 11965 | 670017 | 632688 | -37329 | 5.837 | 6.592 | 0.1209 | 0.1328 | 3.325 | 3.558 | 0.0635 | 0.0673 | yes | 88 | 88 / 88 |

All three cohorts are complete; there are no missing indices and no failures in
the included 90. The two non-intervened requests are 75 and 76; both arms are
correct with identical F and Q. Paired flips are 10 false→true and 9 true→false
overall (original30: +4 / -2; new60: +6 / -7). The first intervention has
identical pre-state and identical physical rows in 88/88 cases; the independent
trace re-check agrees with the declared check in 88/88.

This is a **paired single-factor trajectory cohort**, not a frozen90 quality or
TPF qualification. The coverage is complete, but the only contrast is
`warm_head_sample` at Q64 half gate; do not read the 90-case scores as a new
overall baseline.

## Gate single factor: half -> quarter (Q64, full-head)

Inputs:

* control 0..59: `/gs/results/llada-hfull-paired/rows`
  (`c64-full-half` vs `c64-full-quarter`, trusted same version)
* GPU5 60..89: `/gs/results/llada-hfull-gate-single-factor-20261128-050055`
* same-device 0/10/20 pilot:
  `/gs/results/llada-hfull-gate-single-factor-20261128-045813`

The 0/10/20 pilot matches the control on all 6 arm checks (output IDs, trace
ignoring `pre_state`, F, Q and correctness). `F` is the self-developed
executor's actual forward count; `Q` is physical rows under the strict cap
(`q ≤ 64`, `layer_physical_q == [q]×20`; max `Q/forward = 64`). Strict cap holds
on every cohort. Pre-state equality at tick 0 is verified on the 30 GPU5 rows
(60..89); the control 0..59 has no `pre_state`, and the 0/10/20 pilot is the
same-version cross-check.

| Cohort | Coverage | Old correct | New correct | Old F | New F | Old Q | New Q | Old macro TPF | New macro TPF | Old macro TE | New macro TE | Old pooled TPF | New pooled TPF | Old pooled TE | New pooled TE | Strict cap | Diverged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| original30 | 30/30 | 19 | 24 | 5258 | 3598 | 280606 | 163115 | 5.293 | 5.643 | 0.1051 | 0.1228 | 2.924 | 3.917 | 0.0548 | 0.0864 | yes | 27 |
| new60 | 60/60 | 42 | 46 | 7532 | 6422 | 389411 | 292229 | 6.109 | 6.596 | 0.1288 | 0.1469 | 3.605 | 4.288 | 0.0697 | 0.0942 | yes | 56 |
| all90 | 90/90 | 61 | 70 | 12790 | 10020 | 670017 | 455344 | 5.837 | 6.279 | 0.1209 | 0.1388 | 3.325 | 4.155 | 0.0635 | 0.0914 | yes | 83 |

All 90 are present with zero failures. The gate change flips 14 false→true and
5 true→false (net +9 correct), reduces executor F by 21.7% and physical Q by
32.0%, and raises macro TPF by 0.442 and macro TE by 0.0179. Seven of 90 did not
diverge at all (the gate never opened in either arm). This is a **gate
single-factor paired cohort**, not a frozen90 quality/TPF qualification.

## Quarter gate + warm-head sampling (Q64, full-head)

Inputs:

* old quarter exact replay: `/gs/results/llada-hfull-quarter-old-rows`
  (0–67 control `c64-full-quarter`; 68–89 the gate run's new quarter)
* new quarter + warm-head: `/gs/results/llada-hfull-quarter-warm-single-factor-20261128-053736`
* reference prompts: `/gs/work/llada-refs/llada21`

Original-encoding verification: all 90 old-quarter `output_ids` and traces match
the source receipts; all 90 old/new `input_sha256` match the reference
`prompt_ids`; all 90 old/new use the same prompt; `samples_sha256` is the
registered `bc9e457d…ab65aed`. 77/90 new output IDs differ; 13 are identical
(including non-intervened 75 and 76), which is not a failure. Zero failures in
the included 90.

| Cohort | Coverage | Old correct | New correct | Old F | New F | Old Q | New Q | Old macro TPF | New macro TPF | Old macro TE | New macro TE | Old pooled TPF | New pooled TPF | Old pooled TE | New pooled TE | Strict cap | Intervened / first |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| original30 | 30/30 | 24 | 25 | 3598 | 3223 | 163115 | 145807 | 5.643 | 6.387 | 0.1228 | 0.1347 | 3.917 | 4.110 | 0.0864 | 0.0909 | yes | 30 / 30 |
| new60 | 60/60 | 46 | 47 | 6422 | 6024 | 292229 | 271209 | 6.596 | 7.275 | 0.1469 | 0.1603 | 4.288 | 4.680 | 0.0942 | 0.1040 | yes | 58 / 58 |
| all90 | 90/90 | 70 | 72 | 10020 | 9247 | 455344 | 417016 | 6.279 | 6.979 | 0.1388 | 0.1517 | 4.155 | 4.482 | 0.0914 | 0.0994 | yes | 88 / 88 |

Net +2 correct (7 false→true, 5 true→false), executor F -7.7%, physical Q
-8.4%, macro TPF +0.700, macro TE +0.0129. The two non-intervened requests 75
and 76 are both correct with identical F and Q. This is a **single-factor
interaction** on all 90 frozen requests, not a mixed-budget policy ranking.
The same-cohort historical IBS64 is 72/90, macro TPF 4.397 and pooled TE
14.56%; Flow W2 is 74/90 and 10.006, without a verified selected-history-row
efficiency denominator. Each of the 180 raw trajectories also passed a direct
20-layer width, position, cap and F/Q reconciliation (the absence of a trace
or layer widths cannot qualify).

### Where Q64 is spent

[row-bill.csv](row-bill.csv) partitions all physical decoder rows into old
write debt, first-visit MASK, warm MASK, clean block-local prefix, and all
other projected rows. The full-head projection includes clean rows regardless
of the nominal sparse prefix quota: a prefix visit here is not necessarily
an extra row purchased for that purpose. All traced calls had nonzero prefix
KV length. Each sampled edit is attributed once to the input row's role;
reversions to MASK are not T2T. Tail rows and sampled rows are overlays, not
additional costs. The two half-gate controls independently agree on the IDs,
scores and all role counts for every matched request.

| Gate and sampler | Mean physical Q/F | Head Q | Tail Q | Mandatory Q | Clean prefix Q | T2T on clean prefix / all T2T |
|---|---:|---:|---:|---:|---:|---:|
| Half, reference | 52.39 | 402,440 | 267,577 | 47,680 | 156,346 | 2,158 / 3,748 |
| Quarter, reference | 45.44 | 314,210 | 141,134 | 45,153 | 121,252 | 1,125 / 2,076 |
| Half, allow warm head MASK sample | 52.88 | 376,912 | 255,776 | 47,250 | 133,945 | 1,825 / 3,224 |

Reproduce on the receipt host using `tools/llada_gate_row_bill.py` with
`--control`, `--gate`, `--warm-pilot`, `--warm-rest` pointing respectively to
the original four-arm root, the gate complement and the two warm-head roots
above, and `--manifest benchmarks/frozen90/manifest.json`. Only cohort counts
are exported; these are whole trajectories, not a same-state counterfactual.

### What paying for a full second block could buy

On the **quarter gate plus warm-head sampling** profile, all 90 sequences paid
417,016 decoder rows over 9,247 forwards. The ceiling of 64 per call leaves
174,792 unused seats. Replaying admissions and revocations in order from the
recorded states partitions them, without inference or assumptions about logits:

| Reason a Q64 seat was unused | Seats | Share of unused |
|---|---:|---:|
| Second block not open to sampling (includes suppression after head EOS) | 151,904 | 86.91% |
| Partial prompt/output block, no generated position for this seat | 7,900 | 4.52% |
| Tail open, already decoded and clean, some tail MASKs remain | 10,414 | 5.96% |
| Tail open and all tail tokens already decoded | 4,574 | 2.62% |
| Tail open, unbuilt or still MASK and not selected | 0 | 0% |

[Original30, added60 and frozen90](unused-quarter-warm.csv). `open_F` counts
eligibility, even when no tail rows are selected. Only 1,782 of 4,450 eligible
forwards had any untaken tail position. Unlike tail debt, an untaken clean
row is not a correctness violation: its KV is retained. None of these counts
predict an accuracy or speed gain. Reproduce with `tools/llada_gate_row_bill.py
--quarter-warm /gs/results/llada-hfull-quarter-warm-single-factor-20261128-053736
--manifest benchmarks/frozen90/manifest.json`.

Fixed-Q64 original30 GPU experiment: unchanged `quarter_edit_warm` Q64 control;
on warm sample calls with both blocks resident and MASKs remaining in the
eligible successor, use otherwise unused seats to project its remaining clean
rows; either discard the supplemental logits or allow T2T/revocation using the
same measured rows. Do not fill a finished block, open a block earlier, replace
any incumbent MASK, relax thresholds, or introduce a new sampling forward.
The three arms are independent full request trajectories; work and final
output may diverge after the first treatment. See `llada_frontier_seat.py
--factor successor` and `tools/run_llada_small_geometry_a800.sh successor DEADLINE 64`.

**Result: leave the spare clean successor seats alone.** Same physical A1,
all original30, 27 genuinely exposed; the first change checks the same
pre-state, head rows, incumbent MASKs and clock. Only the third arm may
edit added rows. Every physical decoder width was at most 64.

| Fixed Q64, original30 | Correct | Macro TPF | TE | Paid F | Selected Q |
|---|---:|---:|---:|---:|---:|
| Quarter gate + warm-head control | 25 | 6.387 | 9.09% | 3,223 | 145,807 |
| Also project clean successor rows | 23 | 6.252 | 8.62% | 3,402 | 156,376 |
| Also allow edits on those same rows | 22 | 6.186 | 7.86% | 3,981 | 183,327 |

[Paired results and hashes](../llada-slot-allocation/c64-successor-original30.json).
Read-only rescued 1/lost 3 versus control. Editing rescued 2/lost 3
versus read-only; 154 T2T on 4,937 supplemental projections, no revocations
on those rows. Longer trajectories also changed other work. These small
paired differences do not establish a universal harm, but there is no
reason to expand either variant. Flow W2 on the same original30 was
23 correct, macro TPF 9.721 (external, no comparable TE).

**Next fixed Q64 registration: two paid proposals.** Keep the quarter gate,
full head, warm-head sampling, strict cap, current fallback and T2T rule.
Change only M2T admission on the resident current head: in addition to the
native threshold and fallback, accept an unresolved row when the preceding
physical forward in this same head also sampled that unresolved row and both
proposals named the same non-MASK token at measured confidence >0.5. Both
passes must have started with the full valid head resident; a drain, head
change, missed row or different proposal breaks agreement. No second full
logits buffer or new projection. The memo holds at most 32 token IDs.

Run a separate constant-Q arm for every complete request on A7, alternating
arm order. Gate on real change and exact original-control replays on 0/10/20,
then finish original30 and new60 on the frozen90 manifest. Validate every
M2T on both arms from the paid proposals; the first difference must preserve
rows, proposals, clock, debt before the call, and incumbent edits. Count
nonconvergence as failure, never replace tasks. Compare frozen90 quality,
macro TPF and pooled TE; do not promote over the present profile if quality
falls or the forward-rate gain does not repeat on new60. `tools/run_llada_agreement_a800.sh`
starts only after the switch smoke test is valid and A7 is vacant.

**Paid repeat: sealed original30, not yet frozen90.** [Paired source and
bills](q64-agreement-original30.json). The unchanged quarter-gated control
exactly reproduced all 30 independent reference outputs and their bills.
27/30 requests experienced a new admission. Of 5,341 eligible paid
opportunities on treatment trajectories, 1,340 received the second
confirmation. These are additional *acceptances*, not additional Q.

| Fixed Q64, original30 | Correct | Macro TPF | Pooled TE | Paid F | Selected Q | Final N |
|---|---:|---:|---:|---:|---:|---:|
| Quarter gate + warm-head control | 25/30 | 6.387 | 9.09% | 3,223 | 145,807 | 13,247 |
| Same, admit on consecutive paid agreement | 28/30 | 7.271 | 10.83% | 2,675 | 120,244 | 13,022 |
| External Flow W2, same tasks | 23/30 | 9.721 | n/a | | | |

Three rescued, zero lost, zero execution failures; paired 95% descriptive
intervals: quality delta [0,+20] pp, macro TPF [+0.479,+1.280]. The new60
cohort is still required by the preregistration. This is a positive early
signal relative to the matched control, not a win on both metrics over
Flow W2 and not yet a reason to change the default profile. Only M2T
acceptance changed; no T2T or native fallback threshold was relaxed.

**Prospective transfer gate, fixed Q16.** Only if the unchanged Q64
original30 controls all replay, neither original30 nor new60 loses net
quality, both arms have no failures, macro TPF increases on new60 and
frozen90, and pooled TE does not decrease on frozen90, follow on A7 with
a separate Q16 trial. The source control is the sealed, same-manifest Q16
active window (A5), not the old sparse head or the tail-borrowing family.
Repeat the exact Q64 consecutive-proposal rule, including its fully
resident current-head requirement; no new seats, change in cold sampling,
or threshold tuning. Both fixed Q16 arms run on A7, alternating order;
test out-of-device control replay on all 90, and stop if the 0/10/20
anchors differ or have no actual intervention. Report exposed requests,
admissions and complete paired results. This is not a mixed-budget test.
`bash tools/run_llada_repeat_transfer_a800.sh DEADLINE` will not claim or
hold a GPU when the qualification fails.

**Final fixed Q64, all 90.** [Paired receipt](q64-agreement-frozen90.json);
all original-30 reference bills and token IDs replayed, no failed requests.

| Cohort | Correct | Macro TPF | Pooled TE | F | Selected Q |
|---|---:|---:|---:|---:|---:|
| Original 30 | 25 → 28 | 6.387 → 7.271 | 9.09% → 10.83% | 3,223 → 2,675 | 145,807 → 120,244 |
| New 60 | 47 → 47 | 7.275 → 7.873 | 10.40% → 10.89% | 6,024 → 6,153 | 271,209 → 278,807 |
| Frozen 90 | 72 → 75 | 6.979 → 7.672 | 9.94% → 10.87% | 9,247 → 8,828 | 417,016 → 399,051 |

79 exposed, 4,504 extra admissions on already paid Q, 7 rescued / 4 lost.
All three net quality gains came from the original30. On new60, 4 gained and
4 lost; F and Q rose along with output length. Do not equate a higher macro
TPF with fewer total calls on that cohort. Whole-cohort paired 95% intervals
are [−3.33,+10.00] pp and [+0.386,+1.005] TPF. The Q16 transfer gate passed.
Flow W2 is 74/90 and 10.006 TPF; the remaining TPF gap is 2.334.

The opt-in `quarter_window_repeat64` keeps the qualified low-tier window
and Q32 close; only Q64 adds the measured admission rule. A lower offer,
cold head, missed row, intervening drain or head change breaks the pair.
The [completed GPU qualification](../strict-handle/llada-quarter-window-repeat64.json)
verified fixed equality on all five offers, all 25 directed transitions,
live votes discarded on downshift, and actual payment of parked writes.
It does not measure mixed-budget performance or quality.

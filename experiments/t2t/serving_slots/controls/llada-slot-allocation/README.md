# Low-offer LLaDA row selection

## Strong control: use only leftover IBS8 seats for clean prefix

[Sealed n164 fixed90 replay audit](c8-ibs-spare-confirmed.json). Strict IBS8
keeps precisely its original sampler seats. Only unused seats, at most two,
may additionally reproject the rightmost eligible positions of the decoded
prefix inside the current block. Their logits do not participate in edits.

| Fixed Q8 | Original30 | New60 | Frozen90 | Macro TPF | Token efficiency | F | Selected Q |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strict IBS8 | 25 | 47 | 72 | 2.3645 | 29.36% | 19,692 | 145,324 |
| Same + spare prefix | 27 | 46 | 73 | 2.3866 | 28.63% | 18,753 | 143,776 |

5 rescues, 4 losses. Paired quality score interval (95%): -6.14 to +8.48
percentage points; macro TPF delta interval: -0.0451 to +0.0887. Do not
promote solely on the aggregate: original30 and new60 have opposite quality
deltas. 5,437 supplemental positions were actually computed on 2,938 rounds
across 89 requests. The change in total Q includes subsequent trajectory
changes; supplementation did **not** reduce the immediate matched row count.

The control physically projected 44,361 mandatory rows, 82,829 discretionary
MASK rows, 16,561 discretionary decoded generation rows, and 1,573 rows from
partial prompt blocks; 12,212 offered seats remained unused. In 1,835 of
19,692 forwards, all eight seats were mandatory. Among 206/263 recorded edits
(control/trial) no edit revoked a token to MASK. That is an observation, not a
permission to drop such debt on other inputs: the historical IBS capped writer
only registers edits whose new token is not MASK. This isolated experiment is
not a qualification for importing it wholesale as a switchable request.

### What does the strict window spend its Q on?

[Per-cohort counters, four tiers](ibs-paid-ticks.json). Each tier contains
the same 90 requests, and every charged post-prefill call on each request.
No partial prefix cache or model replay is inferred. A cold MASK is a selected
MASK whose absolute position had never been projected; a warm MASK was
projected previously and is still unresolved. An admission counts a MASK
becoming non-MASK, not a final delivered token. An edit changes an already
decoded position. These are different numerators.

| Fixed IBS Q | Dirty / actual Q | Warm admissions / all admissions | Warm admissions on calls also projecting cold MASKs | Edits on mandatory rows / all edits | All offered seats mandatory: F / total F |
|---:|---:|---:|---:|---:|---:|
| 4 | 41,364 / 105,885 | 9,273 / 40,915 | 4,534 | 441 / 455 | 4,137 / 27,397 |
| 8 | 44,361 / 145,324 | 14,740 / 44,155 | 6,396 | 150 / 206 | 1,835 / 19,692 |
| 16 | 44,098 / 206,927 | 20,367 / 43,771 | 6,739 | 213 / 328 | 584 / 15,480 |
| 32 | 42,876 / 268,170 | 22,908 / 42,331 | 0 | 386 / 547 | 178 / 12,694 |

At Q32 all cold/warm MASK mixing disappears on this control, **not** warm
admissions. The best interpretation is to test the use of already-projected
MASKs in Q8/Q16 mixed calls, not to add extra projections or globally loosen
the sampler. Mandatory writes and useful edit opportunities overlap; a new
projection just for an edit is not always necessary. Counts do not establish
the benefit of allowing any particular edit, and are not quotas.

Generation failures (Q4/8/16/32: 6/0/1/2) stay in the relevant 90. Their
completed physical forwards are included; terminal unpaid historical writes
are 6/0/1/2 rows respectively and are not fictitiously charged. No historical
token-to-MASK revocations were observed in these four cohorts. Reproduce on
n164 with `tools/audit_llada_ibs_ticks.py PREFIX90 FROZEN90_MANIFEST`; the
audit checks each per-task bill against the independently scored ledger.

`tools/audit_ibs_spare.py RUN_ROOT MANIFEST` on the receipt host checks
source and return seals, per-request IDs and per-forward debt, bills and
cohort totals. No new model inference is required.

## What differs from strict IBS?

[Measured selection audit](geometry.json). Replay the pinned hard-capped IBS
candidate order and chosen rows against **every real generation tick** on the
same frozen90 cases: Q8: 19,692 ticks; Q16: 15,480. No model or logits are
run for this audit. A candidate retains decoded positions after the frozen
prefix, then the earliest MASKs; actual rows are hard-clipped with debt first.

In a separate, single-step calculation, apply that row rule to the GPU-measured
head-only Request trajectories. Reconstruct the stable prefix on this state up
to the first mask or outstanding write. Compare only full 32-position generated
heads with masks; exclude the first prompt fragment, the last partial block
and verification-only work. This is **not** an IBS rollout or a quality result.

| Fixed Q | Eligible Request ticks | Different Q sets | Rows only the shadow IBS rule selects | Of those, already-decoded non-dirty interior rows |
|---:|---:|---:|---:|---:|
| 8 | 48,619 | 45,492 | 251,108 | 214,851 (85.56%) |
| 16 | 28,165 | 26,077 | 182,591 | 150,195 (82.26%) |

At Q8 the Request-only rows were exclusively unresolved MASKs. The denominator
of 85.56% is **shadow-only rows**, not all decoder Q. The right edge seen by the
current executor is also different from the rightmost actual selected row in
31,791/48,619 Q8 calls. That is a separate candidate factor, not adjusted in
the following study. Source roots on n164 under `gx-erqw-20261123/runtime`:
`t2t-native-1125/prefix90/llada21-optimus-hardcap-endpoints`,
`layout-1128-v1/cross/results/rows`. Use `llada_selection_geometry.py` with
`--ibs`, `--request`, `--manifest`; the report hashes the ordered source files.

## Paired interventions

`llada_frontier_seat.py` runs the normal measured fixed evaluator on each arm.
No serving scheduler, full-logit oracle or budget override is used.

- `--factor frontier`: on an incomplete head with at least two selected cold
  MASKs and a waiting earlier warm MASK, swap the last cold seat for the earliest
  waiting warm MASK. Keep both the projection width and sampler arity on that
  intervention. Q8 A800 (`/gs/results/llada-frontier-seat-q8-1128`); Q4 H800
  (`runtime/llada-frontier-q4-1128/results`). Recheck 0/10/20 before expanding.
- `--factor interior`: only on a fully resident current head and with at least
  two selected non-dirty MASKs. Three arms: original; exchange the last such
  MASK for the earliest clean decoded row after the first MASK and discard its
  logits; exact same exchange and permit T2T on its paid logits. The first
  contrast changes which position is projected and forgoes one MASK sample;
  the second changes only whether one paid row may edit. There are no additional
  Q or sampling-clock advances at the matched intervention. Same visibility,
  no change to optional clean prefix or dirty debt at matched state. Q8 A800
  (`/gs/results/llada-interior-q8-1128`).

Only complete paired cohorts can establish an outcome. Do not infer that the
interior intervention wins from the high frequency of a shadow planner's choices.

### Original30: one cold seat for a waiting MASK

| Fixed Q8 (original 30) | Correct | Macro TPF | Pooled TE | Total F | Output N | Selected Q |
|---|---:|---:|---:|---:|---:|---:|
| A0 head-only control | 11 | 1.635 | 16.28% | 16,220 | 18,468 | 113,469 |
| Same A0, one seat exchanged | 14 | 1.865 | 20.51% | 12,663 | 17,731 | 86,431 |
| Historical strictly capped IBS8 | 25 | 2.265 | 28.36% | 6,652 | 13,907 | 49,029 |

All 30 pairs were exposed, and all 30 output sequences changed (6 rescues,
3 losses, zero failures). The waiting MASK was admitted on 1,817/2,083 exchanges;
this does not imply a causal admission probability on other trajectories.
The fully measured [paired report](c8-frontier-original30.json) improves over
its own control, not over [strict IBS8](../llada21-q8-ibs-original-vs-capped.csv).
The [same A0 extension to all 90](c8-frontier-frozen90.json) passed exact
replays on 0/10/20 before accepting the other original pairs. New60 correct
went from 19 to 32. For frozen90 the two arms had **30 vs 46 correct,
1.681 vs 1.913 macro TPF and 16.54% vs 21.04% TE**. Paid F fell
47,198 to 38,567 and selected Q fell 329,926 to 262,743. Of the 90 tasks,
89 were exposed; 23 were rescued and 7 lost. The descriptive paired 95%
intervals were [+6.67,+28.89] percentage points and [+0.130,+0.336] TPF.
The A0 control is not the independent H4 control (29 correct); neither
version catches the frozen90 strict IBS8 (72 correct, 2.364 TPF, 29.36% TE).

At [fixed Q4 on H0, full90](c4-frontier-frozen90.json), exchanging that
single seat improved 8 to 17 correct on original30 and 10 to 30 on new60.
Overall: **18 to 47/90**, macro TPF **1.182 to 1.299**, TE **27.22% to
31.16%**, with 32 rescues/3 losses, 89 exposed, and zero failures. Strict
IBS4 is still ahead at **69/90, 1.395, 35.66%**. The Q4 result is for the
one-seat variant, not for the active-window experiment below.

### Fixed-width active-window experiment

Separate fixed Q8 and Q16 studies on the original30. Reuse the head-only
executor and keep its per-state real projection width, global dirty rows,
ordered clean-prefix quota, confidence thresholds and cache-visibility rule.
At ordinary sampling ticks, rank remaining seats within the span starting
at the first MASK: retain eligible decoded interior positions and the earliest
unresolved candidates, clipped in physical order to the remaining slots.
A mixed cold call samples only cold positions; a warm call may edit its paid
rows under the existing rule. As a result of different row selection, a
forward may turn from cold to warm or move its rightmost newly initialized
position. Those are effects of the policy, not identical attention inputs.
This is not an exact IBS reproduction: the protected prefix and width, full
resident history, prompt prefill and block retirement are still ours.
First intervention must match input state, width, phase, dirty and prefix,
with physical width checked on every layer on **every** step. Run via
`tools/run_llada_small_geometry_a800.sh window DEADLINE 8` or `16`.

### Original30: active window versus one interior seat

These are fixed decoder offers. Every row covers the same 30 cases. The
strict IBS rows are external references, not the within-device controls.

| Fixed offer and policy | Correct / 30 | Macro TPF | TE | Paid F | Paid Q |
|---|---:|---:|---:|---:|---:|
| Q8 head-only control, A6 | 11 | 1.635 | 16.28% | 16,220 | 113,469 |
| Q8 active window, A6 | 24 | 2.210 | 29.32% | 7,700 | 52,116 |
| Q8 strict IBS | 25 | 2.265 | 28.36% | 6,652 | 49,029 |
| Q16 head-only control, A5 | 14 | 2.495 | 12.06% | 12,229 | 146,300 |
| Q16 active window, A5 | 21 | 2.799 | 20.47% | 6,974 | 71,813 |
| Q16 strict IBS | 23 | 3.189 | 19.32% | 5,278 | 70,484 |

[Q8 pair and row breakdown](c8-window-original30.json);
[Q16 pair and row breakdown](c16-window-original30.json);
[external Q16 reference](../llada21-q16-ibs-original-vs-capped.csv).
Q8 rescued 13/lost 0; Q16 rescued 9/lost 2. Within-pair macro TPF changes
and descriptive 95% intervals were +0.576 [0.413,0.738] and +0.304
[0.009,0.598]. Intervals condition on these tasks, with 4,000 within-dataset
paired bootstrap draws, seed 1128. Original30 triggered extension and is not
independent confirmation. The Q16 control is head-only; the published Q16
handle borrows tail seats and is not the matched control here.

We also paired the window against strict IBS by exact prompt-token digest,
index, dataset, seed and task ID on the sealed reference ledger. At Q8 the
window rescued 2/lost 3, with macro TPF difference -0.055 [-0.206,+0.101].
At Q16 it rescued 2/lost 4, macro TPF difference -0.389 [-0.574,-0.185].
Q8 is not established equivalent; Q16 still trails IBS in TPF on this
cohort. No same-device causal attribution between different executors.
Generate the compact IBS reference using `llada_seat_report.py PREFIX90
--export-reference`, then supply the JSON to the same reporter on A800 via
`--ibs`; input hashes and per-task scores/bills travel, not prompt/output IDs.

At Q8, waiting-MASK projections over complete trajectories fell from 75,765
to 14,725. The window also paid for 5,944 clean interior rows, with 15 edits
or revocations on these rows. At Q16 it paid for 12,630 interior rows with
53 edits or revocations. Changing the window changes subsequent cache
visibility, MASKs, lengths and debt. These counts do not isolate an editing
effect, and a decoded row after the first MASK is not a decoded prefix.

The [Q8 single-seat study](c8-interior-original30.json) tests a different
question. On A1, unchanged / exchange a MASK for one read-only interior row /
allow the same interior row to edit scored **11 / 8 / 13**, macro TPF
**1.635 / 1.498 / 1.629**, TE **16.28% / 15.36% / 16.74%**. Editing
vs read-only rescued 5/lost 0, with 91 actual interior writes. No extension
of the single-seat factor on this evidence.

**Follow-up:** Q8 and Q16 each extend on their original GPU. Both original
arms must exactly replay the input, first change, outputs and bills on
0/10/20 before inheriting the 27 other original pairs. Use
`tools/run_llada_small_geometry_a800.sh window90 DEADLINE 8` or `16`.
The [sealed Q8 extension](c8-window-frozen90.json) changed **19 to 48/60**
on new60 and **30 to 72/90** overall (42 rescued, 0 lost vs the matched A6
control). Frozen90 macro TPF increased **1.681 to 2.300** and pooled TE
**16.54% to 29.94%**. Paid F decreased 47,198 to 22,684 and Q decreased
329,926 to 153,616. Against the external strict IBS8 **72/90, 2.364 TPF,
29.36% TE**, 6 cases were rescued and 6 lost; the paired TPF difference
is -0.064 [-0.140,+0.007]. Equal aggregate accuracy is not identical
answers, and an interval crossing zero is not a proof of equivalence.

The [sealed Q16 extension](c16-window-frozen90.json) reproduced the improvement
on new60: 35 to 49 correct, 2.712 to 2.999 macro TPF and 12.71% to 21.34% TE.
Across frozen90 it moved from **49 to 70 correct**, **2.639 to 2.933** TPF,
**12.48% to 21.05%** TE; F fell from 33,477 to 21,720, Q from 400,510 to
218,267. There were 25 rescues, 4 losses and no failed requests. Strict
IBS16 still leads on quality/TPF (72/90, 3.375, 20.54% TE). In the
window-minus-IBS paired comparison there were 6 rescues, 8 losses; the
descriptive 95% interval was [-11.11,+5.56] pp for accuracy and
[-0.636,-0.213] for macro TPF. On new60 alone the quality comparison was
49 vs 49, but TPF still trailed by 0.469. No demonstrated baseline parity.

Q4 is a separate 90-pair run without inherited cases, unchanged 1/8 law
(flooring to zero optional prefix seats), same window algorithm and three
exercised anchors before continuing. A0 must first finish the Q8 frontier
extension and be free. Failures count; retain separate original30 and
new60 results. No claim about wall time or mixed-offer quality.

**Paid warm MASK test (n164 H0, Q8).** Both arms use the window above. Only
when a single ordinary call already projects cold and resident unresolved
non-dirty rows does the treatment add those resident MASK rows to the
sampler. No extra projections, clean decoded rows, prefix or second block
are added at the matched intervention. Both arms advance the sampling clock
once; the extra candidate can compete in the per-block fallback. Register
0/10/20 then 90, requiring at least one genuine first exchange across the
three anchors with all 20 physical widths and the pre-state equal. Unexposed
anchors must be identical, never replaced. Use `tools/run_llada_admission_n164.sh
DEADLINE paid8`; wait for the complete paired report before claiming an effect.

**Idle interior read, fixed Q8/Q16 (A6/A5).** Preregistered after the
same-device full-window controls. At an ordinary sampling step only, keep
every incumbent Q and every sampler position. Spend otherwise unused seats
on already resident, clean, decoded positions strictly after the first MASK
in the current physical head, lowest absolute position first. They cannot
edit and cannot expand the visible KV boundary. A read may change the
incumbents' logits through attention. No clean-prefix dose change, row
exchange, successor opening, auxiliary forward or mandatory-only drain.
Compare against an unchanged fixed window on all 90 cases; exactly reproduce
the original device's control output digest and cost for every case and check
the identical pre-state at the first actual change. Report only complete
original30/new60/frozen90 cohorts. If 0/10/20 have no eligible read,
stop that tier without replacing any case. An improvement only in macro TPF
with worse quality or a large increase in total work is not an automatic
promotion. `bash tools/run_llada_window_spare_a800.sh Q DEADLINE` waits for
the known window job, checks the sealed parent, and exits on unrelated GPU
occupants; no holder or retry. Full requests and next-step budgets are fixed.

The first A5 launch of this idle-read factor exited before creating a run
directory or loading the model: its copied parent environment lacked the
current `paired` report helper. The restored helper is the repository version
at `3f39db8`; startup import was checked before the same registration was
relaunched. No requests or outcomes were replaced.

**Width versus output per selected Q.** On each of the 90 paired inputs,
`N/F = (Q/F) * (N/Q)`. Split the product difference symmetrically at the
two trajectories' means, *then* average over requests. This reconciles the
macro TPF; the yield in this identity is not the pooled headline TE.

| Window minus capped IBS | Mean Q/F, IBS → window | Width term | Yield term | Δ macro TPF |
|---|---:|---:|---:|---:|
| [Q8](c8-width-accounting.json) | 7.415 → 6.905 | −0.15652 | +0.09228 | −0.06424 |
| [Q16](c16-width-accounting.json) | 13.456 → 10.639 | −0.67691 | +0.23450 | −0.44241 |

At both tiers 88/90 requests ran narrower than their IBS counterparts.
This is descriptive across different output trajectories; it is neither
a latency model nor evidence that projecting arbitrary extra rows helps.

**[H0 mixed cold/warm Q8, complete](c8-paid-frozen90.json).** No rows
were added to the first changed forward. Only the sampler got access to
resident waiting MASKs whose logits that call already produced. On later
calls the KV state, number of forwards and selected rows can diverge.

| Same H800 window → allow paid waiting proposals | Correct | Macro TPF | Pooled TE |
|---|---:|---:|---:|
| Original 30 | 22 → 24 | 2.174 → 2.326 | 26.57% → 30.93% |
| New 60 | 49 → 46 | 2.385 → 2.474 | 30.95% → 32.74% |
| All 90 | 71 → 70 | 2.315 → 2.424 | 29.21% → 32.08% |

No failed requests. 88/90 exposed, 5 rescues and 6 losses, 8,979 accepted
warm MASKs. Total F 24,290 → 21,623 and selected Q 164,184 → 146,571.
Paired descriptive 95% intervals: Δ macro TPF [+0.0482,+0.1744],
Δ correct rate [−7.78,+6.67] percentage points. This does not establish
quality noninferiority. The H0 control is not the A6 window cohort.

**Preregistered follow-up, H0 Q8.** Starting from the H0 *paid* sampler,
keep high-confidence acceptances on all the same measured MASKs. Only
when none pass 0.9, restrict the fallback contest on mixed calls to the
incumbent cold rows. All other calls, T2T, and all physical positions
are unchanged at the intervention. This tests whether low-confidence
competition, not merely extra proposals, matters. Reproduce the 90 H0
paid controls (token digest and bill), audit both samplers from their
actual proposals, and require at least one real first exchange among
fixed anchors 0/10/20 before expansion. If no exchange occurs the pilot
ends; never replace an anchor or count an ongoing run as evidence.

[Full Q8 fallback results](c8-fallback-frozen90.json). All 90 controls
replayed; 80 requests and 2,035 fallback decisions actually changed.
Six rescued, six lost. No failures. Do not adopt the cold-only fallback.

| Cohort | Correct, paid → restricted | Macro TPF | Pooled TE |
|---|---:|---:|---:|
| Original30 | 24 → 23 | 2.326 → 2.355 | 30.93% → 28.46% |
| New60 | 46 → 47 | 2.474 → 2.437 | 32.74% → 30.80% |
| Frozen90 | 70 → 70 | 2.424 → 2.410 | 32.08% → 29.90% |

The restricted arm spent 23,757 F / 161,076 Q vs 21,623 F / 146,571 Q.
The paired TPF difference interval is [-0.074,+0.042]; failure to improve
does not prove the two samplers are equivalent. This only tests the native
fallback on mixed Q8 calls. It does not change the Q16 or Q64 results.

**Preregistered separate Q8 spare-seat experiment (H0).** On the same
90 paid-sampling controls, when the primary head still has cold rows
in the current plan and there is unused capacity, add exactly the
leftmost *unselected, resident, non-dirty, unresolved* head position
to both the projection and sampler. Nothing is displaced. Unlike the
prior paid experiment, it costs an actual extra row; unlike the spare
clean-interior experiment, it can advance an outstanding MASK on that
call. Retain the original native thresholds, fallback and clock. Stop
after the unchanged 0/10/20 anchors if none exposes a real spare seat.
Otherwise report all 90, original30 and new60, plus extra rows and
admissions, failures and same-device control checks. No claim until
the relevant cohort is complete. Raw data: n164 runtime
`llada-window-spare-waiting-q8-1128/results`. Run via
`run_llada_admission_n164.sh DEADLINE spare_waiting8 PREDECESSOR_PID`.

Result: zero eligible admissions or projections on all three pilot anchors;
both arms' delivered IDs, bills and scores were identical. Stopped before
expansion as registered. No 90-case result and no evidence about other requests.

**Spare prefix on the active window, H0 Q8.** The H0 *original active
window* (not the expanded paid sampler, not IBS8) is the comparator.
Supplement a sampling call with up to two otherwise unselected clean,
resident, decoded generated prefix positions, nearest the first unresolved
position of the same physical block. Do not displace MASKs, widen visible
KV, change the sampler set, or edit supplemental rows. Independently
reconstruct the prefix from actual admissions and revocations. Replay the
sealed 90 same-device controls; expand only on real exposure at 0/10/20.
Source: `llada_window_spare`, flag `--factor prefix`; raw results:
`llada-window-spare-prefix-q8-1128/results`. Compare original30, new60 and
full90 quality/TPF/TE. Do not confuse extra immediately paid Q with a
whole-request Q difference, or transfer spare-IBS findings to this arm.

**Spare clean-interior reads, [Q16 completed](c16-spare-read-frozen90.json).**
The 90 same-device active-window controls replayed. Extra clean reads were
real for 83/90 requests, 1,833 supplemental Q in total. Original30
21 → 23 correct but new60 49 → 46; pooled 70 → 69/90, macro TPF
2.933 → 2.877 and TE 21.05% → 20.43%. Five rescues, six losses, no
failed requests. Do not adopt. These are clean seats **to the right**
of the first MASK, not decoded prefixes, and not the Q16 paid-MASK trial.

[Q8, same intervention](c8-spare-read-frozen90.json): 73/90 requests
had actual supplemental reads, 665 rows. Original30 fell 24 → 21,
new60 48 → 46; frozen90 72 → 67 correct (2 rescues, 7 losses),
macro TPF 2.300 → 2.284, TE 29.94% → 28.57%. Full trajectories cost
22,684 → 25,507 forwards and 153,616 → 172,511 selected Q. The 665
supplemental reads are counted at the calls making them, not the net
whole-trajectory difference. No failures in either arm.

**Contested seat, preregistered before launch.** Fixed Q8/A6 and Q16/A5,
old = sealed original active window on the same GPU (not the paid-MASK
variant). On a full, warm current-block sampling call, if a selected optional
clean prefix and an unselected resident MASK compete, exchange the farthest
selected prefix for the leftmost unselected waiting MASK. No sampling-only
change, no cold row, no debt replacement. Check identical width and visible
KV, identical sampling clock, independently reconstruct the two token states,
and require replay of every old output hash, bill and score. If none of
0/10/20 is exposed, stop without expanding. Otherwise include all frozen90
cases; record actual swaps, admissions, original30/new60 and all three metrics.
This tests displacement and is distinct from H0's extra read-only prefix.
Launch via `tools/run_llada_window_spare_a800.sh Q DEADLINE contested`.

Both pilots stopped as registered: no qualifying exchange in 0, 10 or 20
at either Q8 or Q16. All six old arms matched their sealed controls and
the candidate IDs and bills were identical. No full90 effect estimate.

**Q4 active window, A0.** [Sealed frozen90](c4-window-frozen90.json),
[width vs yield breakdown](c4-width-accounting.json): 72/90 / 1.504 /
37.68%, vs strict capped IBS 69/90 / 1.395 / 35.66%. Original30 25 vs
23, new60 47 vs 46; eight rescues and five losses overall vs external IBS.
**Independent follow-up:** run the same Q16 paid-waiting-MASK rule at
fixed Q4 with no additional projected rows or new sample clock. Require
the complete same-device Q4 active-window parent and exact output hash,
bill, metric and row-count replay for every control. Stop on no measured
exposure in 0/10/20. Otherwise report original30/new60/frozen90; compare
to the Q4 window **and** strict IBS; do not replace cases or change the
public offer-switching policy. Launcher: `run_llada_small_geometry_a800.sh
paid90 DEADLINE 4`.

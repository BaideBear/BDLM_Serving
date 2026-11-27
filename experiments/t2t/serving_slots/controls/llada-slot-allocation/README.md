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
Extend unchanged to added60 on the same physical A0, with exact replays of
both arms on 0/10/20 before accepting the inherited original30.

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

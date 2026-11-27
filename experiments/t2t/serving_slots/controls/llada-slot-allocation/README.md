# Low-offer LLaDA row selection

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

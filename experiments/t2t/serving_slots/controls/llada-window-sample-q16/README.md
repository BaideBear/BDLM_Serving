# LLaDA Q16 active-window paid warm-MASK single factor

Baseline: the measured active-window control at Q16 (pinned geometry
`258c9a8b…`, legacy request source). The old arm is that control; the new arm
adds already-paid waiting warm MASK positions on mixed cold calls via
`llada_window_sample.WindowPaidMask`. Row selection, sampler rule and physical Q
are unchanged; the paid MASK rows are already in the plan's `rows`. The old arm
replays the measured control **90/90** (`control_replay`, 0 failures).

Preregistration (`preregistration.json`, written before scores):
- order: `0, 10, 20, 1..89` (frozen); pilot gate `0/10/20`; full90 only if the
  pilot has real paid exposure.
- arms: `old = active-window control`, `new = window paid warm MASK`.
- geometry/sampling/planner hashes, checkpoint, samples and scorer are recorded.

| Cohort | Coverage | Old correct | New correct | Old F | New F | Old Q | New Q | Old macro TPF | New macro TPF | Old pooled TE | New pooled TE | Exposed | First checks | Control replay | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original30 | 30/30 | 21 | 25 | 6974 | 6836 | 71813 | 67253 | 2.799 | 3.183 | 0.2047 | 0.2300 | 28 | 28 | 30/30 | 0 |
| new60 | 60/60 | 49 | 50 | 14746 | 10747 | 146454 | 108338 | 2.999 | 3.443 | 0.2134 | 0.2721 | 55 | 55 | 60/60 | 0 |
| all90 | 90/90 | 70 | 75 | 21720 | 17583 | 218267 | 175591 | 2.933 | 3.356 | 0.2105 | 0.2560 | 83 | 83 | 90/90 | 0 |

Macro TPF is the per-request mean of `N_i/F_i`; canonical TE is pooled
`ΣN/ΣQ`. Paid waiting: 3142 calls, 19172 rows, 5414
admits (all90). First paid exposure tick: min 2 / median 8 / max 68. The first
paid addition keeps the same pre-state, rows, Q, dirty, prefix, mask, cold,
visible KV, phase, sample clock and per-layer physical Q: **83/83 first checks,
0 first failures**. No old/new generation failures.

Conclusion: **positive**. On the measured active-window Q16 control, paid warm
MASK sampling improves correctness 70→75/90, reduces executor F by 19.0% and
physical Q by 19.5%, and raises canonical pooled TE 0.2105→0.2560. Original30
is 21→25 (strict capped IBS16 is **23/30**, nominal 24/30); new60 is 49→50
(strict capped 49/60, nominal 49/60); all90 is 70→75 (strict capped 72/90,
nominal 73/90). Macro TPF for original30/new60/all90 is 3.183/3.443/3.356,
versus strict capped IBS16 3.189/3.468/3.375 (nominal 3.173/3.470/3.371).
This can be enabled as the Q16 scarce-budget single factor. It is not combined
with the negative KV-horizon factor. A4 is free after the run; A5 Q16 window
is main's pilot and was not touched.

Paths:
- run: `/gs/results/llada-window-sample-q16-pilot-20261128-084305` (90 rows,
  source data stays on the cluster)
- archive: this folder (`summary.json`, `summary.csv`, `preregistration.json`,
  `replay-check.json`)
- code: `experiments/t2t/serving_slots/llada_window_sample.py` (q=16 enabled),
  `llada_frontier_seat.py` (paid q=16 registration/validation),
  `tools/launch_llada_window_sample_q16.py`, `tests/test_llada_window_sample_q16.py`

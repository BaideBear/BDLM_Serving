# Q64 verified EOS: optional prefix dose

This is a fixed-Q64 A800 GPU7 pair with the same verified-EOS control in
both arms. See `registration.md` for the inventory check, hypothesis,
staged protocol and decision rule. No public API or chart changes.

## Run and reproduce

The launcher was `bash /gs/runtime/eos-prefix-dose-q64-1128/launch.sh`.
It refuses an occupied GPU or existing output, and has an absolute
2026-11-28 13:24 +08:00 cutoff. The registered model PID was 158616.
The independent overlay imports the original executor, verifier, physical
auditor, model and scorer from the pinned verified-EOS runtime. There is
no waiting-seat code on this path and no second copy of the base repo.

The only production decision changed in the candidate is the clean
prefix quota after dirty payments: floor(remaining seats / 4) instead of
floor(remaining seats / 8). Actual selection is still limited by available
right-edge prefix rows and the ordinary MASK reservation. An increased dose
can replace a MASK or use empty capacity; dirty or cold priority is unchanged.
Different future writes, accepted tokens, clock counts and encoder calls are
possible because the physically sampled positions can differ.

Do not rerun inference to reproduce the CSVs. On A800:

```sh
rt=/gs/runtime/eos-prefix-dose-q64-1128
base=/gs/runtime/verified-eos-1128/repo
CUDA_VISIBLE_DEVICES= PYTHONPATH="$rt:$base" /usr/bin/python3 -B \
  -m experiments.t2t.serving_slots.gemma_eos_prefix_dose_report \
  --checkpoint 90 --out "$rt/compact"
```

`--checkpoint 3` and `--checkpoint 30` read only the corresponding already
sealed pairs. The 3 anchors belong to original30; original30 and new60
partition frozen90. There are no substitute tasks. Six no-model selection
and accounting tests are in `tests/test_gemma_eos_prefix_dose.py`.

## Staging scope (nothing staged by the worker)

From the worktree root, the new source files are:

- `experiments/t2t/serving_slots/gemma_eos_prefix_dose.py`: factor and physical row classification.
- `experiments/t2t/serving_slots/gemma_eos_prefix_dose_eval.py`: one paired worker delegating model execution.
- `experiments/t2t/serving_slots/gemma_eos_prefix_dose_report.py`: offline replay and compact export.
- `tests/test_gemma_eos_prefix_dose.py`: pure selection and accounting tests.

Under `experiments/t2t/serving_slots/controls/gemma-factors/eos-prefix-dose-q64/`
the files are `registration.md`, `launch.sh`, this `README.md`, `pairs.csv`,
`cohorts.csv`, and `provenance.json`. No change to shared serving or plotting
code is required to reproduce this study.

## Read the receipts

- `pairs.csv` has both scores, non-special token counts, F, Q, ID digests,
  first changed head/tick/local addresses, physical role counts and exposure.
  A blank first tick means both full receipts matched. No token IDs or raw
  generation text leave `/gs/results/eos-prefix-dose-q64-1128`.
- `dirty + mask + prefix + verify_clean` equals decoder Q exactly. `mask`
  excludes dirty rows and includes the `cold_mask` subset. `prefix` means a
  projected clean, accepted right-edge seat in sampling, not a prompt token.
  `verify_dirty` is already in `dirty`; it is shown separately for overlap.
  Nonterminal post-prefill encoder rows are added once for `total_Q`.
- `more_prefix`, `fewer_masks` and `extra_width` compare both plans *in
  the same state* at each tick on that arm's own trajectory. They satisfy
  `more_prefix = fewer_masks + extra_width`. On 1/4 these are realized
  higher-dose actions; on 1/8 they are shadow opportunities. Counts across
  diverged traces must not be subtracted as if they were matched ticks.
- `cohorts.csv` reports canonical TE = sum(N)/sum(selected history-reading
  decoder Q), macro TPF = mean(N/F), and paired descriptive
  95% intervals (4,000 bootstrap draws stratified by dataset). Neither TE
  nor TPF measures wall-clock latency. `max_Q` is peak physical decoder width
  on any of the 85 monitored projections, not the post-block encoder width.
  `N_per_total_Q_diagnostic` includes the post-prefill encoder in its
  denominator and is **not** TE. The prelaunch registration is immutable:
  its phrase "physical TE" and initial checkpoint narration designated this
  diagnostic, not canonical TE. The owner clarified the canonical label
  before the full cohort was reported. This concerns accounting only and
  does not change either arm or any replay gate.
- `provenance.json` identifies the runtime, scorer, sample and reference
  digests, experiment source, checkpoint and aggregate raw-pair digest.

## Sealed A7 results: keep 1/8

The 3, 30 and 90 checkpoints passed; all 90 lower-dose receipts replayed
the historical ordered-EOS on arm exactly. No missing, failed or replaced
pairs. Both arms had max decoder width 64 at all 85 monitored projections.
All 90 historical control role partitions also matched the pre-EOS control
after reconciling the terminal suffix and verify/dirty overlap. The other
arm was measured, not synthesized from that comparison. Worker 158616
exited normally; GPU7 had zero compute PIDs and 0 MiB at 08:22:59 +08:00.

| Cohort | Exposed | Correct (1/8 -> 1/4) | N | Paid F | Selected decoder Q | Canonical TE | Macro TPF |
|---|---:|---|---|---|---|---|---|
| Original30 | 30/30 | 29 -> 29 | 10,348 -> 10,503 | 1,314 -> 1,338 | 75,975 -> 77,932 | 13.620% -> 13.477% | 8.0301 -> 8.0556 |
| New60 | 59/60 | 54 -> 56 | 21,577 -> 22,841 | 2,635 -> 2,890 | 152,672 -> 169,382 | 14.133% -> 13.485% | 8.2491 -> 8.0749 |
| Frozen90 | 89/90 | 83 -> 85 | 31,925 -> 33,344 | 3,949 -> 4,228 | 228,647 -> 247,314 | 13.963% -> 13.482% | 8.1761 -> 8.0684 |

The candidate rescued four (#16, #67, #78, #86) and lost two (#24, #80);
10/90 delivered ID sequences matched. Only #76 never exercised the new
dose and its whole receipt matched. For 88/89 first divergences the
higher dose replaced at least one MASK; one first divergence only filled
slack. The candidate's 1,746 exposed ticks out of 2,674 sparse sampling
ticks allocated 5,733 more prefix seats than the 1/8 plan on those same
states: 3,954 displaced MASKs, 1,779 filled previously unused width.
These same-state counters are not the between-trajectory totals below.

Decoder roles (1/8 -> 1/4): dirty 112,005 -> 120,049; other MASK
76,918 -> 79,560; clean prefix 4,980 -> 11,328; other verification
34,744 -> 36,377. The 264 -> 167 dirty rows paid during verification
are included once in dirty. The longer on trajectories also encoded more
blocks (82 -> 89); total Q including encoder was 249,639 -> 270,098 and
N/total Q was 12.788% -> 12.345% (diagnostic only, not TE).

Net correctness delta was +2/90, with paired descriptive 95% interval
[-3.33, +7.78] percentage points. Macro TPF delta was -0.1077 with
interval [-0.3383, +0.1024]. Neither is evidence of a reliable quality
or throughput improvement. On the measured set, 1/4 cost 279 more paid
forwards and 18,667 more selected decoder rows; it fails the registered
efficiency rule under both the canonical and diagnostic ratios.

**Non-negotiable:** pay every changed row, sample only fresh projected
rows on the original clock, verify the contiguous terminal prefix including
the full EOS-containing chunk, and encode every continuing block.
**Discretionary:** with debt and cold initialization reserved, the same
Q64 offer can spend its remaining seats revisiting clean prefix rows,
advancing unresolved MASKs, or leaving spare width unused. More prefix
seats are not free: keep 1/8 as the fixed efficiency choice on this route;
1/4 is an observed quality/cost trade-off, not a public profile upgrade.

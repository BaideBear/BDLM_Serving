# A1: rotating the first cold tile to last

The intervention changes only first-visit order: [Q,256), then [0,Q).
Both arms use the guarded verified-EOS profile at fixed Q32 or Q64,
with a single request and a single head. It is not prior half-pacing,
paid-row sampling, waiting-seat, a prefix dose, or a successor experiment.
See `registration.md` for the historical route counts and prelaunch rule.

Actual bounded invocation on A800:

```sh
bash /gs/runtime/cold-frontier-last-a1-1128/launch.sh
```

The model process identity (PID 185042, boot ID, start ticks), exact source,
input and reference digests were registered before model inference. The
second arm reused the same resident model; different requests have separate
seeded sampler state. The loader refuses a nonempty physical A1 or an existing
result directory. Off must reproduce the full historical A7 receipt, not
just its score. Both study arms are actually measured on A1.

The pinned runner checks decoder positions, all 85 projector widths, the
ordered terminal argmax tape, every dirty write and refresh, one clock per
sample and the encoder's actual inputs. The separate audit reconstructs the
entire first-visit permutation for every head. It checks coverage without
assuming the candidate has the control's output, head count, warm steps
or score. Cold cost includes the sampling call that finishes the canvas;
its subsequent payment falls in the next phase. Warm excludes closing
drain and ordered verification. Encoder calls after prefill count in F.

Canonical TE is pooled non-special N / selected history-reading decoder
Q, *never* decoder-plus-encoder Q. Macro TPF averages per-question N/F.
The paired descriptive 95% intervals use 4,000 within-dataset bootstrap
draws. `warm_entry_frontier_masks` counts unresolved positions among the
leftmost Q on first entering warm/close; the per-head rate is also reported.
First-head-only counters do not depend on how many later heads were generated.

Offline compact export on A800 (no model or GPU):

```sh
rt=/gs/runtime/cold-frontier-last-a1-1128
CUDA_VISIBLE_DEVICES= PYTHONPATH="$rt:/gs/runtime/verified-eos-1128/repo" \
  /usr/bin/python3 -B -m experiments.t2t.serving_slots.gemma_cold_frontier_last_report \
  --checkpoint 90 --out "$rt/compact"
```

Use `--checkpoint 3` or `30` for the corresponding sealed subsets. The
only local compact outputs are `pairs.csv`, `cohorts.csv` and
`provenance.json`. Raw generations, IDs and tick traces stay under
`/gs/results/cold-frontier-last-a1-1128` on A800. Independent sources:
`gemma_cold_frontier_last.py`, `gemma_cold_frontier_last_eval.py`,
`gemma_cold_frontier_last_report.py` and `tests/test_gemma_cold_frontier_last.py`.

## Result: keep the leftmost tile first

Both A1 tiers sealed all 90 pairs (360 actual requests total). No missing
pairs, replay failures, unexposed pairs or physical overages. All 180 off
receipts matched the historical verified-EOS-on receipts field for field.
Peak physical decoder widths were 32 and 64 in their respective tiers.
The preregistered quality and efficiency criteria failed at both tiers.
This is not a public profile and does not qualify dynamic switching.

| Q | Cases | Correct off/on | Macro TPF off/on | Canonical TE off/on | Total F off/on | Rescues/losses |
|---:|:---|:---|:---|:---|:---|:---|
| 32 | Original 30 | 29/25 | 4.832/4.098 | 15.34%/13.43% | 2247/2572 | 1/5 |
| 32 | New 60 | 56/51 | 4.902/4.282 | 15.77%/14.10% | 4627/5763 | 1/6 |
| 32 | All 90 | 85/76 | 4.879/4.220 | 15.63%/13.89% | 6874/8335 | 2/11 |
| 64 | Original 30 | 29/25 | 8.030/7.262 | 13.62%/12.20% | 1314/1539 | 0/4 |
| 64 | New 60 | 54/51 | 8.249/7.257 | 14.13%/12.36% | 2635/3199 | 2/5 |
| 64 | All 90 | 83/76 | 8.176/7.258 | 13.96%/12.31% | 3949/4738 | 2/9 |

Frozen90 paired macro TPF differences (on minus off; descriptive 95%
intervals): Q32 -0.658 [-0.826, -0.497]; Q64 -0.918 [-1.204, -0.646].
Only 1/90 and 2/90 output-ID sequences remained identical. Reordering
first visits changes which keys are available and which rows receive the
seeded random draws; the experiment does not separate those two mechanisms.

| Frozen90 bill (off/on) | Q32 | Q64 |
|:---|:---|:---|
| Non-special N | 32774/35272 | 31925/33607 |
| Selected history-reading decoder Q | 209689/253892 | 228647/273036 |
| Decoder F | 6794/8244 | 3867/4649 |
| Post-prefill encoder F | 80/91 | 82/89 |
| Number of heads | 170/181 | 172/179 |
| Cold F per head | 15/15 | 7/7 |
| Warm F per head | 17.65/23.20 | 11.44/14.83 |
| First head only: total warm F across 90 | 1783/2281 | 1197/1474 |

In both arms at both tiers, every cold first visit made a real write and
there were zero idle cold seats. With B=256, B first visits, B resulting
refresh obligations and at most Q of them carried across the phase boundary,
the measured trajectories require at least ceil((2B-Q)/Q) cold forwards per
head. Both orders already attain it. The extra cost is largely warm:
+1198 of +1461 total F at Q32; +688 of +789 at Q64. The first-head-only
counters show this is not merely the consequence of generating more heads.

On the unchanged off trajectories, even granting *all* separate closing
drains for free (165 Q32; 149 Q64), the optimistic macro TPF ceilings are
only 5.008 and 8.522. The historical capped IBS figures are 6.840 and
11.050 on these same cases (`controls/fixed-request-comparison/metrics.csv`);
they are descriptive external baselines, not paired A1 arms. This bound
applies to that single class of calls, not to all possible scheduling.
Mandatory written rows, ordered verification through measured EOS, and
nonterminal encodes still have to be paid. Keep the reference order; do
not promote or extend this intervention.

[Cost of catching strict IBS by warm calls alone](warm-gap.csv).
`python -m experiments.t2t.serving_slots.gemma_warm_gap` joins both complete
work tapes to the verified deliveries by case, seed, output digest and bill.
It solves `mean_i(N_i / (F_i - r * warm_F_i)) = TPF_capped_IBS` for one
fractional reduction `r` applied to every request. Keep all other phases
and output lengths fixed. At Q32, 3,001 of 6,874 calls were warm and `r`
is 70.21%; at Q64, 1,967 of 3,949 calls were warm and `r` is 54.42%.
These are conditional break-even requirements, not achieved speedups.

Eight synthetic no-model tests pass. The worker exited normally and physical
A1 was checked empty (0 MiB, no compute PID) at 09:37 +08 on 2026-11-28;
there is no successor. Only compact CSVs and provenance were copied back,
with all three hashes verified; raw receipts remain on A800.

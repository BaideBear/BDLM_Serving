# Q32: measure the first physical tile but defer its first sample

The baseline is the historical **on** arm of `verified-eos-1128` at Q32.
Both arms here are real single-request A1 runs. They have the same
right-edge 1/8 prefix policy, no tail, no reuse, seed, input, sampler,
ordered verification and nonterminal encoder. Only the candidate disables
`consume` on rows `[0,32)` at block opening. This is neither half-pacing
across cold batches nor retention of rejected noise after sampling.

The first call keeps its physical width and really writes the initial
canvas KV at all 30 decoder layers. **Initialized** then means KV resident,
not accepted and not previously sampled. Those 32 rows have no SC, no
acceptance and no dirty write. RNG and sampler age stay at their initial
values. Warm selection eventually reprojects those positions, takes
their **current** logits and performs their first ordinary sample. The
worker rejects a missing or premature revisit. All other cold tiles use
the usual algorithm; assigning the next draw and temperature age to a
different physical tile is a consequence of skipping the first sampling
action. This experiment does not separate those trajectory effects.

The pinned verifier measures 85 projection widths, actual per-step
positions and argmaxes, dirty transitions, encoder history inputs and the
physical F/Q bill. Every control must equal the complete historical row,
including output IDs and all tick digests. The first physical logits of
the first block of each pair match byte-for-byte by SHA-256; for later
corresponding blocks only the initial random canvas, RNG and width are
matched, not the possibly diverged encoder history. No draft constitutes
an EOS or a verification result.

On A800 the guarded invocation is `bash /gs/runtime/gemma-first-tile-kv-q32-a1-1128/launch.sh`.
Raw receipts remain in `/gs/results/gemma-first-tile-kv-q32-a1-1128`.
The registration records the source and reference hashes, PID, boot ID,
start ticks, physical device, environment and deadline. To rebuild the
compact from immutable receipts without loading a model or using CUDA:

```sh
bash /gs/runtime/gemma-first-tile-kv-q32-a1-1128/report.sh 90
```

`3` and `30` are also valid sealed checkpoints. `pairs.csv` contains a
row per matched request; `cohorts.csv` separates anchors, original30,
new60 and frozen90. `provenance.json` binds these to the registration,
the complete historical control set and the report implementation.
The experiment files are `gemma_first_tile_kv{,_eval,_report}.py` and
`tests/test_gemma_first_tile_kv.py`; no shared executor or profile changed.

`F` counts decoder plus **post-prefill** encoder calls. Macro TPF is the
mean of per-case non-special N/F; canonical TE is pooled N divided by
selected history-reading decoder Q only. Encoder Q is billed separately.
Cold includes every call through the completion of physical KV coverage,
warm is the remaining sample-eligible work, close is drain-only after
sampling ends, and verify is the ordered physical sweep including any
fused first payment. These phase counts do not double count rows or F.
Both the by-head rates and first-head-only counts are provided because
the factor can change output length and the number of blocks opened.
Paired differences use 4,000 within-dataset bootstrap resamples with the
fixed seed 1128. The three anchors alone have no interval estimate.

## Sealed result: keep ordinary first-tile sampling

| Cohort | Correct off/on | Macro TPF off/on | Canonical TE off/on | Total F off/on | Rescues/losses |
|:---|---:|---:|---:|---:|---:|
| Original 30 | 29/29 | 4.832/4.444 | 15.34%/14.32% | 2247/2317 | 1/1 |
| New 60 | 56/53 | 4.902/4.648 | 15.77%/15.03% | 4627/4583 | 1/4 |
| Frozen 90 | 85/82 | 4.879/4.580 | 15.63%/14.79% | 6874/6900 | 2/5 |

All 90 control rows replayed exactly; 90/90 pairs matched on the first
measured logits and exercised the action. No missing/failed pairs or
over-budget calls. At #0/#10/#20, control F was 79/76/35; candidate F was
84/80/42. Both arms scored 3/3. The frozen90 macro TPF difference is
-0.299 (paired descriptive 95% interval [-0.418, -0.181]); only 25/90
pairs gained TPF. The accuracy difference is -3.33 percentage points
[-8.89, +2.22]; this does not establish a quality difference or equivalence.
Only 3/90 token-ID sequences were identical.

| Frozen90 measured ledger | Off | On |
|:---|---:|---:|
| Non-special N | 32774 | 31312 |
| Decoder F / selected decoder Q | 6794 / 209689 | 6823 / 211700 |
| Post-prefill encoder F / encoder Q | 80 / 20480 | 77 / 19712 |
| Blocks; peak decoder width | 170; 32 | 167; 32 |
| Cold / warm / closing drain / verify F | 2550 / 3001 / 165 / 1078 | 2338 / 3298 / 154 / 1033 |
| Cold / warm F per block | 15 / 17.65 | 14 / 19.75 |
| First block only: cold / warm F across 90 | 1350 / 1783 | 1260 / 2041 |
| First tile immediate writes / next-call dirty rows | 5440 / 5440 | 0 / 0 |
| First-tile admits at first actual sample | 2228/5440 (40.96%) | 2736/5344 (51.20%) |
| All changed rows / all paid dirty rows | 97711 / 97711 | 96798 / 96798 |

The local hypothesis was partly right: all 5,344 deferred rows were
actually reprojected and sampled once for the first time with full KV
coverage, at sampler age 7, and first-sample admissions improved. The
end-to-end hypothesis failed: each block saved one cold call but used
2.10 more warm calls on average. Even restricting to the first block of
each matched request, cold saved 90 F and warm added 258 F. Fewer
encoder and verify calls reflect different deliveries, not a relaxation
of either requirement. The experiment does not identify which downstream
sampling decision caused the extra warm calls.

On the fixed historical outputs, granting one free F per opened head
with no other cost would yield macro TPF **5.013**. This is a conditional
bound on that saving alone, not a bound on all possible trajectory changes.
For scale, historical fixed-width capped IBS scores 79/90 at TPF 6.840;
it is not a third matched arm. For Q32, keep consuming the first measured
logits; do not promote, extend or schedule a successor.

The worker PID 291644 exited normally. `complete.json` was written at
11:42:41 +08 on 2026-11-28; physical A1 had no compute PID and 0 MiB at
11:42:56 +08. Five no-model tests passed locally and on A800. The three
compact output hashes matched at both ends. No per-tick logs, output IDs
or raw pair files were copied from A800.

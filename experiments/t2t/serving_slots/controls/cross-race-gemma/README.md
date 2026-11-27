# DiffusionGemma fixed cross race

`original30` = indices 0–29 from **this A800 race**, `new60` = 30–89,
`frozen90` = 0–89. These are disjoint slices and their union, not repeated
runs of the same tasks. Historical controls cover the same frozen cohorts;
they are not paired executor interventions.

## One-command recompute

```bash
bash experiments/t2t/serving_slots/controls/cross-race-gemma/gather.sh --require-frozen90
```

Run from the repository root with a local Python 3.10+ (`PYTHON` overrides
the executable). `A800_HOST` defaults to the `A800-80g-2` SSH config alias;
`A800_PORT` and `A800_MUX` optionally select a port and existing SSH socket.
Remove `--require-frozen90` for a coverage check that exits 0 even when the
race is incomplete. With the flag, an incomplete run still refreshes the
audit and eligible outputs, then exits 1. There is no polling loop. The
command streams the script to A800, reads `/gs/results/cross-race-1128` there,
and returns only cohort aggregates, coverage indices and provenance. It does
not copy receipts, prompts, generated text/token IDs, or error text, invoke
a scorer, reserve GPUs, or change a worker. It needs the existing read-only
runtime checkout and the local historical control tables.
The gather-side reporter is pinned to the archived remote runtime digest;
render checks the manifest and fixed-cross arm registry without requiring
the local reporter to have the same unrelated experiment registrations.

## Files and denominators

| File | Content |
|---|---|
| `coverage.csv` | All 18 arms × 3 cohorts, including those not yet eligible. Reports receipt and score counts, exact missing/pending indices, and missing indices observed in worker `START`/`DONE` logs. |
| `arms.csv` | Complete cohorts only. 18 fixed race arms and 8 external references per cohort. |
| `paired.csv` | Complete cohorts only. 15 registered within-race contrasts per cohort, trial minus reference. |
| `provenance.json` | Observation time, paths, manifest/samples/source hashes, runtime, scorer hash availability, log traceback count, bootstrap settings. |
| `original30.svg`, `frozen90.svg` | Compact original30 figure; the large figure spans all three cohorts and is emitted only when all 18 arms have 90/90 scored receipts. |

- Eligibility requires scored receipts for **every** expected index in **every** registered arm. The shared reporter checks manifest identity (index/dataset/task/seed), per-task input digests, duplicates, and work accounting. Missing/score-pending cases never become false scores and no partial estimate is archived. A `START` with no receipt may be in flight, interrupted, or failed; there is no per-task worker failure receipt to disambiguate it. `failed=0` on a race results row means no recorded generation failure among the complete receipt set, not no retries or no wrong answers. Scored wrong/truncated requests remain in the cohort; truncation has its own column. Receipt scorer hashes are absent in the observed run, so scorer equivalence cannot be proven by receipt.
- `macro_TPF` is mean per-request `N/F`, not pooled `sum(N)/sum(F)` or wall-clock throughput. Physical Q counts generated decoder **and** encoder rows, excluding prompt prefill. In this race, decoder Q is also the measured prefix-KV row denominator. The external Native and IBS prefix-KV denominators present in the matrix are kept separately from physical Q. Flow W2/W3/W4 and A800 IBS C512 lack a verified matching prefix-KV denominator in that matrix; the fields stay blank. Flow W3/W4 truncation counts are not provided by the source CSV used here. A blank tail rate or rate delta means undefined (e.g. no cut-off gate candidates), not 0. Tail final-match rate is `tail_final_matches/tail_last_candidate`, not causal token utility.
- Native C256 and Flow W1 are shown once as the archived W1/native reference. Historical IBS C512 is nominal and saturates at a single block; H800 and A800 points stay separate. Historical executors, hardware/software, and sampling/stop paths need not match; do not infer same-device speed or causal effects from these overlays. Flow W3/W4 also use different environments across cohorts.
- Within-race pairs change one registered factor: C512 top gate vs `off`; C64/C128 prefix dose or late borrow vs single-block, unborrowed `1/8`. The race is not factorial; it does not estimate interactions. Intervals are 95% paired request percentile bootstrap (4,000 draws, seed 0), conditional on the observed cohort, not adjusted across contrasts. Sparse discordance can yield degenerate intervals; a tie or an interval touching 0 does not establish equivalence.

## Original 30 readout

- **Top gate (C512):** `off` 28/30, macro TPF 19.657; `eager` 29/30, 21.509, +67,163 decoder rows (+50.6%), opens/discards 38/14. Paired TPF +1.852 [0.608, 3.057]; +1 rescue, 0 loss. `half` adds +1.092 TPF at unchanged 28/30; `quarter` is 29/30, +0.131 TPF, with the TPF interval spanning 0. All four trail Native (24.284 TPF), IBS C512 (23.045 H800 / 21.869 A800) and Flow W2 (23.893).
- **Prefix (C64/C128):** Removing clean-prefix revisits loses 2 scores at each tier and reduces macro TPF by 1.117 / 1.720 versus `1/8`. `1/4` vs `1/8` changes macro TPF by +0.054 / +0.010, with both intervals spanning 0. At C64 the equal correct totals conceal one rescue and one loss. Neither tier's race arm reaches the matched-tier historical IBS macro TPF.
- **Borrow (C64/C128):** All six borrow arms spend tail Q. C64 has no score changes and TPF drops 0.110–0.179. C128 borrow at `1/4` keeps 30/30 but has an uncertain TPF change (+0.096, 95% interval −0.211 to +0.413); `1/2` and `1/8` each lose one score. These are screening observations, not a winner selection.

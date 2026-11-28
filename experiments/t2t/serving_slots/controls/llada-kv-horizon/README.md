# LLaDA Q8 KV horizon: visibility single factor

Baseline: head-only Q8, no borrow, law 1/8, sparse coverage. The old arm is the
main executor with the same selected rows and sampler rule. The new arm is
`KVHorizonRequest`: each forward exposes KV only through the rightmost selected
row (Optimus horizon), while the resident cache keeps its high-water mark for
future reuse/rebuild; the physical causal mask and global dirty accounting are
unchanged. No selection geometry was changed.

Inputs:
- run: `/gs/results/llada-kv-horizon-original30-20261128-072701` (30/30, 0 failures)
- old-arm exact replay: 30/30 output IDs, trace, F, Q and correctness match
  `c8-noborrow` control (`replay-check.json`)
- excluded invalid runs: `...072114` (`--borrow` CLI error, 30 failed rows),
  `...072352` (old ledger, 2 rows)

| Cohort | Coverage | Old correct | New correct | Old F | New F | Old Q | New Q | Old macro TPF | New macro TPF | Old pooled TE | New pooled TE | Strict cap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| original30 | 30/30 | 11 | 7 | 16220 | 19696 | 113469 | 137794 | 1.635 | 1.645 | 0.1628 | 0.1594 | yes |

Canonical TE is pooled `ΣN/ΣQ`; macro TE is not used for this conclusion.
This note records the horizon run at the time it was measured; it is not a
current GPU availability statement.

First-divergence ledger (30/30):
- first horizon exposure tick: min 2 / median 9 / max 99; gap at first exposure
  min 1 / median 4 / max 29.
- first output difference tick: min 4 / median 10 / max 101; gap at first output
  min 0 / median 3 / max 22.
- first plan difference tick: min 6 / median 12 / max 103.
- 30/30 `visibility_only`, 30/30 `same_selection_before`, 0/30 `zero_gap_divergence`.

Conclusion: **negative**. Horizon exposure precedes output divergence (median
tick 9 vs 10), selection geometry is identical until the first output difference,
but quality drops 11→7 while executor F and physical Q both rise 21.4%. Do not
enable or expand this factor, and do not pair it with window geometry Q8 (the
"independent improvement" precondition is not met).

Recommendation:
- Abundant Q (Q64): use the measured quarter gate + warm-head arm (all90 70→72,
  F -7.7%, Q -8.4%); avoid horizon.
- Scarce Q (Q8): keep the full resident right boundary; prioritize global dirty
  refresh and cold MASK sampling. Window geometry Q8 is the promising separate
  lever (24/30, TPF 2.210 vs control 11/30, 1.635), but do not stack it with the
  negative horizon factor; if a next single factor is wanted, run window geometry
  alone on a free device.

The horizon run used A4 at the time; this is a historical record, not a current
GPU status.

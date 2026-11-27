# Gemma: fixed-budget prefix selection and terminal verification

Two separate single-factor studies. Do not pair receipts across the two
source snapshots. `coverage.csv` reports each registered contrast in original
30, added 60 and all 90, including missing coverage. `paired.csv` publishes
absolute metrics on the common, fully scored cohort only. The reference is
repeated per contrast; never sum across contrasts or overlapping cohorts.

| Directory | Matched arms | Diagnostic other than quality |
|---|---|---|
| `rank/` | Within each Q32/64/128/256, keep prefix dose at 1/8. Reference is right edge. Separately change to oldest, seeded uniform, or reuse currently computed unresolved logits when the whole initialized head's debt is paid. | Actual seats differing from same-state right edge, prefix edits and revocations, paid sampling F, and changes to the complete output ID sequence. Different IDs need not mean worse quality. |
| `close/` | Within each Q32/64/128/256/512, compare unmodified head-only stopping with exact-order drain/first-verify fusion. Both sides use the same head-only route; Q512 is an offer, **not** a dual-canvas test. | Check output ID sequences, encoder work, and whether all F and Q reductions exactly match the fused work. Any parity violations remain visible. |

`token_efficiency` is `sum(N)/sum(history_Q)`, not a percentage column;
`macro_TPF` includes post-prefill encoder forwards. Rank and close use real
decoder projections, so here `history_Q == decoder_Q`. Physical Q includes
encoder Q. Rescue/loss, index lists and confidence intervals belong to the
contrast, not an individual side; deltas are trial minus reference. `[]`
means checked with no mismatches; a blank diagnostic on a no-common row is
untested. Scorer and recorded source digests are checked for every receipt
within its study. These are not a hash of the entire runtime.

From the repository root run `bash experiments/t2t/serving_slots/controls/gather_gemma_factors.sh rank`
and the same command with `close`. Optional `A800_HOST`, `A800_PORT`,
`A800_MUX` and `PYTHON` select the SSH host and local Python 3.10+. Optional
second argument selects a different output directory. Only aggregate reports
leave the GPU host. No worker is restarted and no generation runs locally.
The [90-case external controls](../cross-race-gemma/README.md) are not online
interventions on the same sampler.

## First complete slice: original 30, terminal fusion

On the original 30 tasks, all five paid-close comparisons preserved every
final output ID and encoder workload. Observed differences in F and decoder Q
match the fusion counters for each request. Negative and positive opportunities
are both included:

| Fixed decoder offer | Correct, off = on | Macro TPF, off → on | Saved F, total | Saved decoder Q, total |
|---:|---:|---:|---:|---:|
| 32 | 29/30 | 4.607 → 4.607 | 0 | 0 |
| 64 | 29/30 | 7.713 → 7.742 | 6 | 51 |
| 128 | 30/30 | 12.024 → 12.209 | 14 | 96 |
| 256 | 29/30 | 17.177 → 19.072 | 54 | 884 |
| 512 (head only) | 28/30 | 19.657 → 22.184 | 54 | 740 |

The intervention removes a duplicate projection, not final verification or
the encoder. It is easier for the original first verification slice to contain
all remaining dirty rows at a large offer. The table is a statement about
these 30 inputs only; a real two-canvas C512 finish has not been tested here.

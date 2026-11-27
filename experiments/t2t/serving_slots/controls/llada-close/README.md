# LLaDA 2.1 fixed Q32: what is a terminal write worth?

This is a head-only experiment, not native inference and not a proposed mixed
schedule. Ordinary decoding, contiguous partial-block handling, the 1/8
clean-prefix rule, greedy sampler, 0.9 admission and 0.5 edit thresholds
are identical. The only intervention is on the first **mask-free** writeback
of each physical head, before retirement. All decoder calls are capped at 32.

| Arm | Rows when this opportunity arises | What the logits do |
|---|---|---|
| `unchanged` | Mandatory dirty inputs only | No sampling |
| `full_write` | All valid generated rows of the current physical block | Refresh all those KV; no sampling |
| `edit_once` | The same full head | Offer one post-MASK T2T pass, bill any resulting dirty inputs |

Clean rows in `full_write` are extra physical decoder work even if no token
changes. They can change later blocks through KV. Sampling in `edit_once`
can edit a token or revoke it to MASK. After that single pass, ordinary
decoding and debt settlement resume; neither revocation nor a down-the-line
write grants another terminal edit pass for that head. Complete residency,
all required writes and EOS checks still precede final retirement.
This is deliberately not the checkpoint's repeated post-MASK edit loop.

The three arms run on one A800 in rotated order per task. The original strict
Q32 1/8 arm must replay the archived frozen indices 0, 10, 20 (IDs, prompt,
score, N, F, Q). Each paired trajectory is checked up to the first change,
including the selected and sampled positions, written positions and debt.
The full-head arms must buy exactly the same first physical positions.
Twenty layer widths, first-layer block mask, output-head width and the origin
of every token write are validated by the common fixed evaluator.

`tools/run_llada_close_factor_a800.sh` runs the experiment. From the same
runtime checkout, invoke `python -m experiments.t2t.serving_slots.llada_close_factor
report /gs/results/llada-close-1128 benchmarks/frozen90/manifest.json`.
Only cohorts containing all registered pairs produce scores.
TPF is mean per-request final non-special N / generation F; TE is sum N /
sum selected decoder Q reading prefix KV (no encoder). The three-case
qualification is not the 30- or 90-case result.

The [paired receipt](paired.json) includes the 0/10/20 qualification:

| Arm | Correct on 3 | Macro TPF | TE | F | Q | N |
|---|---:|---:|---:|---:|---:|---:|
| unchanged | 3 | 4.279 | 18.44% | 237 | 4404 | 812 |
| full_write | 3 | 5.541 | 21.38% | 175 | 4060 | 868 |
| edit_once | 3 | 5.237 | 18.16% | 215 | 4896 | 889 |

Full write changed the complete IDs on all three cases; the third arm made
five terminal edits. This is evidence the intervention reaches the model,
not a speed or quality recommendation. For full-90 comparisons use the
[registered references](../block-history90-matrix.csv): native B32 75/90,
5.677, 15.56%; IBS32 71/90, 4.495, 15.23%. Do not compare these scores
to the qualification. The archived LLaDA native and Flow W2 runner uses
0.5 M2T and 0.5 T2T; IBS and our strict control use 0.9 M2T and 0.5 T2T.

The [budget decision rules](../../../../../docs/request-budget-decisions.md)
register the independent cutoff comparison; it uses **unchanged** retirement.

## Original 30 complete

| Method at Q32 | Correct / 30 | Macro TPF | Pooled TE | F | Q | N |
|---|---:|---:|---:|---:|---:|---:|
| unchanged: dirty-only | 23 | 3.945 | 14.94% | 5174 | 99295 | 14832 |
| full_write, no sampling | 26 | 4.741 | 15.34% | 3994 | 86453 | 13259 |
| edit_once | 26 | 4.811 | 15.31% | 4170 | 90225 | 13817 |
| Historical native B32 | 26 | 5.728 | 15.55% | 2575 | 82400 | 12814 |
| Historical Optimus IBS32 | 25 | 4.271 | 13.34% | 5099 | 106270 | 14177 |

No failures among the 90 executions. Dirty-only to full_write changed 29/30
returned ID sequences, with 4 rescues and 1 loss. Increased coverage has a
downstream effect even though the additional rows never sample at closure.
Full_write to edit_once changed 14 sequences, rescued one and lost one; the
61 terminal edits were all token-to-token, with zero revocations. Do not label
the difference a mechanical reduction in F; output N also changes. None of
these 30-case candidate scores may be labeled as new60 or frozen90.

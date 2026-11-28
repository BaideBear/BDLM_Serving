# First-forward attribution: six preregistered H4 states

One real prefill per case, then four independent branches cloned from that
same native pre-sampling state; no rollout. All 20 QKV hooks recorded the
same widths: prefill separately, then 64/64/64/32. Six prefills cost 448 Q
in total; 24 decoder calls cost 1,344 Q. Inputs were BF16, returned logits
FP32. The unchanged dense and compact first-step decisions both reproduced
their respective sealed trajectories.

| Change from dense64 | Head rows with any bit change / 192 | Top-1 changes | M2T decision changes | T2T decision changes | Max absolute logit delta |
|---|---:|---:|---:|---:|---:|
| Identical rerun, width 64 | 0 | 0 | 0 | 0 | 0 |
| Zero unopened embedded rows, width 64 | 0 | 0 | 0 | 0 | 0 |
| Project 32, keep key-axis length and head visibility | 192 | 6 | 3 | 0 | 5.0625 |

In both same-width contrasts all six active KV writes were bit-identical
to dense; all six differed at compact width. Index 7 had four top-1 changes
and three M2T choices changed (absolute positions 74, 76, 79); index 55 had
two top-1 changes but no change in writes. The six states do not test later
post-edit opportunities. This supports a shape-associated difference in
these forwards, not a BF16, attention-kernel or MoE attribution, universal
row isolation, or a causal explanation of the 90-case score difference.

`row-deltas.json` contains all 192 active-row comparisons per contrast,
without prompts or generated text. `compact.ndjson` contains per-case totals.
`verify_receipts.py` verifies raw row-receipt digests and all physical widths
on n164. `handoff.ndjson` retains the two validated holder records.

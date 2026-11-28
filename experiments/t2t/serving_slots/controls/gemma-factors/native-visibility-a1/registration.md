# Native visibility: fixed Q256, A800 physical device 1 only

Historical W1 (validated against native IDs and events) uses Torch 2.9.0+cu128;
the strict run uses 2.6.0+cu124. Separate the archived comparison from a
same-card same-stack generate/off/on comparison. Same weights, prompt, seeds,
temperature, entropy acceptance, SC and post-prefill billing. Do not treat
all accepted as native stop, or substitute a draft for strict verification.

Only vary the representation of an unrestricted, dynamic, one-block
bidirectional decoder SDPA mask, after all WaveKV writes: explicit true
versus None. Do not change K/V layout, row selection, clock, RNG, ordered
verification or encoding. Record first-forward Q/K/V, attention output and
raw logits hashes and sampler diagnostics on cases 0,10,20. Both strict
arms pay every actual write and verify their delivered block. Fixed off must
match the pinned Q256 history. Native is a diagnostic reference; its
different close and next-block RNG policies are not changed by this test.

Expand to remaining frozen90 only if on matches same-stack native first raw
logits on all three and off differs on at least one. No score-based extension.
If not, seal at three; no alternate parameters or second GPU experiment.
No claim for other offers or switching until qualified separately. Deadline
1795843443; no successor or holder.

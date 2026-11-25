# DiffusionGemma: repeated KV-only use of newly added positions

Compare `hard_trigger_refresh` against `hard_trigger` and `hard_noedit`. No
new predictor or tuned thresholds: Q16/32/64/128 use the original trigger's
maximum reserved old positions 2/4/8/16. Q256 is a full B256 canvas; no
additional position can exist. At the same input state, compute exactly the
original trigger's physical rows. Determine which selected old rows are
outside the same-state no-reserve plan; only these are KV-only. Persist their
new KV, but do not update their cached raw logits, sampler result, draft,
self-conditioning or acceptance. All other physical rows use the original
sampling logic. Every row written in the preceding step remains mandatory.
The neutral comparison is a CPU calculation of indices, not a second
forward. The first and final/verification steps and re-encoding are unchanged.

Frozen 30 and seeds, full native qualification on each shard; count physical
rows including generated-canvas re-encoding, never prompt prefill. The
primary comparison for the new *action* (as opposed to placement) is the
original trigger at the same Q; also compare to the no-reserve policy for
the net cost/benefit. Record every newly added KV-only position, all reserved
old positions, displaced MASK, and all-stage Q/F. Unlike the single-anchor
factorial, this is a deployable rule repeated at every eligible step; the
arms are not forced to remain on the same trajectory.

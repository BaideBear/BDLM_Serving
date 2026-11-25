# Gemma Q32: candidate KV persistence versus sampling

## Locked contrast

All four arms use the old `hard_once_rank` selector, the same first eligible
swap, the same physical forward, and then the shared no-reserve policy.
Only two switches at that anchor differ. In `hard_once_xy`, `x` is `k` to
keep the newly written candidate KV at every layer beyond this step, `d` to
restore its exact pre-step values. `y` is `k` to expose newly computed raw
logits and allow candidate actions, `d` to expose its previous raw logits and
freeze candidate acceptance, canvas, draft, and self-conditioning. Neither
switch restores the displaced victim. No extra forward, row, or RNG draw.

The candidate's new KV is visible to other selected rows *within the anchor
forward* in all four arms. This identifies only the effect of **persistence
after that forward** at each fixed score setting. The score switch includes
possible global entropy and stopping effects; it is not simply an edit veto.
This is a causal diagnostic on one anchor, not a repeated deployment policy.

Use all frozen30; no eligible anchor is a valid no-op. Independently qualify
full-width IDs/work against native before each hardware shard. Include all
bootstrap, verification, and re-encoding. Require KK to reproduce historical
rank replay. Compare within each score setting on common physical Q,
incoming full-state hash, new physical candidate row, unchanged other KV,
anchor action rows, post-anchor non-KV and RNG. Require the persisted row hash
to match the measured new/old row per arm. Report contrasts as paired wins
and losses, macro TPF, pooled token efficiency, total F/Q and final sequences.
Report both eligible and whole-cohort denominators explicitly. Do not select
samples or choose the primary switch after looking at results.

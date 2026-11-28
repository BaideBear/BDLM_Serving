# Common LLaDA admission pilot, all five fixed offers

At Q4, 8, 16, 32 and 64 replace the live handle's extra MASK acceptance rule
with the same rule. Keep its row planner, compulsory writeback, ordinary
confidence threshold, per-block fallback, edits, successor mask and retirement
unchanged. This is a common admission rule, **not yet a common row planner**.

If the head already has some local KV, start with its leftmost unresolved
row. Walk unresolved positions from left to right. Stop at a missing measured
proposal or a MASK proposal; bridge already resolved positions. Normal
admissions do not spend additional risk. For additional admissions require
the running sum of `1 - measured confidence` not to exceed **0.35** for this
forward. This is a heuristic, not a calibrated probability guarantee. No
logits from unprojected positions, no extra projections at the decision.

Same-board pair at each fixed offer, seed and prompt, alternating arm order.
Nine predeclared indices 0/10/20/30/40/50/60/70/80 at *every* offer. Do not
select a new rule for an individual offer after observing the results. Show
the first differing actual write, correct, macro TPF, pooled TE, failures,
original/new cohorts. Pilot is for falsification, never a new live profile.
Only consider expanding this exact all-tier rule if it changes writes, no
tier loses two or more correct on the nine, and mean per-request TPF improves
at least three tiers. Even passing this gate is not a comparison on frozen90.

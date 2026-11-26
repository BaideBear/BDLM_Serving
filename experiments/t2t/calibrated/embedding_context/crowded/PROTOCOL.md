# Q8: one predicted edit in place of a MASK

On each original frozen30 LLaDA 2.1 request, follow the complete hard-noedit
path until the first real `swap_mask` opportunity under the already frozen
embedding gate. Ignore all earlier `swap_old` opportunities. Protect every
fresh row (including strict edits and overflow) and the first selected MASK.
A missing opportunity is a complete no-op, never grounds to replace a case.
At that anchor, compare (1) no change, (2) exchange the last selected MASK for
the highest-scoring old row, and (3) exchange the same MASK for a uniform
row outside the base, using `context-swap:{seed}:{block}:{clock}`. No later
interventions on this request. This is not the every-step gate's trajectory.

Offline fit and 5% score threshold are unchanged. Raw optional observations
after the forward cannot inform placement. Qualify both shards on all 30 full
width controls and independently verify the base against the original Q8
noedit outputs, phase costs and physical selection. Verify matched pre-state
and post-sampling RNG, full pre-anchor history, and all 20 Q projections.

**Mechanism:** use the existing four-denoise beneficiary observer, separating
the common selected MASKs from the single displaced MASK. Primary numerical
comparison is the paired, per-request equal-weight change in stable wait on
common MASKs: rank minus base; rank minus random separately. A strictly
negative difference favors earlier agreement. Also report the combined
common-plus-victim row-round balance; no claim of free benefit when the
victim pays for common gains. All reference IDs are base delivery, not a
ground-truth answer. Exclude only undefined offsets from this specific
mechanism metric and report their counts. When the block finishes early,
carry the last observed state; do not count fictitious forwards. Report
anchor coverage, admissions, direct candidate edits and spread across cases.

**Serving:** macro per-request N/F, correct/30, and pooled N/physical Q; all
four generation stages charged. A deployment pass requires strictly more
correct requests than base without lower unrounded TPF. Rank vs random is a
separate attribution check. Report original Q8 IBS/OBS and repeated gate for
context, not as matched physical expenditure. No retuning or seed shopping.

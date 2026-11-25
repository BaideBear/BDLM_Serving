# Frozen protocol: unresolved neighbors in slack-only Gemma seats

Specified before inspecting the boundary results or archived comparison totals.
Use the verified native Gemma B256 L15 diagnostics, original frozen 30 requests,
and the existing joint high-drift old-row labels and tie handling. Keep the
five dataset-stratified whole-request folds (24 train, six held out). Fit on
all training old-pool rows as before. No test-frame fitting or retuning.

Append exactly two columns to the ten position + prior-write features. For an
eligible accepted old row at position i, use 1/(i-j) for the nearest unresolved
j strictly to its left, and 1/(j-i) for the nearest strictly to its right.
An absent neighbor contributes zero. Use the acceptance mask **before** the
current step, within this canvas only. No current KV, drift, logits, token IDs,
proposals or current acceptance may enter the ranker. Keep the existing ridge
logistic fit (L2=2, unpenalized intercept, no scaling) and position tie-break.

For Q16/32/64/128 replay the same hard-slack layout with old reserve cap Q/8;
never displace a MASK. Preserve mandatory fresh, actual width, reserve count
and neutral leftmost backfill. At Q256 use the existing dense no-reserve layout.
Compare the twelve-feature ranker with the ten-feature fit and original
frontier on the same states. Reproduce the archived comparator totals before
interpreting results. Count all selected high-drift old rows **including
backfill**, once per selected row per step. Report old-row swaps, high-drift
hits gained and lost relative to zero, and MASK displacement; budgets are
separate replays and are never added together.

Recommend a sparse rollout only if total held-out hits strictly exceed **both**
comparators at Q32 and at least one of Q16/Q64 and are no lower than either at
the other. Q128 and Q256 are descriptive and cannot rescue the gate. This tests
sparse-available inputs on a native trajectory, not actual sparse quality,
throughput or a causal benefit from refreshing a row.

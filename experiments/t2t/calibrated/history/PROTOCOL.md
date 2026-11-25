# Frozen before running the history ablation

Decision: practical improvement requires strictly more total reserved strict
old-edit hits at two or more of Q4/1, Q8/1, Q16/2 and no decrease at any of
the three. A tie is not a gain. On failure stop; do not retune features,
window, penalty, threshold, or budgets. Top-1/4/8 and intervals are context,
not alternative decision rules. Frozen30 has been used in previous studies;
this is a prospective ablation, not a new independent test set.

Fit both arms on the same disjoint verified-identity calibration24 split with
the existing positional logistic fit (L2=2 on slopes, unpenalized intercept,
no scaling or tuning). The comparator is position only. Append exactly five
raw features on each old candidate in the second arm: (1) own count of strict
edits since block start, clipped at 2; (2) own ever-strict indicator; (3,4)
counts of strict events at offsets -4..-1 and +1..+4, separately, summed over
the preceding at most four completed rounds; (5) total non-self writes at
those offsets over the same rounds. Count repeat events, not unique rows. A
write is an admission or a changed non-MASK to a different non-MASK; a
revocation, unchanged proposal or prompt row is not a write. All features
are computed before observing the current transition. Reset at each block;
bootstrap writes seed history. No logits, KV, token-ID features or lexical
classes. Only selected native states and non-fresh, already-resolved rows
are scored. Use the existing shortlist and reserved-seat evaluator, not
equal top-k as a proxy for actual reserve availability.

Report paired history-minus-position hit-count differences. For contextual
95% intervals resample whole requests with replacement within each of the
three datasets (10 per dataset), sharing draws between arms and all six
metrics: 10,000 replicates, NumPy PCG64 seed 20261126, nearest 2.5 and 97.5
percentiles, weights held fixed. No sparse rollout or quality claim.

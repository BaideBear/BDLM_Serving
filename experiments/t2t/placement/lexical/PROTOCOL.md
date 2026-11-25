# LLaDA lexical hazard ablation

Freeze before scoring the lexical arm on native30. Unit of prediction: a
non-fresh, accepted output row in a still-masked, non-bootstrap B32 step.
Predict a strict accepted-token change, not revocation, KV drift or quality.

Compare the original 25 positional features to those same features plus one
six-way one-hot. Decode the current OLD accepted ID alone using the pinned
tokenizer.json's decoder, before reading the step outcome. Special IDs, empty
decodes and Unicode replacement-character fragments are `other`. Nonempty
whitespace-only tokens are `whitespace`; otherwise strip edge whitespace and
classify characters as Unicode digit, alphabetic, punctuation/operator (ASCII
punctuation or Unicode P* or Sm), internal whitespace, or other. All alike
yields that class; different kinds yields `mixed`. No token frequencies,
neighbour text, model scores, current-step emissions or fitted token mapping.

Fit both arms separately on the same disjoint calibration24, using the existing
logistic fit (L2=2, intercept unpenalized); no gate, tuning or refit on frozen30.
Use the original verified 30 traces, same smaller-position tie-break and
hard allocator. Report top-1/4/8 and fixed reserved Q4/1, Q8/1, Q16/2.
Precision = total reserved/list hits per seat, recall = hits per 191 selectable
residual targets. Ordinary backfill hits are not reserved hits. A promising
cross-budget result requires more reserved hits in at least two budgets with
no loss in the third; shortlist changes alone do not qualify. No further
tuning on these requests. This is a native-history replay, not a sparse rollout.

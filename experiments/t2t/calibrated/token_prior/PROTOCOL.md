# Token-specific edit risk, conditional on position

Freeze before evaluating the native30. The visible, already-accepted ID may
carry edit risk beyond the old row's position. Use the frozen calibration24
positional model as is. For each token t, sum predicted edit probabilities E
and observed strict old edits O on all its eligible selected-phase rows on the
same 24 calibration requests. Add log((O+1)/(E+1)) to the positional ranking
score; unseen tokens have offset zero. This is a smoothed rate ratio, not a
calibrated probability. One pseudo-event; no tuning, clipping, lexical buckets
or checkpoint access. At serving time this would require only one lookup per
old candidate and no additional forward, KV, draft or logits. Training on
calibration labels is intentional; no frozen30 label enters the table.

Use the same 30 native traces, layouts and tie order as the positional
comparator. Main comparisons: Q4/1, Q8/1, Q16/2 reserved-seat hits out of 191.
Advance to online testing only for strict improvement at two or more and no
decrease at the third. Top-1/4/8 rankings are descriptive. Report seats and
request-level differences and publish no token IDs or ID-level rates. Do not
infer sparse rollout quality or TPF from native-trace recall.

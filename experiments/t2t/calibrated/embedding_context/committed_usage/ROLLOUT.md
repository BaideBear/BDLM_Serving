# Committed-usage sparse rollout, frozen before the first live result

Use the unchanged native-study fit, SHA256
`41e96254fa0c3af75cf0a46278f7fefbbd7a72470ba2be4bd3861edfa0adbb11`.
Do not tune on the 30 endpoint scores. Run physical LLaDA sparse decode at
Q4/E1, Q8/E1, Q16/E2 on the same frozen30 tasks, seeds and limits. All
fresh writes (including strict edits) must be selected; allow overflow.
Reuse the fixed context allocator: only change the old-row ranking function.
Build a separate counter for each request on completed generated suffixes.
Archive and count charged bootstrap, verification, commit, and observed
per-layer Q, not just nominal sparse seats. Report macro TPF, correct/30,
pooled token efficiency and strict reserved hits per actual reserved row.

Compare separately to the same-Q zero reservation and the exact previous
28-coefficient fixed context rollout. Also include historical IBS and OBS
where available. Per-budget deployment criterion versus zero: strictly more
correct and nonlower macro TPF. A gain only versus old context, on native
hit counts, or in total correct at lower TPF, is not a deployment pass.
This is exploratory: frozen30 has been reused; no independence or generalization
is implied by confidence intervals on these cases.

# LLaDA Q8: decide when a MASK may give up its seat

This experiment changes timing, not which candidate ranks first. At an ordinary
Q8 zero-reserve selected step, use the previously frozen embedding 5% gate.
Discard `swap_old` offers. Only a real `swap_mask` meeting all original
protections can be approved; every previous newly decoded or edited token
remains mandatory even on overflow. No more than one exchange per request.

Fit a binary logistic model on marginal-victim admissions from all crowded
gate offers on the existing disjoint calibration24 native traces. Fixed
features: intercept, victim's MASK ordinal / 7, MASK count / 32, fresh count
/ 8, distances to left/right MASK and nearest fresh capped at eight and divided
by eight (missing distance eight), and MASK density within three positions.
L2=2 on slopes, unpenalized intercept. Train only once. Approve scores at or
below the median predicted risk on calibration offers; use that calibration
approval frequency for a runtime Bernoulli timing control. No full-row logits,
current outcomes, labels, fresh embedding lookups or online KV measurements in
the victim score.

1. Native replay, frozen30: keep entrant/victim/gate fixed. Report retained
   strict edits and base admissions lost vs original gate, exact uniform
   acceptance with the same count **per request**, and hindsight (diagnostic
   only). A screen is informative only if it retains more hits and loses fewer
   admissions than the matched random expectation; no refitting if it fails.
2. Sparse stop-on-first: use the measured archived first-offer rank path if
   approved, otherwise its verified zero-reserve path for the rest of the
   request. No new forward is needed. Report matched-count timing expectation
   by dataset, and both labels from the matched fork. Do not call this a new
   GPU run.
3. Direct sparse **defer**: compare first approved low-risk crowded offer vs
   first approved calibration-rate coin offer. Both can defer across steps
   and blocks and use the identical old-position ranker. After the first
   approved exchange revert permanently to zero reserve. Run both directly
   regardless of the native replay outcome: the transfer of a native
   admission score to a sparse path is the question. Controls need not realize
   identical accepted counts. Fixed 30 original cases (10 per dataset),
   same seeds, both cards qualify against native B32 first. Against zero reserve
   require strictly more correct and macro TPF no lower at full precision.
   Against coin report attribution separately, including real hits, victim seat
   opportunity when identifiable, physical Q/fresh/MASK allocation, F, N, and
   pooled physical efficiency. After trajectories diverge do not infer missing
   counterfactual admissions from off-Q rows.

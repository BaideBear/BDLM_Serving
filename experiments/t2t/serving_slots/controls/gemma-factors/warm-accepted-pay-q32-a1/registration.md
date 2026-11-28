# Q32 verified EOS: accepted dirty rows are payment-only on mixed warm samples

Fixed head-only Q32; keep normal first tile, 1/8 clean right edge, verified
EOS, no successor or paid-mask reuse. On a sparse sampler call beginning
with all 256 KV and SC rows resident and sampling any unresolved row,
physically project the unchanged complete plan. Do not consume logits on
already accepted dirty rows on this same payment. All other sampled rows
(including dirty unresolved and accepted clean-prefix rows) retain their
original eligibility. Never skip an input/acceptance debt, synthesize logits,
advance an extra clock, delete a verify chunk, or omit a required encoder.
This changes where T2T is allowed and the random draws/SC downstream; it
does not change acceptance thresholds, hold a row permanently, or implement
IBS acceptance. The first physical projections on the first changed tick
must hash identically. Track possible missed edits and revocations on the
control, plus new admits and later warm work on each arm.

Control historically reproduced: `/gs/results/verified-eos-1128/C32-pair-XX.json`
`on`, all fields. One instrumented real GPU1 worker, both arms on that same
empty A800, alternating order, frozen task/seed/input/scorer. Deadline
2026-11-28 15:00:00 +08 (1795849200); no holder, retry or successor.
Run unique anchors [0,10,20,30,50,70] first, then the remaining original30
and new60 only if qualified. Require exact control replay in all six, an
actual changed sampler input in each cohort, same first physical logits
and protected work, no net loss of correct answers across six, positive
paired mean TPF delta, and lower total F or lower total selected decoder Q.
Gate is screening, not independent evidence of an improvement.
Report any partial cohort as partial. No promotion on observed quality loss
or no demonstrated TPF improvement; do not combine factors or retune.

Motivation: strict capped IBS at Q32 has 5,973 decoder calls selecting MASKs
and 221 replay-only in 6,194 decoder calls (186,829 decoder Q); our reference
has 1,576 pure drains in 6,794 decoder calls, including 221 warm drains.
The IBS MASK-work calls all admitted at least one row. Its replay-only
category is not our warm phase. IBS samples only unresolved positions, has
admission-only debt and emits a resolved canvas, unlike our native selected
entropy sampler, re-noise/T2T writeback and ordered argmax close. Only the
single above eligibility rule is borrowed; no claim of equivalent work.

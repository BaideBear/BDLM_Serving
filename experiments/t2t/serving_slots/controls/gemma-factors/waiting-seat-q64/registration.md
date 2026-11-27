# Prelaunch registration: one warm unresolved seat at fixed Q64

Both arms use the pinned A7 fixed verified-EOS Q64 **on** reference in
`/gs/results/verified-eos-1128`, not the pre-EOS baseline. Fixed canvas
B256, Q64, sparse selector 00, right 1/8, guarded first verify and ordered
measured EOS. These two arms differ only in which *one* clean unresolved
position gets a discretionary seat in a contended, fully initialized head.

Rationale: the prior fixed90_fused Q64 split was 1,204 cold, 1,967 warm
(200 pure drains, 1,767 sampled), 149 closing drain, 688 verify and 82
post-prefill encoder F. The EOS control removed exactly 141 suffix verifies,
not warm sampling. Thus 1,767 is an upper bound on warm seat opportunities,
not the number of actual swaps. Prior oldest/uniform tests ranked accepted
prefix seats. Paid MASK reuse changed a pure payment into a sample. Neither
is an age-based exchange between clean **unaccepted** positions.

Hypothesis: late-index unresolved rows that have not been projected for a
long time can hold up block completion while lower indices receive repeated
service. With the head fully initialized, shadow both choices: keep every
dirty row, every original right-edge prefix seat and the smallest unresolved
row if selected. Consider the highest-address remaining selected clean MASK.
Exchange it for the least recently **physically projected** outside MASK
(lower index breaks ties) only if the outside row is strictly older. No
second exchange. No logit or attention value is read by the planner.
Sample actual fresh logits in sorted selected order once. Width, cold
policy, RNG rule, T2T, re-noise, entropy acceptance, age limit, dirty
payments, conditional EOS and encoder remain unchanged. The chosen IDs,
future bill and answer need not be equal after intervention.

Predeclared operational gate: Q64 only, real A7, 0/10/20 first and included
in the maximum 90. Alternate arm order. Off must match the complete historical
on receipt (including decoder/argmax tape and encoder digest). The first
different physical call must be the same-state, same-width one-seat exchange;
an unexposed pair must be identical. Any registration, physical, clock,
scoring or replay failure stops the study, with no replacement index.
**Do not expand** if no exposed anchor, any correct-to-incorrect anchor,
no individual reduction in paid F, or aggregate anchor F increases.
No tuning or retries of the policy after these results. If qualified,
use indices 0..89 exactly once per arm; report separate original30/new60 and
their union, score rescues/losses, N, physical F/Q, macro mean N/F, exposures,
and first changed rows. For a positive fixed-profile recommendation require
lower total F, higher macro TPF and no net quality loss in either cohort.
An anchor no-go is not a measured effect on the remaining 87.

Physical lane protection: require the prior EOS handle complete receipt,
absence of `/proc/122082`, and an empty physical GPU7; check again just
before loading weights. Bounded 3,600 s, no holder and no other GPU.

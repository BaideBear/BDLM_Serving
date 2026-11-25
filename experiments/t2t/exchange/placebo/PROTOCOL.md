# Does the exchange need the predicted candidate?

Lock this protocol before the new GPU runs. Use the exact frozen30, original
seeds, B32, cap 2048, checkpoint, scorer and exchange table at `dad7d55`.
The primary budget is Q4/1; Q8/1 is the prespecified negative control.
Compare the historical 4x exchange and zero reserve to two fresh executions
with the unchanged 4x gate/victim and a uniform old candidate. Do not treat
the two random streams on the same questions as 60 independent requests.

On its own current state, each placebo first calls the actual 4x lookup rule.
If that rule would keep the zero-reserve Q, execute it unchanged. Otherwise
retain its victims, number of swaps and mandatory fresh. Draw from eligible
previously resolved nonfresh rows outside zero-reserve Q, without replacement;
eligible ranked candidates remain in the pool. Use Python's private `Random`
with key `exchange-placebo:{a|b}:{request_seed}:{block}:{forward_clock}`.
The draw neither reads nor changes the sampler RNG. The selected gather and
all physical forwards, bootstrap, check and commit stay unchanged. No probe.

Require original full-width output and work parity on 30/30 in each GPU run.
Check agreements up to the first physical Q difference vs ranked. After that
the model may generate a different trajectory: gate/victim are matched at a
given state, not forcibly synchronized across diverged states. Compare paired
correctness, mean request N/F, sum N / sum physical Q, work, first divergence,
swap type and strict edit of the added row. List both placebo arms irrespective
of outcome. Higher edit coverage alone is not evidence for an end-to-end gain.

Secondary diagnostic (specified while the GPU workers were running their full-width
bridge, before any placebo output): on a visited swap state with `K` eligible old
rows, let `hit` indicate whether the **drawn** candidate strictly edits in the
actual forward. Record `K * 1[draw == ranked] * hit` and `hit`. Dividing the sums
by the number of swaps gives a one-step inverse-propensity estimate for the
ranked action on placebo-visited states and the observed uniform rate. Recheck
each deterministic draw against the published key; keep singleton pools and
report collisions, observed collision hits and effective support. Interpret
under uniform pseudo-random assignment, not as unobserved labels, full policy
evaluation, or an unbiased finite-sample trajectory estimate. Report A/B
separately and bootstrap whole requests within each dataset for descriptive
dispersion; this does not reflect policy search, unseen tasks, or model seeds.

Additional sparse-state analysis fixed during the second random streams, before
reading their reward/score totals: reconstruct last physical query time from
earlier forwards in the same block. Estimate strict-hit rates if drawing uniformly
from the oldest or newest `ceil(K/2)` eligible rows, keeping the original gate
and victim. Split ties fractionally at the boundary; all-tied states have weight
one in both arms. Importance weights are at most two. Report all states (including
singletons), the number of age-informative states, weighted rates and a paired
request interval. This is candidate screening, not a deployed controller. Require
the oldest-minus-newest contrast to be positive in each budget and each stream
before considering a dedicated cache-age rollout; do not select other cutoffs.

Direct follow-up after all four contrasts passed: execute that exact oldest-half
policy at Q4 and Q8, for draw salts A and B separately, on the original frozen30.
Retain the same 4x gate, victim and one-entering-seat cap. Choose a uniform set of
boundary ties with the base random key plus `:half-tie`, then use the original
key on the sorted restricted pool. If all times tie, sample the full pool: same
marginal distribution and exact original draw. Reset clocks at each B32 and log
each pool, prior times and actual physical rows. Compare each stream to its own
uniform run, the original rank and zero reserve. Do not pick the better salt.
Advance a budget as a quality/TPF candidate only if in **both** salts it answers
more than zero reserve with macro TPF no lower, and neither metric drops versus
its corresponding uniform arm. Keep all costs regardless of the outcome.

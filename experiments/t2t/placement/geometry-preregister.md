# LLaDA B32 geometry-v1: fixed-rule preregistration

Locked before running this probe or opening its raw per-step evaluation
labels. The historical position-hazard finding motivates the rule; no
coefficient, threshold, candidate radius, or per-case policy is trained or
chosen on frozen30. The reused cohort is a descriptive check, not a new blind
test. No independent calibration set is used.

## Rank

Use the caller's exact pool of editable, resolved, non-fresh positions.
All indices are block-local. For candidate `i`, compute:

- `dm`: distance to the nearest editable MASK.
- `df`: distance to the nearest fresh write `j` with
  `0 <= last_write[j]` and `last_write[j] >= last_query[i]`.
- `g = max(dm, df)`: the smallest radius containing both anchors, possibly
  on the same side. A single near anchor does not suffice.
- `s`: minimum `abs(i-j) + clock - last_write[j]` over distinct previous-step
  strict-edit anchors satisfying the same dirty-write predicate. One tick
  adds one distance unit by convention, not from a fit.

Missing distances are infinity. Rank in increasing lexicographic order by
`(min(g,s), g, s, dm, df, -last_write[i], last_query[i], i)`. A prior strict
edit can supply local risk even away from a MASK/fresh intersection. On
equal primary risk prefer the intersection; the remaining keys handle
missing anchors and deterministic fallback. Same-tick writes count because
the query precedes the write. No token IDs, logits, confidence, or case
identity are inputs. The ranker does not assign seats or activate a policy.

## CPU evaluation

Evaluate all 30 pinned LLaDA 2.1-mini B32 native traces, with one fixed policy
per arm across all requests. The four nominal budgets are Q4, Q8, Q16, Q32:
slack-only caps 1, 1 and 2 respectively for the first three; Q32 is a
full-width control with no reservation. Q16 fixed 2 is a separate
MASK-displacing negative-control arm. Slack is measured after accounting
for all fresh and MASK rows, not just the first mandatory MASK.

Use teacher-forced native before/after writes, fresh and prior strict edits;
maintain one simulated query clock per Q from the zero-reservation policy,
shared across all comparisons within that Q. Bootstrap and no-MASK verify
are full-width; only selected-stage steps can reserve a seat. Rank before
reading current acceptance/token changes. Labels are resolved-to-resolved
ID changes in the editable span, with the pre-step fresh set removed for
old-position hits. No proposal/draft outcomes are used.

For each arm report the reserved seats, old targets and reserved hits,
equal-seat `hard_frontier` reserved hits and exact uniform expected reserved
hits (`seats * old_targets / old_pool` in each state); also report old hits
including neutral backfill and the change in MASK coverage against
zero-reservation. Count selected and full-width stages separately; include
no-MASK strict edits only in the all-stage denominators/hits. The geometry
ranking may change reserved hits without changing executed positions. Do
not infer online quality, TPF, or speed from this probe.

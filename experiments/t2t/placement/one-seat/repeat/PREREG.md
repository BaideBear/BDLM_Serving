# Gemma Q32: prospective execution of the conditional first swap

Before reading repeat outputs, freeze the checkpoint, all original 30 tasks in their
original order, B256, cap 2048, and the `plan_seat` ranking and victim ordering at
commit `1ccac4d`. Use two groups, original seed and original seed + 1000003 on
each request. The repeat is paired on seeds and is not a held-out question set.
Do not omit or replace any request. Score completed, capped and failed requests
under the original protocol. No threshold tuning on either set of outputs.

* Every newly written row (including a strict edit) must return on the next
  forward. Zero-reservation allocates remaining Q to unresolved rows, then old
  backfill. No hard truncation of mandatory overflow. Debit actual physical rows.
* In both direct arms observe the first legal physical swap. The control takes its
  unchanged Q. Treatment exchanges its victim for the targeted old candidate
  **only if that victim is unresolved**; any legal anchor consumes the opportunity.
  Never search again after a nonmatching first anchor, even across canvases.
* At each seed compare independently executed base and conditional arms. For the
  original seed verify per-request output IDs, N, F and physical Q against the
  historical rank/base splice. For the new seed collect native references with
  the unmodified model and require the full-width event/output/work bridge.
* Audit the paired pre-state, Q sets, victim queried, pre/post acceptance at the
  victim and equality of non-intervened outputs/work. A renoised value change
  without an acceptance transition is not an admission.
* Report actual quality, mean request N/F, sum N/sum physical Q, wins/losses,
  legal/matching anchors and paired victim admissions. Quality gain with no
  TPF loss is a descriptive criterion, not a significance test or a speed claim.

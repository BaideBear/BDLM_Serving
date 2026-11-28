# Can the first proposed MASK displace one clean replay row?

This is an audit of sealed LLaDA 2.1 capped IBS traces, not an inference
result. The independent input fingerprints, physical F and physical Q match
the [paid-tick audit](ibs-paid-ticks.json) at every offer.
[Counts and cohort splits](ibs-frontier-seat-exposure.json).

The cap already fits all last-step writes. Consider a swap only when the
original selection fills the cap with resolved rows, and the original IBS
candidate list nevertheless contains a MASK. Replace the rightmost selected,
already projected, resolved, non-prompt row without writeback debt with the
first proposed MASK. Reject a swap if any KV between the block start and the
new right edge would be missing. No logits or post-forward information enter
this decision. All other selected rows retain their normal sampling rules.

| Fixed Q | Original F | Full calls that omitted all proposed MASKs | Eligible swaps | Requests exposed | Donor edited in original call |
|---:|---:|---:|---:|---:|---:|
| 4 | 27,397 | 5,604 | 1,402 | 85/90 | 2 |
| 8 | 19,692 | 2,340 | 622 | 84/90 | 1 |
| 16 | 15,480 | 665 | 245 | 70/90 | 0 |
| 32 | 12,694 | 0 | 0 | 0/90 | 0 |

- Q4: all six requests that reached the generation step limit could use the
  swap on their last call. They account for 364 of the 1,402 eligible calls.
  Of 84 requests that completed generation, 79 are also exposed. Their 1,038
  eligible calls include both observed donor edits. Completion is not a
  correctness score. Do not remove the six failures or extrapolate recovery.
- Every incoming MASK in this table is a first projection, **not** a
  previously projected and waiting MASK. The donor would have changed its
  token only rarely, but its projection, as well as first projection of the
  new MASK, can change other rows' attention. At Q4 other selected rows
  actually edited 327 times on the eligible original calls. Preserving their
  T2T eligibility is not a guarantee that the new logits or commits match.
- Q16 has one failed generation; it never met the trigger. At Q32 the rule
  is a no-op on all original paths. Gemma's existing capped IBS already
  reserves a first MASK after writeback, so this audit is not evidence of a
  new Gemma feature.
- Use the same frozen prompts, seeds, ordinary stopping rules and scoring
  for the prospective GPU test. Keep the original control, observer-only
  replay and treatment distinct. Report original 30 / new 60, previously
  completed / previously failed, quality flips, and measured N/F/Q. No
  estimated speedup follows from counting eligible calls.

Reproduce without running a model: run `python3 tools/ibs_frontier_exposure.py
<frozen-IBS-directory> --q <offer>` on n164. The four source directory names
and the sample manifest hash are in the paid-tick audit linked above. The
scripts leave raw answers and per-step traces on n164.

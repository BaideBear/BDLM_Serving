# How many additional proposals should one paid step accept?

Registered screen, not a result. Compare two rules on hard-budget paid IBS:
the archived [two-measurement rule](v1.md), and the same rule with one
allowance on its *additional* admissions. Do not change the original label
or the base sampler, physical Q selection, RNG draws, EOS, mandatory KV
writes, or Gemma encoder and self-conditioning.

For each originally rejected ordinary token proposed on the same MASK in
two consecutive measured decoder calls, retain the existing strict `>0.5`
threshold on both observations. Assign that row cost
`1 - min(p_previous, p_current)`. In order of increasing cost (ties go to
the lower absolute position), keep the longest prefix whose total cost is
at most `0.5`. Return chosen rows in the original sampler order. Unselected
positions have no vote. A missed physical tick, block change, native
acceptance or changed proposal invalidates the pair. The number `0.5` is a
sampling hyperparameter, not a calibrated probability guarantee or an extra
physical Q budget. A single qualified proposal always fits; this changes
only the case where multiple proposals compete.

- Model pairs: LLaDA 2.1 Q16 on H0, DiffusionGemma Q64 on A2. Same original
  fixed six (`7,13,25,33,55,71`), no task replacement or length filtering.
  Freeze one allowance for both models before observing treatment outcomes.
- On each board, find states where the allowance actually rejects a v1 extra.
  Do not infer effects from aggregate counts. If prior receipts lack a
  predecessor score, use a measured read-only observer, not an imputed score.
  If no state is exposed, stop after observation.
- Reproduce the same-board parent control and/or its archived probe using
  IDs, decisions, F and Q. At the first intervention, verify identical
  pre-state, physically measured rows, proposals and native acceptance;
  the only difference may be a subset of additional admissions. All
  following writes remain real and billed. Compare the two policies and
  include the separately named, same-board unmodified IBS point.
- Do not expand a candidate that is point-dominated in correctness and
  macro TPF by the original paid IBS or the uncapped repeat rule. To give
  a new 90-case run priority, require six-case correctness and TPF at
  least equal to paid IBS, with one strictly improved, and not both total
  F and Q higher. Report disagreement with stronger quality references
  separately. Do not declare quality superiority from six.

This measures whether limiting the combined uncertainty of extra writes
changes the quality/forward trade-off. It does not estimate a pure ranking
effect at equal admission count, or select a new prefix quota. The shared
[implementation](../../../../../bdlm_serving/methods/t2t/capped_agreement.py)
extends the frozen eligibility rule, and has no access to future observations.

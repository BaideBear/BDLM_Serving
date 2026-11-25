# Predicting native token edits

Native LLaDA 2.1-mini B32 and DiffusionGemma 26B-A4B B256, same frozen 30
prompts and seeds as the published baselines. No FlowBlock stopping changes.

1. **Observe native paths.** Record each denoise step's block, clock, input and
   resulting IDs, resolution/acceptance, draft and confidence proxy, on the
   cluster only. Preserve the original model calls and sampling. Batch one
   matches both archived native runs. Gate on every request's output IDs,
   logical and physical work, and per-step counts where available.
2. **Forecast from history.** At step t, compare rankings using only information
   through t-1. For LLaDA an edit changes a non-prompt previously resolved ID.
   For Gemma the primary label requires acceptance on consecutive steps and
   a changed accepted ID. Re-noising/revocation, re-admission, and argmax draft
   churn are not interchangeable with this label. Include the entire charged
   canvas; values are not limited to the delivered pre-EOS prefix.
3. **Allocate real Q.** Reserve configurable slots for newly resolved/edited
   IDs (their new token has not yet been encoded), unresolved positions, and
   predicted edits outside the refresh set. De-duplicate before executing;
   account for closure/re-encoding, real query rows and padded rows. Compare
   with no-targeted-edits and equal-cost random selection, at identical nominal
   budgets. Teacher-forced coverage is not an online speed or quality result.

## Protocol

`tools/trace_native.py` takes `--model llada21|gemma --checkpoint DIR --samples
FILE --reference DIR --output NEW_DIR`. Use separate pinned model environments
as for the baseline results. Run with the repository on `PYTHONPATH`. No raw
traces, prompts, answers, weights or cluster paths are published.

`tools/analyze_t2t.py --model MODEL --raw VERIFIED_TRACE_DIR --out TABLE_DIR`
evaluates last-step low confidence (native top-1 probability on LLaDA; top-2
logit gap on Gemma), youngest token, distance to another changed position,
and youngest then least confident. Positions break ties. Expected uniform
sampling uses the same number of seats in each individual step. Report micro
precision and recall at 1/2/4/8 seats both over all resolved positions and
outside the mandatory-refresh pool. No extra current-step logits as features.
Compare all 30 descriptively; within each dataset the first seven manifest
indices are development and the last three held out for positional scoring.
These 30 prompts were used in earlier research; not a new independent quality
confirmation set. A dense-trace ranking does not prove stale sparse features
work online. The CPU allocator in `bdlm_serving/methods/t2t/planner.py` is not
a sparsified GPU forward.

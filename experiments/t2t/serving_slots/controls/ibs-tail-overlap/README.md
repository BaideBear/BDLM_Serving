# LLaDA Q8: one physical call for the head and its successor

Experimental factor on strict capped IBS, fixed Q8, not a measured gain. The
[shared rule](../../../../../bdlm_serving/methods/t2t/close_overlap.py) supplies
only genuine unused seats. The [adapter](fused.py) extends the projection, not
the head's dense indices or sampling state. Keep the original head key horizon;
successor queries can read the sealable head and the projected successor prefix.
The head and successor both apply the original IBS sampler to their own group.
Writes in the successor are initially tentative: expose them to the next tick
only after a clean head retirement, then require all changed successor inputs
to be projected before freeze. Otherwise invalidate their cache eligibility and
discard the writes. Charge the full joint projection in either case. This does
not authorize caching speculative Gemma KV across a generation encoder call.

**Implementation gate:** the [end-to-end unit tests](../../../../../tests/test_ibs_joint_projection.py)
run the actual pinned IBS source and upstream controls against a two-layer
instrumented fake forward, not against a model. They check stock/no-op identity,
physical invocation count, row-specific attention masks, returned inputs
on the first successor payment, invalidation on a head edit, a prompt ending
mid-block, and the actual output cap. Set `BDLM_OPTIMUS_CONTROLS` to the
directory of the two pinned upstream control sources to run without the
submodule. Tests are not evidence of accuracy or TPF. The paired GPU screen
uses frozen indices `7,13,25,33,55,71`; a fused-width change can change head
logits numerically even when the legal KV addresses are identical.

## Six-case decision: do not expand this variant

All 18 GPU requests completed on physical A1. At fixed Q8, using spare head
rows for the next block did not improve the quality/TPF trade-off against
strict IBS on the same board. No-op matched stock on all six full traces.
This pair used the original IBS stopping endpoint in **both** arms.

| Arm | Correct | Mean request TPF | Pooled TE | N | F | Decoder Q |
|---|---:|---:|---:|---:|---:|---:|
| Strict IBS | 6/6 | 2.5355 | 33.41% | 1,474 | 593 | 4,412 |
| Fused successor, ordinary IBS acceptance | 5/6 | 2.5360 | 28.70% | 1,418 | 645 | 4,941 |

43 joint forwards charged 186 tail rows. All 43 promotions succeeded, and
all 139 accepted non-MASK writes were reprojected with their actual inputs.
Five cases used fewer F, but #55 used 146 vs 261 and went from correct to
incorrect. It generated 362 vs 381 nonspecial tokens, with 0 vs 6 T2T edits.
Both trajectories reached EOS normally. Later blocks also needed more
forwards, so the extra work cannot be attributed just to one more block.
We do not know which particular early choice changed the answer. Do not
remove this case. This rejects the tested intervention, not block-causal
attention or every possible overlap policy.

[Per-case scored totals and raw digest](six-measured.json) ·
[Frozen runner](runner_a1.py) · [Rescoring script](report.py).
Run `report.py RUN_DIR FROZEN90_SAMPLES` in the pinned n164 scorer environment.
Only the final complete stream from `results/llada21-ibs-q8-tail-overlap-a1-v1`
(`runner.exit=0`) was used. A failed earlier partial stream that included
case 7 was overwritten during development; no metric or sample from it was
merged. Future attempts must have distinct output directories.

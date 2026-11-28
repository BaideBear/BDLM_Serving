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

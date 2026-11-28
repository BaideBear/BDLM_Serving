# Does refreshing an already-paid generated prefix affect nearby MASK decisions?

Before execution: DiffusionGemma, paid IBS, fixed Q64, physical A0, fixed six indices 7,13,25,33,55,71, no replacements. No change to decoding, sampler, termination, or the paid Q limit. This is an instrument, not a proposed decode speedup.

Per request take the **first** block-0 sampler call with a positive paid contiguous frontier, at least one omitted prefix row and at least one selected unresolved row. If none exists, retain the request as unexposed. In one state with identical input IDs, self-conditioning, legal K interval and selected MASKs, run three additional full-visible-width decoder calls:

1. Clamp **every** originally omitted row's per-layer K and V to the corresponding resident values; selected rows use their projections. This controls for changing projection width.
2. Repeat exactly; require bitwise-equal selected logits.
3. Remove the clamp **only** on the omitted, paid, generated prefix inside this physical block. All other omitted K/V keep their original resident values. Encoder prefix is unchanged, no future K is introduced. Recomputed prefix rows consume their stored self-conditioning; that is part of the operation under test, not an independent self-conditioning effect.

Record real production widths and whole-request identity for stock/observer/probe. Record the number and identities of affected rows, top1, confidence crossing, and actual native admission including fallback. Diagnostic F/Q are separately charged and never counted as a production speedup. Report clamp-vs-real separately from fresh-vs-clamp. A divergent control admission makes that request ineligible for attribution to its real paid step; do not remove it from the registered six. If zero control-valid requests flip an admission, do not tune the dose or expand this state to 90 on these data alone. An observed flip is exposure, not proof of accuracy or TPF benefit.

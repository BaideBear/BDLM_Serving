# First physical Q32 tile: original random KV only

Use the frozen verified-EOS Q32 on path, right-edge clean prefix 1/8, no
tail, no paid-mask reuse, no other factor. Only suppress sampling on the
first physical tile [0,32) of each new head. Still project it through all
layers and LM head, discard these logits, keep its KV. No token/acceptance
write, SC, RNG draw or sampler tick on that first call. Later first-time
sampling uses fresh projections and missing SC (zero conditioning), not
invented SC. All other cold tiles take the normal actions. Use unchanged
row selection, block seed, physical verify/EOS and all mandatory encodes.

Distinct from half-pacing (smaller repeated cold samples), frontier-last
(different physical row order), and rejected-noise retention (ordinary
sampling/RNG/acceptance and a changed noise writeback).

Authorize A1 only with zero memory and no compute processes; deadline
1795843443. One registered worker, alternate off/on, frozen tasks/scorer.
Require full off row replay of each historical verified-EOS Q32 on receipt
including every step hash, output and encoder digest. Three anchors 0/10/20
first: same initial RNG/canvas, same real Q32 positions, mask/attention route,
off has sampled writes, on has none; every deferred row physically reprojected
and sampled after KV coverage, no SC or clock fabrication. Require a changed
action and local next-call cost (dirty-row count); failure stops before
expansion. No quality-directed stopping. After passing run remaining original30
then new60 on same card, no repeat anchors.

Hypothesis: defer 32 first samples so the first projection creates no debt
and those positions receive more informative first logits later. This need
not save any net work: eventual sampling still requires a paid revisit and
its writes are mandatory. Report all physical F/Q, cold/warm/close/verify,
first-tile admissions and debt, macro TPF and pooled N/history decoder Q,
paired score transitions and uncertainty. No default recommendation for
worse quality or no demonstrated TPF gain.

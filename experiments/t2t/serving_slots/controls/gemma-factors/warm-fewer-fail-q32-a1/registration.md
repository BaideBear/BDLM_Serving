# Fixed Q32: one warm seat by unsuccessful real sample count

Off is the frozen verified-EOS Q32 on arm with the normal first tile, right
prefix 1/8, single head, original sampler and clock. Count each actual sample
whose row is unaccepted immediately afterward, including revocations. Counts
are lifetime per physical head, across cold and warm samples; pure projections
do not count. Never read logits to rank. On a contested full-width warm
sampling call protect every dirty row, accepted clean prefix, and the earliest
unresolved row. Consider only the last (highest local index) remaining
selected clean unresolved row. Swap it with the unselected unresolved row
minimizing (failure count, row index) only if its count is strictly smaller.
Otherwise retain the exact original plan. Do not alter acceptance or RNG.

Use existing full per-row evidence on the registered controls where available;
otherwise replay 0/10/20 on a truly empty A800 GPU1. No holder. To expand,
require at least one actual differing physical set on these anchors, with
an identical pre-difference trajectory, equal Q, conserved protected rows
and exact off replay; zero exposure is no-go, not a speedup. Do not select on
anchor scores. Fixed ORDER through original30 then new60, no repeat anchors.
Only count durable complete pairs, every missing case explicit. Exit by
1795843443 even if fewer than 90 finish; no automatic resumption.
Compare macro N/(post-prefill decoder+encoder F), pooled N/selected decoder Q,
matched correctness, actual Q, changed IDs, exposures, and losses/rescues.
No promotion on partial, no TPF improvement, or observed quality loss.

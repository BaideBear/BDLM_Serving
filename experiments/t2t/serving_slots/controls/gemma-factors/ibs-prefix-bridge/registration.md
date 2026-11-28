# Hard-capped IBS, warm-mask slack bridge, fixed nine pairs per tier

Before any GPU run: Q32 and Q64 use the same nine manifest indices
`0,10,20,30,50,70,40,60,80`, in that order. This is the original anchor in
each stratum and the first and midpoint indices of each added-20 stratum;
no selection by observed scores. Rank one request, B256, no OBS successor.
On physical A800 GPU1 only, deadline 2026-11-28 15:00:00 +08, no holder.

**One rule:** On a capped IBS plan that includes a previously visited unresolved
row, if original positions leave seats, add up to four consecutive clean,
previously projected, resolved rows ending at `cached_prefix_len-1`.
Stop at the first ineligible row. Preserve every original measured position,
unresolved sampler row, rightmost visible position, admission debt, and encoder.
Do not fire on a pure drain, cold-only MASK call, full selection, or when debt
would be left unpaid. Charge these real projections. As in original IBS,
each projected row refreshes its KV and its stored raw-logit SC; added rows
never enter sparse unmasking or get a token draw. No extra forward, clock,
threshold change, or new terminal test. Downstream costs and outputs may differ.

Hypothesis: a recent measurement of the committed front can improve attention
for revisited unresolved positions without displacing cheap IBS progress.
For each pair compare the naked pinned runtime to instrumented off exactly.
Check all 85 physical projections, positive MASK progress, all paid admissions,
EOS delivery, encoder inputs, and first change vs the same pre-call subset.
Any replay/invariant failure invalidates that pair and stops inference.
No adaptive case expansion or tuning. A worthwhile Q32 signal requires at
least one net correct rescue with exposure, Q32 macro TPF at least 98% of
off, and no net loss of correctness at Q64 with exposure. Even a pass is
only a reason for a separately authorized larger study. If control is at ceiling,
report quality headroom, not quality equivalence or recovery of six cases.

Plain control and off/on are independently run on the same device with identical
seed, prompt, scorer and checkpoint. Historical freeze is the original Optimus
gen code (`a09d0447...`), controller commit `7343a37b2a69337caa452f17b59727dc6665f039`,
frozen90 samples (`bc9e457d...`), original native config and scoring function.
Historical Q32/Q64 row archives are not present here; do not claim cross-board
per-case equality. Historical 79 vs 85 correct is a net 90-case difference,
not six identified rescue opportunities among these nine.

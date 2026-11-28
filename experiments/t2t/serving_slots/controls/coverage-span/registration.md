# Coverage-driven window, single request, LLaDA

Nine fixed indices 1/11/21/31/41/51/61/71/81 at **all five** fixed offers.
Three arms in rotated order: same-board live control, common span executor,
and the identical executor with coverage-aware admission. No retuning by
budget, no borrowing future Q, no padding to claim a full projected head.
The comparison to live is a complete alternative; only span vs covered
isolates a single factor. These nine did not participate in the preceding
risk-ribbon screening.

Every offer pays dirty rows first. In active physical blocks buy at most
one eighth of free seats in the clean right-edge prefix before considering
the contiguous span from the first MASK. Spare seats may read additional
prefix after all eligible spans. An unfinished head with one quarter or less
MASK and initialized KV allows the next physical block; front-to-back
attention remains forbidden. Within an eligible block projected positions
can edit; no new sample on a full-debt drain. A covered head is one for
which **every valid head row was actually projected on this call**. Only
there, only in the third arm, the current paid head MASK proposals can pass
the model's 0.5 threshold instead of 0.9. Next-block threshold stays 0.9.

Stop if any physical proof or paired control/input check fails. Pilot screen
for each whole algorithm (span, covered): no new nonconvergence; no tier
loses two correct on nine; compared to live, at least three tiers gain
macro TPF; and B or 2B gains TPF without losing accuracy. Do not piece
together methods at different tiers. If both pass, choose by number of
tiers matching live on both, then by the smaller relative TPF at B and
2B (maximize the worse ratio); final ties prefer the simpler span.

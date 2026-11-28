# Gemma Live Q32 A3: last physical tick, recorder gate

Preregistered fixed indices `0,10,20`, no replacements. Physical A800 A3 only,
public `dense_stop_joint_verified_eos`, B256/Q32 for the entire request, cap
2048, stock right-prefix share 1/8, frozen inputs/seeds/checkpoint/scorer. Bare
has only the existing physical accounting hooks. No-op inserts a narrow
proposal and state recorder; its exact event tape, delivered IDs and physical
position, encoder and cost ledger must equal bare. No sampling rule changes.

The current call must be ordinary sparse sampling on a built, resident head.
The recipient must be currently unresolved, resident, SC-bearing, clean,
omitted, not the frontier, and sampled while unresolved on the immediately
preceding physical ordinary sample in the same head/encoder epoch. Require
a real ordinary multinomial proposal (not a special token), rejected by the
actual entropy decision and neither admitted nor written. Record cold/warm
provenance of that prior sample separately. A drain breaks adjacency.
Tie by lower absolute position. The donor is the last eligible actually
sampled clean MASK in stock order; protect the frontier, rightmost stock
selected row, the entire prefix and debt. Keep current versus counterfactual
Q, sampled count and visible key addresses equal. The highest query address
itself is recorded separately. At most one counted exchange per head.
An accepted/revoked or missing proposal is not counted as a rejection.

These predicates use the field definitions of `fe9fbd9`, not the audit's
placeholder strict reducer: current/counterfactual shapes are compared, not
adjacent-tick widths, and pre-call MASK membership controls eligibility.
Gemma has no literal MASK ID for unresolved state. Historical Q32/Q64 tapes
do not carry actual prior per-row proposals; old Q32 three-anchor lifetime
failure rank and Q64 projection-age rank are different controls.

Zero legal exposures on the fixed three is a no-go, not a claim about all 90.
Only a legal exposure permits a separate preregistration of `7/13/25/33/55/71`
for same-A3 stock/no-op/treat, one exchange per head. No 90-case extension
without review. All raw records stay on n164; only token-free metrics
and source can be committed.

## Geometry preflight

`geometry.py` gives a concrete witness for the stock `choose` planner,
including an unchanged rightmost query, with a legally possible dirty-state
transition. It is not a reached model state or a measured exposure. In stock
`SelectedRowSampler._consume`, rejection draws a new random input ID, which
normally creates mandatory debt. But `torch.randint` does not exclude the old
ID, so equality can leave a rejected row clean. The case must be measured,
not inferred from old tapes or a universal zero claim.

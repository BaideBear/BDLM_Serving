# LLaDA: one selection rule at five Q budgets

**Decision on the nine screening cases: expand `covered`, not `span`. Neither replaces [Live Profile v1](../../../../../docs/live-profile-v1.md).** The `covered` arm improves TPF at every offer but loses one correct response at three offers, and loses token efficiency at Q8, Q16 and Q32. This is not evidence of a quality win.

| Fixed Q | Live v1 | Common span | Common span + covered-head acceptance |
|---:|---|---|---|
| 4 | 6 · 1.526 · 40.42% | 5 · 1.599 · 41.52% | 5 · 1.598 · 41.50% |
| 8 | 6 · 2.438 · 33.78% | 5 · 2.571 · 26.85% | 5 · 2.529 · 26.64% |
| 16 | 6 · 3.310 · 28.13% | 6 · 4.025 · 21.87% | 6 · 3.925 · 23.78% |
| 32 | 6 · 5.547 · 19.94% | 6 · 5.034 · 14.50% | 5 · 7.355 · 15.47% |
| 64 | 6 · 7.238 · 13.45% | 6 · 5.476 · 10.22% | 6 · 8.017 · 16.45% |

Cells: **correct out of 9 · macro TPF · pooled token efficiency**. Within each offer the same board executed all three arms on the same nine inputs, in rotating arm order. The cases are identical across offers and distinct from the preceding risk-ribbon pilot.

Both candidates use the same Q selection: pay outstanding writes, reserve at most one eighth of remaining seats for clean prefix before the active span, and allow the causal successor when the established head has no more than one quarter MASK left. `covered` differs from `span` only at admission: **this actual forward**, not the cache as a whole, must cover all valid head rows before the head's paid MASK proposals may use 0.5 instead of 0.9. Comparing either to live changes more than just acceptance.

**Confirmation protocol:** run the same selected candidate and Live v1 on the other 81 frozen requests at every offer. Alternate arm order and use the pilot board at each offer. Report the 81 not used for this selection separately from the combined 90; the 81 are not an unseen benchmark for earlier research. Use the original30 and new60 splits on the complete cohort. Incomplete tiers are not frozen90 results. Do not select different arms by offer.

[Registration](registration.md) · [Audited screen and decision](pilot.json) · [Physical selection](../../coverage_span.py) · [Receipt audit and confirmation runner](../../coverage_span_report.py).

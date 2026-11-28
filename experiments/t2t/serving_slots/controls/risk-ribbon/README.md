# LLaDA: replacing tier-specific extra admissions with a common confidence rule

**Decision: do not expand.** All 45 same-board request pairs had a different paid token write, but the same replacement improved TPF at only two of the five offers. The registered screen required three. No new model profile was selected or published from this pilot.

Each cell is **correct / 9 · macro TPF · pooled token efficiency**. Every offer used the same nine frozen indices; 45 pairs do not mean 45 different questions.

| Q | Live Profile v1 | Common acceptance rule |
|---:|---|---|
| 4 | 8 · 1.519 · 38.41% | 8 · 1.614 · 42.44% |
| 8 | 8 · 2.321 · 31.68% | 8 · 2.304 · 31.10% |
| 16 | 9 · 3.427 · 28.41% | 9 · 3.646 · 26.36% |
| 32 | 8 · 5.997 · 18.87% | 8 · 5.535 · 16.13% |
| 64 | 6 · 7.715 · 9.92% | 7 · 7.709 · 9.54% |

The candidate kept each tier's existing row planner. Only the extra MASK admission rule changed: traverse measured proposals left-to-right, spend at most 0.35 in cumulative `1 - confidence` on extra writes. Thus Q16 made more progress per forward but **less** per selected Q. These data do not establish that improving one metric improves the other.

[Preregistered test](registration.md) · [Compact receipts, subgroup counts and gate outcome](nine-pairs.json) · [Frozen control](../../../../../docs/live-profile-v1.md).

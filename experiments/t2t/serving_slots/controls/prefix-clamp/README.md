# Does a paid generated prefix change the decision on another row?

**Yes, on some observed states. Not a demonstrated quality or TPF gain.**
Frozen, fixed-budget GPU diagnostics on 2026-11-28. All raw GPU records are on n164. No production request used the scratch logits, and no six-question observation is a 90-question score.

## DiffusionGemma, Q64

Six registered questions (`7,13,25,33,55,71`). All first qualified at tick 2 of block 0. The six stock/observer/probe runs reproduced the same final IDs, score, RNG and physical bill per question. Their production totals were 88 forwards and 5,194 decoder rows for **each** arm.

The [first comparison](gemma-q64-six.json) projected the full *visible* interval (70–108 rows, not B256) on both sides and clamped all other omitted rows. The untreated wide control already changed admissions relative to the real call on 3/6 requests. We therefore did not treat it as a measurement at the native query width.

In the [registered native-width replication](native-width.md), keep the actual 64 queries and replace only per-layer K/V from the omitted, paid, generated prefix. Verify that the unmodified copy is bitwise-identical to the real selected MASK logits on all six requests. No prompt, future block, new visible key or selection threshold is changed.

| Index | Prefix rows substituted | Measured MASK | Top-1 changes | Newly accepted | No longer accepted |
|---:|---:|---:|---:|---|---|
| 7 | 6 | 45 | 3 | 31 | 54 |
| 13 | 9 | 49 | 0 | none | none |
| 25 | 40 | 40 | 0 | none | none |
| 33 | 9 | 17 | 1 | none | none |
| 55 | 9 | 37 | 13 | none | 39, 44 |
| 71 | 44 | 44 | 7 | 94 | 95, 101 |
| **Total** | **117** | **232** | **24** | **2** | **5** |

These are row decisions, not extra correct answers. Even the direction is not a simple increase in acceptances. Creating the counterfactual prefix bank and verifying it cost an additional **24 F / 1,653 decoder Q**, excluded from all production metrics. This is not a way to get those K/V for free. The [compact receipt](gemma-native-q64-six.json) contains per-input bills and hashes; [reducer](summarize.py). Raw directories under `n164:/data1/project_data/gx-erqw-20261123/results/`: `gemma-prefix-clamp-q64-a0-13ef960` is the invalid hook-width attempt, retained but excluded; `gemma-prefix-clamp-q64-a0-34272ee` is the wide control; `gemma-prefix-clamp-q64-a0-ad487d5` is the native-width replication.

## LLaDA 2.1, Q16

The H4 panel independently registered six different indices (`8,18,28,34,54,74`); do not pool model-level effects. Six matching states had 37 omitted paid generated prefix K/V rows which changed under refresh. At equal scratch width and fixed legal K, **2 of 60** original MASK rows changed their native admission, both accepted → rejected and both previously resident. A separate sparse-width replay matched all original logits. The equal-width resident control itself differed from live on **one** admission, so unlike Gemma's final experiment this is a scratch-width comparison. Production `N/F/Q = 1,680/627/8,492`; extra diagnostic `24 F / 439 Q`. [Result](llada-q16-six.json); raw: `llada-q16-generated-prefix-shadow-h4-six-v1` on n164.

## Separate cross-block execution test

At LLaDA Q64 (2B), compact Flow W2 opened and sampled the successor normally but omitted its clean, already-paid non-MASK rows on later calls, preserving resident K/V. In **this variant** the sampler could still consult the last measured logits for omitted rows. That makes it unsuitable as evidence for a fresh-logits-only algorithm. On its registered six, stock → treatment: **6/6 → 6/6**, macro TPF **11.937 → 11.772**, pooled TE **21.94% → 21.96%**, actual F **121 → 128**, Q **5,952 → 5,925**. Three requests actually omitted rows; two final outputs changed. [Receipt](llada-q64-successor-receipt.json); raw: `llada-flow-successor-bank-h0-six-v1`. Do not extend this stale-logit variant.

## Next production test

The [shared selector](../../../../../bdlm_serving/methods/t2t/prefix_seat.py) exchanges one optional, resident unresolved MASK for one paid, generated prefix row at most once per physical block. It protects all mandatory writebacks, the first unresolved position and the right edge of the visible interval. Only a row measured this tick may supply a new proposal. This test measures the **trade of one MASK row for one prefix row**; it does not identify a pure KV effect. The scratch tests above motivate the question; they do not determine the answer or an optimal prefix fraction.

# Gemma Q64 paid IBS: same-state fresh-visible-KV counterfactual (A0, six cases)

Read-only diagnostic on A800 physical 0. Fixed `Q=64` paid IBS, six screening
indices `7,13,25,33,55,71`. Each request runs `stock`, `observer` and `probe`;
the probe adds exactly one same-state scratch and must reproduce the observer
bit for bit.

## Result: calibration at full coverage, not a stale-K finding

All six requests reach the registered state at **block 0, tick 1** (the first
non-first ordinary sampling step). At that tick the planner selects the whole
visible interval (`selected = visible_width = 64`), so the paid re-projection
refreshes every visible K/V. Hence `unrefreshed_warm = 0` and the resident sparse
output is **bit-identical** to a fresh visible-KV recompute:

| # | paid D | paid F | paid Q | N | correct | sampled→unresolved | selected/visible | unrefreshed | max abs logit Δ | top1 / p≥.9 / unmask flips | fallback |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7  | 16 | 16 | 961  | 198 | ✓ | 55 | 64/64 | 0 | 0.0 | 0 / 0 / 0 | equal |
| 13 | 19 | 20 | 1154 | 260 | ✓ | 51 | 64/64 | 0 | 0.0 | 0 / 0 / 0 | equal |
| 25 | 13 | 13 | 757  | 222 | ✓ | 16 | 64/64 | 0 | 0.0 | 0 / 0 / 0 | equal |
| 33 | 14 | 14 | 829  | 229 | ✓ | 51 | 64/64 | 0 | 0.0 | 0 / 0 / 0 | equal |
| 55 | 18 | 18 | 1108 | 248 | ✓ | 52 | 64/64 | 0 | 0.0 | 0 / 0 / 0 | equal |
| 71 |  7 |  7 | 385  |  99 | ✗ | 12 | 64/64 | 0 | 0.0 | 0 / 0 / 0 | equal |

Paid totals: F=88, Q=5194, D=87. Extra diagnostic (excluded from the bill):
F=12, Q=768 (two fresh forwards per case at width 64).

Reading: at the first non-first step that re-samples an already-sampled
unresolved MASK, the paid path re-projects the **entire** visible interval, so
there is no residual stale K and the sparse output is exactly the fresh output
(top-1, `p≥.9`, native unmask and the guaranteed fallback all match; max abs
logit difference 0.0 over the compared rows). This state therefore cannot be the
source of KV-freshness error. A tick where the interval is only **partially**
re-projected would be needed to probe stale-K residual K; that was not run here
to stay within the fixed single-state control.

## Controls and release

`stock == observer == probe` held on all six (full delivered IDs, trace,
decoder/encoder events, all 85 Q/K/V projection widths, F/Q/D bill, post-sampler
RNG). `#71` is an ordinary Q64 miss for all three arms; correctness is not the
diagnostic. A0 was released `rc=0`, 0 MiB, no compute apps, locks closed
(2026-11-28T21:08:42+08:00). No holder; no 90-case extension; no result files on
the A800 persistent disk.

Raw receipts (log + 6 case JSON + summary), SHA-256
`6900f0d0a8fbe452842b4138211a8aa9602984c6fbd4bed15eeea2908b76a178` for the log:

```
n164:/data1/project_data/gx-erqw-20261123/results/gemma-q64-a0-freshkv-20261128T130316Z/
```

[Registration](registration.md) · [Compact receipt](receipt.json)

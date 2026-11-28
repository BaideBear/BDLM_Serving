# A0 Gemma Q64 same-state fresh-KV diagnostic — registration

Fixed before the A800 A0 launch on 2026-11-28 (UTC+8). This is a read-only
counterfactual, not a treatment and not a production algorithm.

## Fixed setup

- Device: A800 physical 0, `GPU-d3c3fc9c-b15d-9776-9094-7d3b85b705c1` (host `n1`).
  No holder; single real job; release on exit.
- Model: `diffusiongemma-cd40195` (bf16, SDPA, `grouped_mm`), transformers 5.12.0.
- Decoder budget: fixed `Q=64`, certified paid-EOS + paid-agreement planner
  (`Stop("paid", 64)`, offers `(64,)`). Identical to the Q64 paid-IBS path.
- Six fixed screening indices: `7, 13, 25, 33, 55, 71`. No 90-case extension.
- Seed: the frozen task seed per request (`task["seed"]`); three arms per
  request in alternating order.

## Arms

- `stock`: certified paid-EOS IBS only.
- `observer`: repeat agreement ledger observed, never applied (no-change repeat).
- `probe`: observer semantics plus exactly one fresh-visible-KV scratch at the
  registered state. The scratch is read-only and must reproduce `observer`.

Whole-request `stock == observer == probe` is required on the full delivered ID
list, decoder/encoder events, per-layer Q/K/V projection widths, trace, physical
bill (F/Q/D) and post-sampler RNG. Extra diagnostic forwards/rows are counted
separately and excluded from the paid bill.

## Registered state

In the **first physical block** (block 0), at the **first non-first ordinary
sampling step** (a tick with unresolved rows, later than the first sampling
tick) whose selection contains a row that was **already sampled at an earlier
tick and is still an unresolved MASK**. Fire once per request.

At that tick, with the same real canvas IDs, the actual self-conditioning
(`sc_bank`, masked to pre-tick initialized positions), the original encoder
cache and absolute positions, re-project the exact legal visible KV interval
`[start, end)` with `bank.bypass=True` (fresh, no `extend`/`visited` writes) and
compare the same selected-unresolved rows' top-1, `p>=.9` crossing and native
unmask decision (including the guaranteed fallback row) against the resident
sparse output.

## Boundaries recorded up front

- Same width only: the fresh attention width equals the resident visible width
  `end-start` (64 here); there is no artificial B=256 expansion.
- For selected rows the self-conditioning is exact; for visible rows not
  re-selected this tick the stored `sc_bank` is the only available value and is
  flagged, not claimed exact.
- A warm row's stale K also feeds every other query, so any disagreement is not
  by itself proof of pure cache error. Conversely, agreement at **full coverage**
  is a calibration (no stale K remains), not evidence about stale-K packs.
- Not word-level quality evidence; six screening cases only.

## Source pins (staged on A800 from independent worktree)

```
freshkv    bf377c1d05435579f2c51780800045caf23bf78229add9a1a07aeb914c82398b
adapter    f0907b171e330c74c4a2a4692be999b3ffd02e16e2c3e4ff35a577e7ec0ba567
launcher   79cb35e2fdc29c33f0cbda06e764e4c9f4afc0217bee44f5ffbdb2d352b83869
eos        2f8ee643f5d29b35f35238dfcfadff16d4503a7730fcae991e6fafd17b37a65a
agreement  11cabca063e902178c5e947447bde5cb5662162762dce0e41df34ea8c6fd6aaf
eval       f88351d84aa40a99b2b44a0ce1d7afc94dddae506fa6a03ea5b211a61375d0ff
runtime    a09d04477361cb2a260dac85ce2195c0bfee9218306b7cfb96d7ddf4f1dba9af
```

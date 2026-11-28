# Gemma Q128 provisional recheck: negative six-case result

**Do not expand to 90.** All three arms were 6/6. The correction had no score
rescue, left total generation forwards unchanged, and lowered macro TPF.
The small reduction in decoder rows is not a latency or quality improvement.

| A3 arm | Correct | Output N | Decoder F | Encoder F | Total F | Decoder Q | Macro TPF | Pooled TE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original repeat | 6/6 | 1,375 | 77 | 2 | 79 | 8,252 | 18.034 | 16.66% |
| Repeat + no-op observer | 6/6 | 1,375 | 77 | 2 | 79 | 8,252 | 18.034 | 16.66% |
| Repeat + one-shot correction | 6/6 | 1,378 | 76 | 3 | 79 | 8,238 | 17.836 | 16.73% |

Prompt prefill is excluded from F; generation encoder calls count toward F.
The correction added three real replacement-row payments, billed within
the following Q128 and Q20 decoder calls. The *net* F/Q columns include all
ensuing route and length changes and are not the price of three isolated rows.

## Physical Payment Evidence

All three corrections were on ordinary scheduling ticks, not terminal
suffix-only ticks. Positions and ticks are zero-based; widths are full
physical call widths, not just the corrected-row count.

| Case | Position | Extra write tick | First payment tick / Q | Current top p | Replacement payment tick / Q |
|---|---:|---:|---:|---:|---:|
| #13 | 127 | 3 | 4 / 128 | 0.993600 | 5 / 128 |
| #13 | 133 | 3 | 4 / 128 | 0.992268 | 5 / 128 |
| #55 | 236 | 9 | 10 / 40 | 0.911295 | 11 / 20 |

On each first divergent tick, the three arms had the same physical position
order, input IDs, sampler decisions including repeat admissions, sampled
output/noise IDs, and pre/post RNG. Every replacement ID was then seen in
the actual decoder input at its second payment. The independent replay
checked debt against the frozen frontier, encoder inputs, delivered IDs,
and the physical bill (all 85 Q/K/V hooks; no decoder over Q128).
#13 changed delivered IDs, gained six output tokens, and incurred two net
forwards including one encoder, with 226 more decoder rows. #55 changed
delivered IDs, emitted three fewer output tokens, and used two fewer
forwards and 240 fewer decoder rows. The other four requests were identical
across all arms. The implementation handles terminal replacement debt with a
bounded suffix-only repayment; a state-machine test covers that branch, but
this six-case GPU run did not exercise a correction on that branch.

The original repeat matched every available archived apply field and the
observer matched all archived first-payment records, including all 25 extra
first payments. The older compact apply receipts do not carry complete IDs;
full ID/tape equality is between the newly run repeat and observer. This is
an added opt-in T2T update for repeat extras, not native Gemma IBS.

## Reproduction and Release

[Registration](registration.md) | [Compact receipt](receipt.json) |
[Opt-in module](../../../gemma_provisional_recheck.py) | [GPU runner](run.py) |
[Offline receipt audit](summarize.py)

Original observer source: `5fced00090128fbcddde10cba1dd951058be6213`.
The GPU source overlay was `/gs/work/gemma-provisional-recheck-a3-v2-20261128`;
it used the pinned A3 `rl` environment (torch `2.6.0+cu124`, transformers
`5.12.0`), bf16 SDPA and grouped_mm. The attempted v1 launch stopped at its
environment check before any inference (wrong expected torch version);
its distinct receipt is retained and was not included.

Raw receipts, including all three full output ID lists and stage records,
are on n164 at
`/data1/project_data/gx-erqw-20261123/results/gemma-provisional-recheck-a3-v2-20261128/raw.log`
(SHA-256 `2d5bd2b2b16e2c9b396e4b368c848e05efe7b540c7137f65c7af460b8b3dd162`).
The receipt records source hashes at launch, reference hashes, and model and
runtime pins. Reproduce the compact read of these receipts on n164 with:

```sh
base=/data1/project_data/gx-erqw-20261123/results
python3 summarize.py "$base/gemma-provisional-recheck-a3-v2-20261128/raw.log" \
  "$base/gemma-payment-observer-a3-v1" "$base/gemma-paid-agreement-q128-01"
```

Physical A3 only, UUID `GPU-94c61e79-d80b-cae2-d85f-6bd2a2c99a1c`, worker
`807369`, launcher `807339`. The release receipt at 20:37:26 UTC+8 on
November 28, 2026 had `rc=0`, 0 MiB and no processes; independent readback
confirmed the lock was free and private scratch was removed. No holder.

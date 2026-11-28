# LLaDA Q8 partial-successor fallback

Fixed B32, Q8, cap 2048, indices 7/13/25/33/55/71. The intervention
reuses the sealed fused single-forward projection and native selector.
For the partial successor only, when no paid non-MASK proposal meets the
native confidence threshold, remove just the native forced top-one
selection if it would write a non-MASK token. When any ordinary proposal
qualifies, preserve the complete native selection. EOS retains the native
non-MASK treatment and the original stopping rule.

The observer records the exact gathered softmax confidence fed to the pinned
selector, in hexadecimal, and separates admission from an actual promoted
write. All measured successor scores were float32; the native comparison is
`score >= float32(0.9)`, threshold `0x1.cccccc0000000p-1`. Reproducing the
sealed fused receipt and full original/observer equality were execution gates.

## Result

| Arm | Correct | N | F | Q | Macro TPF | Pooled TE |
|---|---:|---:|---:|---:|---:|---:|
| Sealed strict stock, rescored | 6/6 | 1474 | 593 | 4412 | 2.53548 | 33.41% |
| Original fused / observer | 5/6 | 1418 | 645 | 4941 | 2.53598 | 28.70% |
| Filtered fused | 6/6 | 1436 | 595 | 4532 | 2.55101 | 31.69% |

In the original fused path, 37 of 43 joint calls had a natural non-MASK
qualifier; the other six made a material forced write on cases 33 and 55.
The gate qualified on case 33, zero-based event tick 44: absolute row
192 would write at confidence `0x1.6e2b140000000p-1`
(`0.7151724100112915`). The input, joint projection, head result and
promotion matched; only the successor write and its next-round debt
were removed. The filtered paths suppressed four material writes total,
counted on their own trajectories.

Case 55 was the only answer rescue and went from 261 to 183 F. Case 33
remained correct and went from 89 to 117 F. The four other requests replayed
unchanged. The factor is not degenerate on this cohort, but the filtered
arm still used 2 more F and 120 more Q than strict stock. The six-pair
result does not justify promotion; no 90-case job was run.

## Receipts

The fresh n164 attempt is
`/data1/project_data/gx-erqw-20261123/results/llada21-ibs-q8-tail-fallback-a1-v1/attempt-20261128T115455Z`.
A1 process `761617` verified the prior source registration and checkpoint
shards, then completed all 18 arms with `runner.exit=0`. The final A1 probe
found no worker, compute PID, GPU memory use, or attempt scratch directory.
`receipts.ndjson` holds the exact confidence tapes and generation receipts.
The initial export scored the cases but failed while reading the untyped
preflight record; `summary-rescore-v2.json` is the successful read-only
rescore of those same receipts, without rerunning inference or overwriting
the initial output. `six-summary.json` is the local compact copy. The old
tail-overlap attempt was read but not modified or archived here.

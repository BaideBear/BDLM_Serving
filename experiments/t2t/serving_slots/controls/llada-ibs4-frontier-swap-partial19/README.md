# LLaDA hard-capped IBS4: negative partial result, not a full90

Independent archive on n164: `/data1/project_data/gx-erqw-20261123/results/llada-ibs4-frontier-swap-a2-partial19-v1`.
Only `attempt-02`, physical A800 index 2, frozen registration `72291b781e64f88a380dff67eca52dd72536bd08c9d59515626b6612d5d00e69`. Per-request batch 1, B32/Q4, cap 2048; native T2T, sampler, termination and cache unchanged. Full bare / no-op / swap triples, same task, prompt, seed and scoring. `attempt-02/registration.json` contains all frozen hashes and the exact rule. The six fixed pilot indices 0,10,20,30,50,70 passed the preregistered physical/parity gate; quality was not in that gate. A resource guard stopped the remaining stream after 13/84. Its combined message does not identify the predicate. No `remaining.ok`, no full90 or resumption. Missing requests are not scored as failures. The old four-case stream is excluded.

| Observed / frozen | Correct bare → swap | Outputs with different IDs | Delivered N | Charged generation F | Charged generation Q |
|---|---:|---:|---:|---:|---:|
| All 19/90 | 18/19 → 17/19 | 18/19 | 7047 → 7869 | 5062 → 5925 | 19461 → 22796 |
| Original 16/30 | 16/16 → 15/16 | 15/16 | 5308 → 5781 | 3849 → 4277 | 14803 → 16455 |
| New 3/60 | 2/3 → 2/3 | 3/3 | 1739 → 2088 | 1213 → 1648 | 4658 → 6341 |

Only #20 flipped correct to incorrect; none gained. Both arms produced EOS on 19/19, no progress failure, recovery or induced failure. No A2 bare failure was available for a recovery-rate estimate. The historical risk strata have coverage 0/6 and 19/84. Full records bare/no-op equal 19/19; A2 bare/H800 historical events equal 0/19. Only A2 bare/swap differences are intervention contrasts. `N` excludes special tokens; `F` and `Q` exclude prompt prefill. Macro TPF averages the 19 values of N/F; pooled TE is total N / total Q (not task-average).

- `pairs.csv`: 19 paired rows, first differing output offset (zero-based, blank for identical sequences), per-arm output digests and same-request work/quality deltas. No raw answer or token sequence.
- `cohorts.json`: separate observed and frozen denominators, exact numerator/denominator for pooled TE, counts, deltas, macro TPF, historical risk coverage. No pooling across devices.
- `attempt-02/`: exact packets, metrics, gate, protocol, logs and diagnostic reports; `tools/`: the actual six postprocessing/worker/launch source files. No PEM files or other credentials. Original runtime/dependency digests are recorded in the registration.
- `SHA256SUMS`: exact relative name and SHA-256 for **every other file** in the server archive. Verify on n164 with `sha256sum -c SHA256SUMS`. `seal.py` is CPU-only. For report reproduction without writing a new seal, import `tools/ibs4_frontier_report.py` and call `report(Path("attempt-02"), diagnostic=True)`.

The repository handoff carries only this note, the archive builder, `pairs.csv`, `cohorts.json`, and `SHA256SUMS`. Raw packets and answers stay on n164.

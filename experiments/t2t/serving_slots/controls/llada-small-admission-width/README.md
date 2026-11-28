# LLaDA Live: width on first successor admission

H4 only, six frozen indices 7/13/25/33/55/71 (three original, three added).
Two fixed rolling offers, three arms per offer, all 36 fresh and scored on
H4. No other GPU, no 90-case extension. The rule truncates the contiguous
new successor rows to min(valid,32,2q) on the first physical visit, samples
those measured rows, and leaves the subsequent planner untouched. It does
not limit fully charged mandatory work in other rows. No exact previous
pair was found on this Live controller; H0 Q64 Flow first16 and strict
capped IBS Q8 tail overlap have different parent algorithms.

| q | Arm | Correct | N | F | Decoder Q | Peak | Macro TPF | Pooled TE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | strict | 6/6 | 1353 | 897 | 3380 | 4 | 1.5425 | 40.03% |
| 4 | full | 6/6 | 1412 | 787 | 4254 | 32 | 1.9354 | 33.19% |
| 4 | 2q | 5/6 | 1340 | 836 | 3326 | 8 | 1.6351 | 40.29% |
| 8 | strict | 6/6 | 1344 | 585 | 3995 | 8 | 2.3394 | 33.64% |
| 8 | full | 6/6 | 1310 | 475 | 4191 | 32 | 3.0879 | 31.26% |
| 8 | 2q | 6/6 | 1387 | 547 | 4020 | 16 | 2.7117 | 34.50% |

Compared with full, 2q reduced total Q at both offers but increased F and
reduced macro TPF. Q4 lost #7 (no rescues); Q8 had no correctness flips.
Q8 sits between strict and full on mean TPF, but costs 25 more rows than
strict on the six and is not hard-Q8. No recommendation to replace full
or claim preserved population accuracy from six cases.

All 12 triplets had the first divergence on admission, on matching
parent-selected rows and common input tokens after identical observed
history. Strict/full/2q widths at #7: 4/32/8 at Q4 tick 20; 8/32/16 at
Q8 tick 15. Q8 #7 alone shows why the first-call difference is not a
request-level saving: 600 to 651 total Q, 72 to 89 F. The independent
audit checked actual decoder rows in all 20 layers, debt payments,
write identities, IDs reconstructed from writes, empty terminal debt
and the per-request seal. TPF excludes actual prefill; TE pools final
nonspecial N over measured history-reading decoder Q (no encoder).

Frozen historical inputs match. Historical exact final IDs vs fresh H4:
Q4 strict 1/6, full 1/6; Q8 strict 0/6, full 2/6. The historical arms
are descriptive references, not substitutable for the same-H4 causal
pair. Their paths and bills are included with each row in `compact.json`.

Source and raw tapes:
`n164:/data1/project_data/gx-erqw-20261123/results/llada-live-first-2q-h4-six-v1/`;
one `q{4,8}-{ix:02}-{strict,full,limit}.ndjson` per request. The
`registration.json`, `source.sha256`, `complete.json`, `handoff.ndjson`,
`worker.ndjson`, compact report and code are in the same directory.
The `.preparation-only` sibling did not touch the holder and is not a
result source. Guard 3831652 and worker 3832239 exited; sole returned
H4 holder 3866431, start ticks 2043341829.

# A3 Gemma Live Q32 last-miss gate: no-go

All six requests (bare and recorder no-op on each of `0/10/20`) completed
on A800 physical A3. The arms were exactly equal on complete output IDs,
events, physical positions, encoder inputs and paid work on all three.
Only observation hooks ran. No treatment was registered or launched.

| Index | Warm ordinary sampling calls | Same-head adjacent ordinary samples | Donor available | Prior-tick ordinary rejected rows | Still clean | Legal exchanges |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 33 | 31 | 28 | 257 | 0 | 0 |
| 10 | 29 | 27 | 25 | 204 | 0 | 0 |
| 20 | 10 | 9 | 9 | 6 | 0 | 0 |
| Total | 72 | 67 | 62 | 467 | 0 | 0 |

Each counted rejected row had a real non-special multinomial proposal,
was unresolved before the preceding physical sample and at the current
sample, and was rejected by the actual entropy decision. None remained
clean after the previous call's writes. Thus no recipient passed even the
cleanliness gate, independent of whether a donor could be spared. The
rightmost stock row, frontier, prefix and writeback rows were protected.

This is a measured no-go on the preregistered anchors, not a proof about
all requests. The earlier `fe9fbd9` tapes cannot provide these fields, and
the committed planner witness is compatible with a clean rejection in
principle. Do not conflate this probe with the Q128 post-drain prefix
exchange, the Q64 projection-age study or the Q32 lifetime failure count.

[Preregistration](registration.md), [geometry witness](geometry.py),
[recorder](recorder.py), [verifier](report.py), [compact receipt](receipt.json).
Raw source and receipts are on n164 at
`/data1/project_data/gx-erqw-20261123/results/gemma-q32-lastmiss-a3-g6O5gxK5`.
Run `python3 -B report.py RUN_DIR` there to reproduce the receipt without
a GPU. The independent A3 check at 22:09:30 UTC+8 recorded 0 MiB,
0% utilization, no compute PID, no scratch directory and a free lock.

# Gemma Live Q128: one prefix-to-MASK seat after a warm drain

**Decision: keep the result as a six-case trade-off; do not expand or enable.**
Five requests changed output IDs. The treatment saved 17 generation forwards
and 1,695 actual decoder rows, but lost a correct answer on #25 and rescued none.

All 18 requests completed on A800 physical A3. Bare and no-op matched the
sealed A3 Live controls on every request, including the event tape, measured
positions, encoder inputs, output IDs, and bills. The parent is
`dense_stop_joint_verified_eos`, not paid IBS. This direction gives a prefix
seat to a MASK, not a MASK seat to a prefix. The five exchanges occurred
after an unchanged pure writeback forward. None removed a writeback forward
at its point of intervention.

| Arm | Correct | N | Decoder F | Encoder F | Total F | Decoder Q | Macro TPF | Pooled TE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bare | 6/6 | 1,420 | 117 | 3 | 120 | 12,451 | 12.160 | 11.40% |
| No-op | 6/6 | 1,420 | 117 | 3 | 120 | 12,451 | 12.160 | 11.40% |
| Treatment | 5/6 | 1,363 | 102 | 1 | 103 | 10,756 | 13.763 | 12.67% |

| Index | First exchange, block-local | Total F (no-op to treatment) | Decoder Q (no-op to treatment) | Correct |
|---:|---|---:|---:|---|
| 7 | 0 to 131 | 17 to 16 | 1,826 to 1,750 | yes to yes |
| 13 | 0 to 131 | 25 to 16 | 2,707 to 1,851 | yes to yes |
| 25 | 0 to 132 | 18 to 20 | 1,619 to 1,715 | yes to no |
| 33 | 0 to 133 | 21 to 14 | 2,215 to 1,597 | yes to yes |
| 55 | 0 to 131 | 32 to 30 | 3,227 to 2,986 | yes to yes |
| 71 | unexposed, identical | 7 to 7 | 857 to 857 | yes to yes |

All five exchanges first forked at zero-based decoder tick 4, head 0,
with exactly 128 physical rows before and after. The encoder epoch,
visible key addresses, and frontier were unchanged; the rightmost selected
query advanced by one position. The independent audit reconstructed
eligibility, acceptance and dirty ledgers, verified physical positions and
the 85 projector hooks, and confirmed the unexposed triplet was identical.
Whole-request work savings reflect changed generation, including two fewer
generation encoder calls. They are not a measured wall-clock gain or a
quality result beyond these six prompts.

## Reproduction and artifacts

[Registration](registration.md), [adapter](seat.py), [runner](run.py),
[independent audit](report.py), [compact receipt](receipt.json).
Raw output, source manifest, reference, and release check are on n164 at
`/data1/project_data/gx-erqw-20261123/results/gemma-q128-post-drain-a3-sWAyEkki`.
The original A3 reference is the sibling `gemma-live-q128-a3-frozen90-7iDh2EqM`.
On n164, run `python3 -B report.py RUN_DIR ORIGINAL_A3_RUN_DIR` to reproduce
the compact receipt without a GPU. The final independent release check
found 0 MiB, 0% utilization, no compute PID, and a free lock; no holder
was created.

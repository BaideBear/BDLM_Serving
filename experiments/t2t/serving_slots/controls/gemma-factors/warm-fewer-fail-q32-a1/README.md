# Q32 warm selection: stop at the three-anchor gate

**No-go on physical exposure, not a frozen90 performance result.** Fixed
single-request Q32, verified-EOS baseline, first tile sampled normally and
right-edge 1/8. Count unsuccessful real samples separately for each row of
each head. Protect dirty rows, prefix and earliest unresolved row. On a
full-width warm sample, exchange only the highest-index remaining selected
unresolved row, and only if the best unselected unresolved row has a
strictly lower count. The exact tie rule was frozen in `registration.md`.

| Anchor | Warm sampled calls | Contested full-width calls | Different physical row sets | Total F (both) | Selected decoder Q (both) |
|---:|---:|---:|---:|---:|---:|
| 0 | 33 | 29 | 0 | 79 | 2,407 |
| 10 | 29 | 27 | 0 | 76 | 2,316 |
| 20 | 10 | 9 | 0 | 35 | 1,099 |
| Total | 72 | 65 | 0 | 190 | 5,822 |

Counts use one reference trajectory per pair, not both arms as independent
evidence. Each contest had a legal row to give up and another to take, but
the latter never strictly beat the registered donor. No claim that all
counts tied: the receipts distinguish strict wins from non-wins, not
ties from worse alternatives. The pinned runner observed physical positions
and widths at 85 projections, checked the dirty ledger, clock, encoder
inputs, and ordered verification. All controls exactly matched historical
Q32 verified-EOS-on receipts. Treatment and control matched in full
(including all step hashes, delivered IDs, score and bills); both scored
3/3. There was no executed swap and thus no on-hardware evidence about the
quality or cost of an actual swap. Source-level invariants and synthetic tests
are not a substitute for an exposed GPU pair.

Only 0/10/20 of original30, zero of new60. The other 87 cases were **not run**.
No quality or speed estimate for them; do not extrapolate anchor identity.
Do not expand, weaken the tie rule, or change the profile on this evidence.

A800 GPU1 only, model PID 356967. Empty-board checks before loading, no holder.
The worker returned normally after `checkpoint-3.json` and `no-go.json`;
at 12:51:04 +08 on November 28, 2026 the PID was absent and physical A1
showed 0 MiB and no compute PIDs. There is no successor.

Raw: `/gs/results/warm-fewer-fail-q32-a1-1128`. Runtime and exact launcher:
`/gs/runtime/warm-fewer-fail-q32-a1-1128/launch.sh`. Three no-model tests
passed locally and on A800. Compact `anchors.csv` and `provenance.json`
contain no raw outputs. To re-audit existing receipts without any inference:

```sh
rt=/gs/runtime/warm-fewer-fail-q32-a1-1128
CUDA_VISIBLE_DEVICES= PYTHONPATH="$rt:/gs/runtime/cold-frontier-last-a1-1128:/gs/runtime/verified-eos-1128/repo" \
  /usr/bin/python3 -B -m experiments.t2t.serving_slots.gemma_warm_fewer_fail_report
```

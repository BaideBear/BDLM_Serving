# LLaDA B32 geometry-v1: native same-state results

CPU replay of the verified LLaDA 2.1-mini frozen30 traces.
[Fixed rule and evaluation plan](geometry-preregister.md) |
[Aggregate CSV](geometry-native.csv)

For Q4/Q8/Q16 there were 1,170 selected-stage decisions, with 191 strict
non-fresh targets. The other 974 steps were full width and contained 120
additional strict non-fresh targets. Across all 2,144 steps there were
754 strict edits (443 fresh, 311 non-fresh).

| Q / allocation | Reserved seats | Reserved hits: geometry / frontier / uniform expectation | Selected-stage old hits incl. backfill: geometry / zero reservation | Changed query sets vs zero | MASK rows lost vs zero |
|---|---:|---:|---:|---:|---:|
| 4 / slack 1 | 56 | 2 / 2 / 0.65 | 2 / 0 | 56 | 0 |
| 8 / slack 1 | 190 | 10 / 6 / 2.16 | 10 / 0 | 185 | 0 |
| 16 / slack 2 | 949 | 34 / 34 / 12.06 | 36 / 7 | 472 | 0 |
| 16 / fixed 2 | 1,688 | 54 / 46 / 21.20 | 56 / 7 | 837 | 739 |
| 32 / dense | 0 | n/a | n/a | 0 | 0 |

Q32 bypasses the selector and covers 311/311 old targets and 754/754 strict
targets. For slack-only arms, the smaller geometry hit delta over the
existing frontier (0, +4, 0 on reserved rows) matters more than the lift
over uniform or zero reservation when deciding whether to replace frontier
ordering. Q4 has only two hits. Fixed Q16 is a negative control: more old
hits do not establish a favorable tradeoff when MASK coverage falls.

The two hit columns answer different questions: a reserved-hit count is
conditional on that allocation's given seats; the selected-stage count
includes shared positional backfill and uses all 191 selected-stage old
targets as its denominator. Within each Q all arms see the same starting
query clocks (from zero reservation). Writes, fresh and strict are
teacher-forced from native decisions, including writes at rows a sparse
policy would not have queried. This is not a sparse rollout, a quality or
TPF measurement, or an independent blind test. The rule was not fitted on
the cohort and was not changed after replay.

## Integration contract

Call `geometry.rank_old(edit_pool, editable, resolved, fresh, last_write,
last_query, clock, previous_strict)` at the old-candidate ordering point.
Use block-local positions, the pre-forward clock and previous-step observed
strict edits. The result is a newly sorted list of the supplied pool;
fresh overflow, MASK priority, reserve limits, and backfill remain the
caller's responsibility. Select the arm once per run, not per request.

## Reproduction

Run at repo root with a new CSV path. Only the Python standard library and
the repository CPU selection code are used.

```sh
CUDA_VISIBLE_DEVICES= python -m tools.geometry_probe \
  --raw "$VERIFIED_LLADA_TRACE_DIR" --out "$NEW_CSV"
python -m unittest discover -s tests -p test_geometry.py
```

The run used `runtime/t2t-native-1125/llada21` and checked every trace's
frozen-manifest identity, frame continuity, and native edit totals.

## Online geometry-v1 comparison

Run this separate CPU report from the repository root after publishing the
online tables. It reads published CSVs; it does not score model outputs,
load weights, or run inference.

```sh
CUDA_VISIBLE_DEVICES= python -m tools.geometry_comparison \
  --geometry-dir experiments/t2t/placement/geometry-online \
  --adaptive-dir experiments/t2t/adaptive \
  --hard-dir experiments/t2t/hard \
  --placebo-dir experiments/t2t/placement \
  --out experiments/t2t/placement/geometry-online
python -m unittest discover -s tests -p 'test_geometry*.py'
```

All paths above are defaults. Point each `--*-dir` at the directory containing the published
`llada21-online-requests.csv` and `llada21-online-summary.csv`, not the raw
run directory. Q8 `hard_slack_placebo` was launched in `geometry-v1-llada21`
and reads from `--geometry-dir`; Q4 and Q16 placebo read from `--placebo-dir`.
The report writes only `arms.csv` and `paired.csv`, even when reporting
into the geometry scoring directory. The optional `llada21-audit.csv` provides strict hits among
actually reserved rows. `llada21-paired-cost.csv` is a fallback when that
policy has no audit row; its work totals must match the requests. An
unverified audit, missing data, and a verified zero are distinct.

The 12 comparisons for the launched arms are:

| Geometry arm | Q / edit cap | Same-Q references |
|---|---|---|
| `hard_slack_geometry` | 4 / 1, 8 / 1, 16 / 2 | `hard_slack`, `hard_slack_placebo`, `hard_noedit` |
| `hard_geometry` | 16 / 2 | `hard_frontier`, `hard_random`, `hard_noedit` |

Q8 has a slack geometry/placebo comparison; fixed Q8 was not run and is
not included as a missing arm. Fixed Q16 is the MASK-displacing negative control.
`hard_noedit` always has zero edit seats. The fixed and slack allocations
are not interchangeable; no similar-but-different quota substitutes for
an unavailable arm.

[Arm inventory](geometry-online/arms.csv) gives both available figures and
missing inputs. [Paired comparisons](geometry-online/paired.csv) joins by
`(index, dataset, task_id)` against the frozen30 manifest and requires all
30 requests and a matching `all` summary on each side. Wins/losses are
geometry-only/reference-only correct. The three metrics are accuracy,
mean per-request `N/F`, and `sum(N)/sum(Q)`; realized `Q/F` is
`sum(Q)/sum(F)`. Work includes all charged generation stages. Missing or
partial arms keep their comparison row with empty paired results; missing
reservation evidence leaves only those counters empty. Published tables
do not carry per-request seed or same-state evidence. This compares
realized online trajectories on the reused cohort; equal nominal Q does
not imply equal physical work or identical runtime states.

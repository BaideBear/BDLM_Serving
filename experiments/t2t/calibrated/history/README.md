# Does four-round edit history improve LLaDA old-row selection?

**No under the frozen rule. Keep the positional hazard.** History gained at
none of the three physical budgets and lost at all three. The rule required at
least two gains and no loss. No feature, window, penalty, or budget was retuned.

| Evaluation | Seats per arm | Position hits | Position + history | Difference [paired 95% interval] | Requests better / worse |
|---|---:|---:|---:|---:|---:|
| Q4/1 reserved | 452 | 9 | 8 | -1 [-4, 2] | 1 / 2 |
| Q8/1 reserved | 710 | 23 | 18 | -5 [-13, 2] | 2 / 4 |
| Q16/2 reserved | 1,688 | 56 | 46 | -10 [-19, -1] | 1 / 7 |
| Top 1 | 880 | 28 | 23 | -5 [-13, 2] | 2 / 4 |
| Top 4 | 3,437 | 88 | 82 | -6 [-12, 0] | 1 / 5 |
| Top 8 | 6,533 | 145 | 137 | -8 [-17, 0] | 1 / 6 |

Each row has the same 191 target events across frozen30. A hit is a strict
change from one non-MASK token to another on an already-resolved, non-fresh,
editable row. Top-k ranks the whole old pool. Physical counts are hits on
actually available **reserved** seats after mandatory fresh and first MASK;
they do not credit incidental backfill. Q16/2 had seven requests lose hits,
one gain, and 22 tie. Its loss is not just a lower shortlist score.

Both fits used only the disjoint 24 calibration traces (8 per dataset), the
same 25 positional features, intercept and L2 penalty. The history arm appended
the five fixed counts/indicators in the [protocol](PROTOCOL.md). Prior edits
and writes come from completed native rounds; reset at each block and exclude
prompt positions. No current label, logits, KV, or lexical information enters
ranking. We checked request IDs, source indices and prompt hashes for overlap;
the positional fit reproduces the archived coefficients to the shared tolerance.
Intervals resample paired whole requests 10,000 times within dataset, conditional
on the fitted weights. These are native teacher-forced events, not sparse
rollout, latency, or quality results. Frozen30 is not a new test population.

**Implementation cost.** At B32 a packed implementation using byte counters
needs 32 bytes of saturating own-edit counts plus 32 bytes for four rounds of
strict/write bitmasks, per active block, plus ring bookkeeping. The ever bit
comes from the counter. Neighbour masks are shared constants. A ranked row
adds at most 12 popcounts and five linear-score terms, with no extra Q or model
calls. On n164 CPU core 23 the Python history update and feature extraction took
7.28 microseconds per native denoise round (median of 21 calibration24 passes
after one warm-up, process CPU time, 413 blocks / 2,306 rounds / 28,054 old-row
feature evaluations). This replay included block resets and every denoise phase;
inputs were predecoded. It excludes base ranking, scoring, I/O, and device
transfer. The packed size is not the size of Python containers.

[Per-request counts](requests.csv) | [Totals, precision and recall](summary.csv) |
[Paired differences](paired.csv) | [Coefficients](coefficients.json)

Run at the n164 repo root:

```sh
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
../venv/bin/python -B -m experiments.t2t.calibrated.history.study \
  --cal ../t2t-native-1125/calibration24-v1-llada21 \
  --native ../t2t-native-1125/llada21 --out experiments/t2t/calibrated/history
../venv/bin/python -B -m pytest -q -p no:cacheprovider \
  experiments/t2t/calibrated/history/test_history.py tests/test_lexical_hazard.py
```

The history and lexical arms share one fit/measurement loop. Rerunning the
original lexical CLI to a temporary directory reproduced its archived
`requests.csv`, `summary.csv`, and `coefficients.json` byte-for-byte.

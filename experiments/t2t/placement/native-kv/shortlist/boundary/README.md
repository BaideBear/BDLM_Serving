# Gemma L15: do unresolved neighbors help place slack seats?

**No sparse rollout recommended.** Adding two prestate distances to the ten
position + prior-write features missed the [frozen gate](PROTOCOL.md): four
fewer total high-drift hits than the stronger comparator at each of Q16, Q32
and Q64. Q128 gained seven, but was descriptive. No features, ridge penalty,
folds or decision thresholds were changed after observing the results.

Held-out totals include reserved **and** neutral backfill hits. Each budget
replays the same 30 original requests (five stratified whole-request folds),
325 eligible native steps, 40,350 old-row instances and 4,182 high-drift
targets. Budgets are alternative selections, not independent samples to sum.

| Q | Zero reserve | Position + write | Frontier | With distances (reserve + backfill) | Margin vs stronger | Old swaps vs zero | Hits gained / lost vs zero | MASKs displaced |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 5 | 7 | 10 | 6 (2 + 4) | -4 | 58 | 2 / 1 | 0 |
| 32 | 30 | 40 | 37 | 36 (19 + 17) | -4 | 210 | 19 / 13 | 0 |
| 64 | 148 | 199 | 190 | 195 (83 + 112) | -4 | 715 | 83 / 36 | 0 |
| 128 | 863 | 936 | 916 | 943 (333 + 610) | +7 | 2,380 | 329 / 249 | 0 |
| 256 | 4,182 | 4,182 | 4,182 | 4,182 (0 + 4,182) | 0 | 0 | 0 / 0 | 0 |

The last three columns describe the distance-augmented ranker. A swap is one
newly selected old row replacing an old row selected by zero reserve, not two
operations. Q256 selects every row without a reserve. [budget.csv](budget.csv)
contains the same counters for all three rankers, including physical seats on
eligible steps only. The replay checks both comparators' counts against the
[archived summary](../summary-budget.csv) overall, per dataset and per fold.

The distances need only the acceptance mask before the step. Current KV,
logits and drift labels are not predictors. This measures placement against
native L15 labels on a fixed native trajectory, not the effect of a sparse
refresh on later states, output quality or serving throughput.

From the repository root with the verified diagnostics available locally:

```sh
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
../venv/bin/python -B -m tools.drift_shortlist --boundary --raw ../t2t-native-1125/kv-drift-v1-gemma
../venv/bin/python -B -m pytest -p no:cacheprovider -q tests/test_drift_shortlist.py
```

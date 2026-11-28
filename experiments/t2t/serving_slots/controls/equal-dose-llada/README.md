# Q16 paid IBS: repeat identity versus current confidence

H4 generated all 18 requests on the registered six with their original seeds.
Stock and repeat reproduced the six sealed H0 pilot rows **and tapes** exactly
on H4. No H0 formal case was used as an H4 measurement. No offline dose replay,
panel extension, or change to a shared module. Six cases, not a quality estimate.

| H4 arm | Correct | Output N | Decoder F | Selected Q | Extra accepts | Paid-close F/Q | Macro N/F | Pooled N/Q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original paid IBS | 6/6 | 1379 | 399 | 5229 | 0 | 1/6 | 3.862 | .2637 |
| Repeat agreement | 6/6 | 1384 | 371 | 5010 | 152 | 1/1 | 4.156 | .2762 |
| Equal-count current confidence | 6/6 | 1574 | 387 | 5098 | 144 | 3/21 | 4.452 | .3087 |

Each comparison was on one physical H4, Q16, 2048 cap, using the same native
proposals, selected-only T2T, dirty writeback, scorer, and paid frozen-EOS
stop. Every history-reading decoder call counts, including suffix debt payment;
prompt prefill is separate. The audit reconciled all 20 projected widths,
every owed token's next input ID, and each applied extra's observed proposal.

On a common state the two experimental arms use the same genuine ledger for
`k`; only the identities of the `k` extra accepts can change. Current ranks the
current measured native-rejected MASKs with nonspecial proposal and `prob > .5`
by descending confidence, ties by ascending absolute position. After divergence,
each arm recomputes `k` on its own state. Totals are not fixed across trajectories.

| Case | First tick / k | Repeat positions | Current positions | Current minus repeat N/F/Q |
|---|---|---|---|---|
| 7 | 24 / 1 | 138 | 148 | -9 / +1 / +21 |
| 13 | 7 / 3 | 82,83,84 | 86,87,90 | +86 / +22 / +256 |
| 25 | no difference | | | 0 / 0 / 0 |
| 33 | 2 / 6 | 82,83,84,85,86,89 | 82,85,86,89,90,92 | +13 / -16 / -193 |
| 55 | 2 / 2 | 51,54 | 56,58 | +100 / +9 / +4 |
| 71 | no difference | | | 0 / 0 / 0 |

All four first forks had identical incoming rows, token IDs, native proposals,
measured probabilities, native acceptances, edits, and debt. Both no-fork cases
matched to termination. In the original three, current minus repeat is **+23F,
+277Q, +77N**; in the additional three it is **-7F, -189Q, +113N**. Current
has higher macro TPF overall, but did not match repeat's absolute bill. This
panel neither proves that repeat identity is required for quality nor supports
promoting current confidence as a cost saving over repeat.

Source and compact evidence are in this directory. Raw receipts remain only
on n164 in `/data1/project_data/gx-erqw-20261123/results/llada-ibs-q16-equal-dose-h4-six-v1`.
To verify without inference, run the following there:

```bash
set -euo pipefail
T=/data1/project_data/gx-erqw-20261123/runtime
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$T/llada-ibs-q16-paid-agreement-frozen90-h0-v1:$T/fixed-baseline-deps"
sha256sum -c source.sha256
sha256sum -c report.verified
"$T/venv/bin/python" -B study.py report | cmp - compact.ndjson
"$T/venv/bin/python" -B paired.py | cmp - paired.csv
```

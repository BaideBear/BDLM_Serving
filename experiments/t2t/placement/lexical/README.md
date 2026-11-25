# Does the old token's lexical class improve LLaDA edit prediction?

**Decision: retain the positional hazard.** The lexical class improved all three
unconstrained shortlists slightly, but not reserved-seat coverage across budgets.
At Q16/2 it lost one hit on each of three different requests and gained on none.
No tuning, rollout or Gemma work followed this comparison.

Both models were fitted from scratch on the same calibration24 (8 per dataset,
169 positive old rows out of 16,092). The original frozen30 requests share no
source row or prompt digest with calibration. The verified native evaluation
has 1,170 selectable rounds, 880 with candidates, 14,875 candidate row events
and 191 residual strict edits. Fresh, bootstrap and full verification edits are
not targets. All 30 requests are retained, including those with zero hits.

Each cell gives **hits; precision, recall (%)**. Precision uses the seats
actually selected under that rule. Recall always uses the 191 residual edits.
The last column counts requests with more / equal / fewer hits under lexical.

| Rule | Seats | Position | Position + lexical | More / equal / fewer |
|---|---:|---:|---:|---:|
| Top 1 | 880 | 28; 3.18, 14.66 | 30; 3.41, 15.71 | 3 / 26 / 1 |
| Top 4 | 3,437 | 88; 2.56, 46.07 | 89; 2.59, 46.60 | 3 / 25 / 2 |
| Top 8 | 6,533 | 145; 2.22, 75.92 | 148; 2.27, 77.49 | 2 / 28 / 0 |
| Q4 / 1 | 452 | 9; 1.99, 4.71 | 9; 1.99, 4.71 | 1 / 28 / 1 |
| Q8 / 1 | 710 | 23; 3.24, 12.04 | 24; 3.38, 12.57 | 2 / 27 / 1 |
| Q16 / 2 | 1,688 | 56; 3.32, 29.32 | 53; 3.14, 27.75 | 0 / 27 / 3 |

These are same-native-history comparisons, not estimated answer quality, TPF
or sparse rollout behavior. Shortlists ignore the physical layout; budget rows
count only actual reserved seats after mandatory fresh and protected MASK, not
neutral backfill. The baseline reproduced the six archived calibrated hit/seat
counts exactly (`../../calibrated/native-shortlist.csv` and `native-replay.csv`
in that same directory), not the separate five-fold frozen30 baseline.

[Frozen feature and decision rule](PROTOCOL.md) | [Summary](summary.csv) |
[Paired request counts](requests.csv) | [Fitted coefficients](coefficients.json)

CPU reproduction from the n164 repository root:

```sh
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
../venv/bin/python -B -m tools.lexical_hazard \
  --cal ../t2t-native-1125/calibration24-v1-llada21 \
  --native ../t2t-native-1125/llada21 \
  --tokenizer /data0/ysy/models/LLaDA2.1-mini/tokenizer.json \
  --out experiments/t2t/placement/lexical
../venv/bin/python -B -m unittest discover -s tests -p test_lexical_hazard.py -v
```

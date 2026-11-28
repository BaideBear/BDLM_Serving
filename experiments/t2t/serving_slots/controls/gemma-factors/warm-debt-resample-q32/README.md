# Gemma Q32 warm resolved-debt resample single factor

Control: verified-EOS frozen90 `early=True` at fixed Q32. Treatment: on the
first warm pure all-debt drain of a physical head (head fully initialized), the
same projected drain also samples its measured **resolved** dirty rows through
the original native sampler. It does not sample unresolved MASK rows (that is
the paid-mask factor), does not add a row/Q/clean prefix, and fires at most
once per physical head. The Q64/Q32 public profiles are untouched.

Same-device paired exact control reproduction against the verified-EOS frozen90
references (`/gs/results/verified-eos-1128`): all IDs, output text, bills,
seeds, route counts, encoder digest and tape matched before the treatment was
read. Control/treatment order alternated per request. Pilot 0/10/20 had real
first-difference exposure on every anchor; the run continued to 90.

| Cohort | Requests | Control correct | Treatment correct | Control F | Treatment F | Control decoder Q | Treatment decoder Q | Control macro TPF | Treatment macro TPF | Control pooled TE | Treatment pooled TE | Warm-debt F | Warm F | New admits | New edits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pilot3 | 3 | 3 | 3 | 190 | 191 | 5822 | 5772 | 5.323 | 5.224 | 0.1628 | 0.1611 | 5 | 45 | 0 | 2 |
| original30 | 30 | 29 | 28 | 2247 | 2227 | 68412 | 68296 | 4.832 | 4.894 | 0.1534 | 0.1550 | 56 | 514 | 0 | 20 |
| new60 | 60 | 56 | 55 | 4627 | 4728 | 141277 | 144505 | 4.902 | 4.934 | 0.1577 | 0.1596 | 116 | 1062 | 0 | 38 |
| all90 | 90 | 85 | 83 | 6874 | 6955 | 209689 | 212801 | 4.879 | 4.921 | 0.1563 | 0.1581 | 172 | 1576 | 0 | 58 |

`Warm-debt F` is the number of `warm_debt_resample` forwards; `Warm F` is the
number of ordinary control `drain` forwards. The treatment made **0 admits**
and **58 edits** on the retry calls (all90); it changes already-resolved tokens,
not MASK fills. Macro TPF is `mean(N_i/F_i)`; pooled TE is `ΣN/Σdecoder_Q`.

Conclusion: **mixed, not a win**. Correctness falls 85→83/90 (pilot 3→3,
original30 29→28, new60 56→55), while F rises 1.2%, decoder Q rises 1.5%,
macro TPF rises 0.85% and pooled TE rises 1.14%. The 58 resolved-token edits do
not pay for the extra clock/forwards. Do not enable this factor at Q32.

Paths:
- run: `/gs/results/gemma-warm-debt-resample-q32-1128` (90 pairs + summary;
  source data stays on the cluster)
- archive: this folder (`summary.json`, `summary.csv`, `pairs.csv`,
  `manifest.json`); the 90 raw pair JSON files stay on `/gs` and were removed
  locally after their SHA-256 matched the cluster copies
- code: `experiments/t2t/serving_slots/gemma_warm_debt_resample.py`,
  `gemma_warm_debt_resample_eval.py`, `tools/launch_gemma_warm_debt_resample.sh`
- tests: `tests/test_gemma_warm_debt_resample.py`

The compact `pairs.csv` keeps per-pair values and hashes (output IDs, tape,
encoder, pair SHA-256) but no raw traces or output IDs; it is not a quality
claim for mixed budgets.

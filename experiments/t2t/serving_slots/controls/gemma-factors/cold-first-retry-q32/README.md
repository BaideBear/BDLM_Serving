# Gemma Q32 cold-first-retry single factor

Control: verified-EOS frozen90 `early=True` at fixed Q32. Treatment: on the
first pure all-debt drain of a physical head, before full initialization, the
same projected drain also samples its measured unresolved dirty rows through
the original native sampler. No extra row, no extra Q, no clean prefix, no
head dirty addition; one actual sampler tick instead of zero; at most one retry
per physical head. The Q64/Q32 public profiles are untouched.

Same-device paired exact control reproduction against the verified-EOS frozen90
references (`/gs/results/verified-eos-1128`): all IDs, output text, bills,
seeds, route counts, encoder digest and tape matched before the treatment was
read. Control/treatment order alternated per request. Pilot 0/10/20 had real
first-difference exposure on every anchor; the run then continued to 90.

| Cohort | Requests | Control correct | Treatment correct | Control F | Treatment F | Control decoder Q | Treatment decoder Q | Control macro TPF | Treatment macro TPF | Control pooled TE | Treatment pooled TE | Cold F | Warm F | New admits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pilot3 | 3 | 3 | 2 | 190 | 178 | 5822 | 5462 | 5.323 | 5.744 | 0.1628 | 0.1781 | 5 | 45 | 54 |
| original30 | 30 | 29 | 26 | 2247 | 2501 | 68412 | 77108 | 4.832 | 4.899 | 0.1534 | 0.1476 | 60 | 514 | 520 |
| new60 | 60 | 56 | 50 | 4627 | 4572 | 141277 | 138412 | 4.902 | 5.151 | 0.1577 | 0.1642 | 113 | 1062 | 1122 |
| all90 | 90 | 85 | 76 | 6874 | 7073 | 209689 | 215520 | 4.879 | 5.067 | 0.1563 | 0.1583 | 173 | 1576 | 1642 |

`Cold F` is the number of `cold_first_retry` forwards; `Warm F` is the number
of ordinary control `drain` forwards; `New admits` is the number of admitted
rows on those retry calls. Macro TPF is `mean(N_i/F_i)`; pooled TE is
`ΣN/Σdecoder_Q` over all actually read history decoder Q. `N_per_history_Q` in
the raw summary is not the main metric for this project.

Conclusion: **mixed/negative at Q32**. Correctness falls 85→76/90 (pilot 3→2,
original30 29→26, new60 56→50), while macro TPF rises 4.879→5.067 and pooled
TE rises 0.1563→0.1583; F rises 2.9% and decoder Q rises 2.8%. The
correctness loss is large and the pilot already lost one of three anchors, so
do not enable this factor at Q32 on the strength of the TPF increase.

Paths:
- run: `/gs/results/gemma-cold-first-retry-q32-1128` (90 pairs + summary;
  source data stays on the cluster)
- archive: this folder (`summary.json`, `summary.csv`, `pairs.csv`,
  `manifest.json`); the 90 raw pair JSON files stay on `/gs` and were removed
  locally after their SHA-256 matched the cluster copies
- code: `experiments/t2t/serving_slots/gemma_cold_first_retry.py`,
  `gemma_cold_first_retry_eval.py`, `tools/launch_gemma_cold_first_retry.sh`
- tests: `tests/test_gemma_cold_first_retry.py`

The compact `pairs.csv` keeps per-pair values and hashes (output IDs, tape,
encoder, pair SHA-256) but no raw traces or output IDs; it is not a quality
claim for mixed budgets.

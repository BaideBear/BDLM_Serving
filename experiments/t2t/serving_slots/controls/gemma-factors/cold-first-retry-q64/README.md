# Gemma Q64 cold-first-retry single factor

Control: verified-EOS frozen90 `early=True` at fixed Q64. Treatment: on the
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
| pilot3 | 3 | 3 | 3 | 126 | 142 | 7046 | 8720 | 8.578 | 8.544 | 0.1365 | 0.1037 | 5 | 28 | 80 |
| original30 | 30 | 29 | 28 | 1314 | 1421 | 75975 | 83210 | 8.030 | 7.752 | 0.1362 | 0.1207 | 54 | 289 | 909 |
| new60 | 60 | 54 | 52 | 2635 | 2866 | 152672 | 163798 | 8.249 | 8.035 | 0.1413 | 0.1311 | 113 | 576 | 2265 |
| all90 | 90 | 83 | 80 | 3949 | 4287 | 228647 | 247008 | 8.176 | 7.940 | 0.1396 | 0.1276 | 167 | 865 | 3174 |

`Cold F` is the number of `cold_first_retry` forwards; `Warm F` is the number
of ordinary control `drain` forwards; `New admits` is the number of admitted
rows on those retry calls. Macro TPF is `mean(N_i/F_i)`; pooled TE is
`ΣN/Σdecoder_Q` over all actually read history decoder Q. `N_per_history_Q` in
the raw summary is not the main metric for this project.

Conclusion: **negative at Q64**. Correctness falls 83→80/90 (pilot 3→3,
original30 29→28, new60 54→52), macro TPF falls 8.176→7.940 and pooled TE
falls 0.1396→0.1276 while F rises 8.6% and decoder Q rises 8.0%. The extra
sampler tick changes later trajectories and does not recover the cost. Do not
enable this factor at Q64.

Paths:
- run: `/gs/results/gemma-cold-first-retry-q64-1128` (90 pairs + summary;
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

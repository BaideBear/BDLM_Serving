# LLaDA Q64: projected but unsampled unresolved head positions

Same 17 requests common to the four gate arms at the snapshot: indices 0–15
and 20. This partial cohort is **not** a quality or TPF comparison. All arms
project every valid head row at Q64. When only successor rows are cold, the
reference sampler can select those cold rows to the exclusion of already
projected, warm, non-dirty MASKs in the head.

| Successor gate | Affected requests | Affected forwards | Paid, unsampled warm head MASK rows |
|---|---:|---:|---:|
| Off | 0/17 | 0 | 0 |
| Immediate | 17/17 | 171 | 2,577 |
| Half | 17/17 | 204 | 2,519 |
| Quarter | 17/17 | 143 | 741 |

Counts are opportunities, not tokens saved or fixes to wrong answers. The
single-factor follow-up holds the row plan fixed at the first changed call
and tests whether also sampling exactly those head MASKs improves the final
trajectory. Aggregate CSV/JSON contain counts only.

## Single-factor warm-head sample: paired summary (Q64, half gate)

Inputs are cluster-side receipts only:

* pilot: `/gs/results/llada-hfull-single-factor-20261128-031321` (0, 10, 20)
* pinned remainder: `/gs/results/llada-warm-head-rest-pinned-1128` (87)
* excluded: `/gs/results/llada-hfull-single-factor-20261128-030853`
  (3 rows; index 10 failed before scoring because the first deployment used the
  base interpreter without `math_verify`; excluded, not replaced)

The two included runs share one protocol, checkpoint, samples SHA, Q64, half
gate, `coverage="head"`, and `old_source="run"`. The failed first run is
recorded as metadata only and is never loaded.

Common metrics: per-request TPF = `N/F` (in
`paired-warm-head-single-factor.csv`); **macro TPF = mean(N_i/F_i)** is the
primary token-rate readout; **token efficiency = pooled TE (`ΣN/ΣQ`)**.
Macro TE (`mean(N_i/Q_i)`) and pooled TPF (`ΣN/ΣF`) are different descriptive
statistics and do not replace the two primary metrics. `F` is the **self-developed executor's actual forward
count**, not an extra native-decoder measurement; `Q` is the physical decoder
row count under the **strict cap** (`q ≤ 64` and `layer_physical_q == [q]×20`
on every forward; max `Q/forward = 64`). Both arms are strict-cap on every
cohort.

| Cohort | Coverage | Old correct | New correct | Old F | New F | Old Q | New Q | ΔQ | Old macro TPF | New macro TPF | Old macro TE | New macro TE | Old pooled TPF | New pooled TPF | Old pooled TE | New pooled TE | Strict cap | Intervened | First checks (declared / independent) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| original30 | 30/30 | 19 | 21 | 5258 | 4211 | 280606 | 222430 | -58176 | 5.293 | 6.006 | 0.1051 | 0.1179 | 2.924 | 3.438 | 0.0548 | 0.0651 | yes | 30 | 30 / 30 |
| new60 | 60/60 | 42 | 41 | 7532 | 7754 | 389411 | 410258 | +20847 | 6.109 | 6.886 | 0.1288 | 0.1402 | 3.605 | 3.624 | 0.0697 | 0.0685 | yes | 58 | 58 / 58 |
| all90 | 90/90 | 61 | 62 | 12790 | 11965 | 670017 | 632688 | -37329 | 5.837 | 6.592 | 0.1209 | 0.1328 | 3.325 | 3.558 | 0.0635 | 0.0673 | yes | 88 | 88 / 88 |

All three cohorts are complete; there are no missing indices and no failures in
the included 90. The two non-intervened requests are 75 and 76; both arms are
correct with identical F and Q. Paired flips are 10 false→true and 9 true→false
overall (original30: +4 / -2; new60: +6 / -7). The first intervention has
identical pre-state and identical physical rows in 88/88 cases; the independent
trace re-check agrees with the declared check in 88/88.

This is a **paired single-factor trajectory cohort**, not a frozen90 quality or
TPF qualification. The coverage is complete, but the only contrast is
`warm_head_sample` at Q64 half gate; do not read the 90-case scores as a new
overall baseline.

## Gate single factor: half -> quarter (Q64, full-head)

Inputs:

* control 0..59: `/gs/results/llada-hfull-paired/rows`
  (`c64-full-half` vs `c64-full-quarter`, trusted same version)
* GPU5 60..89: `/gs/results/llada-hfull-gate-single-factor-20261128-050055`
* same-device 0/10/20 pilot:
  `/gs/results/llada-hfull-gate-single-factor-20261128-045813`

The 0/10/20 pilot matches the control on all 6 arm checks (output IDs, trace
ignoring `pre_state`, F, Q and correctness). `F` is the self-developed
executor's actual forward count; `Q` is physical rows under the strict cap
(`q ≤ 64`, `layer_physical_q == [q]×20`; max `Q/forward = 64`). Strict cap holds
on every cohort. Pre-state equality at tick 0 is verified on the 30 GPU5 rows
(60..89); the control 0..59 has no `pre_state`, and the 0/10/20 pilot is the
same-version cross-check.

| Cohort | Coverage | Old correct | New correct | Old F | New F | Old Q | New Q | Old macro TPF | New macro TPF | Old macro TE | New macro TE | Old pooled TPF | New pooled TPF | Old pooled TE | New pooled TE | Strict cap | Diverged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| original30 | 30/30 | 19 | 24 | 5258 | 3598 | 280606 | 163115 | 5.293 | 5.643 | 0.1051 | 0.1228 | 2.924 | 3.917 | 0.0548 | 0.0864 | yes | 27 |
| new60 | 60/60 | 42 | 46 | 7532 | 6422 | 389411 | 292229 | 6.109 | 6.596 | 0.1288 | 0.1469 | 3.605 | 4.288 | 0.0697 | 0.0942 | yes | 56 |
| all90 | 90/90 | 61 | 70 | 12790 | 10020 | 670017 | 455344 | 5.837 | 6.279 | 0.1209 | 0.1388 | 3.325 | 4.155 | 0.0635 | 0.0914 | yes | 83 |

All 90 are present with zero failures. The gate change flips 14 false→true and
5 true→false (net +9 correct), reduces executor F by 21.7% and physical Q by
32.0%, and raises macro TPF by 0.442 and macro TE by 0.0179. Seven of 90 did not
diverge at all (the gate never opened in either arm). This is a **gate
single-factor paired cohort**, not a frozen90 quality/TPF qualification.

### Where Q64 is spent

[row-bill.csv](row-bill.csv) partitions all physical decoder rows into old
write debt, first-visit MASK, warm MASK, clean block-local prefix, and all
other projected rows. The full-head projection includes clean rows regardless
of the nominal sparse prefix quota: a prefix visit here is not necessarily
an extra row purchased for that purpose. All traced calls had nonzero prefix
KV length. Each sampled edit is attributed once to the input row's role;
reversions to MASK are not T2T. Tail rows and sampled rows are overlays, not
additional costs. The two half-gate controls independently agree on the IDs,
scores and all role counts for every matched request.

| Gate and sampler | Mean physical Q/F | Head Q | Tail Q | Mandatory Q | Clean prefix Q | T2T on clean prefix / all T2T |
|---|---:|---:|---:|---:|---:|---:|
| Half, reference | 52.39 | 402,440 | 267,577 | 47,680 | 156,346 | 2,158 / 3,748 |
| Quarter, reference | 45.44 | 314,210 | 141,134 | 45,153 | 121,252 | 1,125 / 2,076 |
| Half, allow warm head MASK sample | 52.88 | 376,912 | 255,776 | 47,250 | 133,945 | 1,825 / 3,224 |

Reproduce on the receipt host using `tools/llada_gate_row_bill.py` with
`--control`, `--gate`, `--warm-pilot`, `--warm-rest` pointing respectively to
the original four-arm root, the gate complement and the two warm-head roots
above, and `--manifest benchmarks/frozen90/manifest.json`. Only cohort counts
are exported; these are whole trajectories, not a same-state counterfactual.

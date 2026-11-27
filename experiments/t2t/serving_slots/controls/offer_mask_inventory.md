# Offer/mask inventory (frozen90, CPU trace audit)

- Data root: `/data1/project_data/gx-erqw-20261123/runtime/t2t-native-1125/prefix90`
- Samples: `/data1/project_data/gx-erqw-20261123/runtime/t2t-native-1125/prefix90/samples.json` (sha256 `bc9e457ddebd1eb0062c495f13f31cfc86b18e3cebbd90c3da08674f7ab65aed`)
- Python: `3.12.13 (main, Aug  7 2026, 02:25:39) [Clang 22.1.3 ]`
- Scorer: bdlm_serving.evaluation.scoring.score
- Generated: 2026-11-26T23:58:53.513822+00:00

All N/F/Q are recomputed from per-request traces and checked against the
stored physical ledgers.  Zero visible opportunities means **not measured**,
not zero effect.

## Full90 arms

| model | kind | tier | mask | req | trig | opp ticks | dual | refresh | cached | F(all) | Q(all) | quality | macroTPF | micro eff | status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gemma | fixed | 32 | on | 90 | 0 | 0 | 0 | 0 | 0 | 4382 | 253672 | 80/90 | 8.308 | 13.64% | not_computable_no_visible_opportunity |
| gemma | fixed | 64 | on | 90 | 0 | 0 | 0 | 0 | 0 | 2775 | 248182 | 82/90 | 12.343 | 12.96% | not_computable_no_visible_opportunity |
| gemma | fixed | 128 | on | 90 | 0 | 0 | 0 | 0 | 0 | 1981 | 268587 | 83/90 | 17.469 | 12.00% | not_computable_no_visible_opportunity |
| gemma | variant | 128+k32 | on | 90 | 0 | 0 | 0 | 0 | 0 | 1945 | 281024 | 85/90 | 17.378 | 11.40% | not_computable_no_visible_opportunity |
| gemma | fixed | 256 | on | 90 | 0 | 0 | 0 | 0 | 0 | 1345 | 344320 | 85/90 | 26.365 | 9.12% | not_computable_no_visible_opportunity |
| gemma | fixed | 512 | on | 90 | 70 | 144 | 138 | 0 | 6 | 1347 | 380007 | 85/90 | 26.415 | 8.32% | computable_paired_intervention |
| gemma | control | 512_open | off | 90 | 70 | 226 | 217 | 0 | 9 | 1368 | 405026 | 86/90 | 26.073 | 7.72% | computable_paired_intervention |
| gemma | variant | 512+spawn50 | on | 90 | 89 | 334 | 291 | 0 | 43 | 1285 | 402369 | 84/90 | 27.083 | 7.84% | descriptive_only |
| gemma | switch | replay | on | 90 | 36 | 69 | 52 | 8 | 9 | 1804 | 333797 | 83/90 | 19.117 | 9.47% | computable_paired_intervention |
| gemma | control | replay_open | off | 90 | 36 | 138 | 77 | 23 | 38 | 1866 | 349552 | 84/90 | 18.751 | 9.09% | computable_paired_intervention |
| gemma | variant | replay+k32 | on | 90 | 36 | 66 | 52 | 7 | 7 | 1911 | 353136 | 82/90 | 19.130 | 9.22% | descriptive_only |
| gemma | switch | dwell | on | 90 | 40 | 75 | 58 | 6 | 11 | 1878 | 351906 | 84/90 | 19.296 | 9.11% | descriptive_only |
| gemma | variant | dwell+k32 | on | 90 | 39 | 70 | 56 | 4 | 10 | 1826 | 342259 | 83/90 | 19.253 | 9.17% | descriptive_only |
| llada21 | fixed | 4 | on | 90 | 0 | 0 | 0 | 0 | 0 | 17608 | 225136 | 74/90 | 3.600 | 20.17% | not_computable_no_visible_opportunity |
| llada21 | fixed | 8 | on | 90 | 0 | 0 | 0 | 0 | 0 | 11701 | 213127 | 74/90 | 4.405 | 19.61% | not_computable_no_visible_opportunity |
| llada21 | fixed | 16 | on | 90 | 0 | 0 | 0 | 0 | 0 | 10010 | 222550 | 76/90 | 4.888 | 18.02% | not_computable_no_visible_opportunity |
| llada21 | fixed | 32 | on | 90 | 0 | 0 | 0 | 0 | 0 | 9237 | 242141 | 74/90 | 5.261 | 17.05% | not_computable_no_visible_opportunity |
| llada21 | fixed | 64 | on | 90 | 0 | 0 | 0 | 0 | 0 | 9237 | 242141 | 74/90 | 5.261 | 17.05% | not_computable_no_visible_opportunity |
| llada21 | fixed | 64+progress80 | on | 90 | 90 | 2598 | 2598 | 0 | 0 | 7178 | 296168 | 73/90 | 6.561 | 13.79% | computable_paired_intervention |
| llada21 | control | 64+progress80_open | off | 90 | 90 | 2740 | 2740 | 0 | 0 | 8832 | 350243 | 72/90 | 5.932 | 11.76% | computable_paired_intervention |
| llada21 | switch | 8 | on | 90 | 0 | 0 | 0 | 0 | 0 | 11701 | 213127 | 74/90 | 4.405 | 19.61% | not_computable_no_visible_opportunity |
| llada21 | switch | 16 | on | 90 | 0 | 0 | 0 | 0 | 0 | 10010 | 222550 | 76/90 | 4.888 | 18.02% | not_computable_no_visible_opportunity |
| llada21 | switch | 8,8,16,16 | on | 90 | 0 | 0 | 0 | 0 | 0 | 11037 | 221900 | 75/90 | 4.559 | 18.46% | not_computable_no_visible_opportunity |
| llada21 | switch | 16,16,8,8 | on | 90 | 0 | 0 | 0 | 0 | 0 | 10306 | 211392 | 77/90 | 4.623 | 18.80% | not_computable_no_visible_opportunity |
| llada21 | reference | dual64 | on | 90 | 90 | 5118 | 4490 | 0 | 628 | 7960 | 395363 | 69/90 | 5.831 | 9.86% | descriptive_only |

## Paired mask-on/off contrasts

Only rows with visible opportunities and both arms are contrasts.  Rows
marked `not_computable_no_visible_opportunity` are **not** zero-effect
estimates.

| pair | cohort | status | exposed | quality on/off | resc/lost | macroTPF on/off (Δ) | eff on/off (Δ pp) | F on/off | Q on/off |
|---|---|---|---:|---|---:|---|---:|---:|---:|
| gemma-q512 | original30 | computable_paired_intervention | 24 | 29/30 | 0/1 | 25.473/24.671 (+0.802) | 7.87%/7.28% (+0.59) | 446/470 | 125952/137598 |
| gemma-q512 | new60 | computable_paired_intervention | 46 | 56/56 | 0/0 | 26.886/26.773 (+0.113) | 8.54%/7.95% (+0.59) | 901/898 | 254055/267428 |
| gemma-q512 | full90 | computable_paired_intervention | 70 | 85/86 | 0/1 | 26.415/26.073 (+0.342) | 8.32%/7.72% (+0.60) | 1347/1368 | 380007/405026 |
| gemma-replay | original30 | computable_paired_intervention | 11 | 29/29 | 0/0 | 18.436/17.816 (+0.621) | 9.17%/8.69% (+0.47) | 618/639 | 110454/116311 |
| gemma-replay | new60 | computable_paired_intervention | 25 | 54/55 | 0/1 | 19.457/19.219 (+0.238) | 9.62%/9.28% (+0.34) | 1186/1227 | 223343/233241 |
| gemma-replay | full90 | computable_paired_intervention | 36 | 83/84 | 0/1 | 19.117/18.751 (+0.366) | 9.47%/9.09% (+0.38) | 1804/1866 | 333797/349552 |
| llada-q64-progress80 | original30 | computable_paired_intervention | 30 | 25/24 | 3/2 | 6.303/5.961 (+0.342) | 13.07%/11.80% (+1.26) | 2562/2757 | 104681/110312 |
| llada-q64-progress80 | new60 | computable_paired_intervention | 60 | 48/48 | 6/6 | 6.690/5.918 (+0.772) | 14.19%/11.74% (+2.45) | 4616/6075 | 191487/239931 |
| llada-q64-progress80 | full90 | computable_paired_intervention | 90 | 73/72 | 9/8 | 6.561/5.932 (+0.629) | 13.79%/11.76% (+2.03) | 7178/8832 | 296168/350243 |

## Per-offer opportunities in switching paths

| model | arm | cohort | offer | steps | dec F | dec Q | opp ticks | dual | refresh | cached | trig req |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma | q128_slack_random | full90 | 128 | 1862 | 1862 | 259776 | 0 | 0 | 0 | 0 | 0 |
| gemma | q512_spawn50 | full90 | 512 | 1204 | 1204 | 381633 | 334 | 291 | 0 | 43 | 89 |
| gemma | replay | full90 | 32 | 393 | 393 | 24301 | 10 | 1 | 2 | 7 | 9 |
| gemma | replay | full90 | 64 | 213 | 213 | 19383 | 1 | 0 | 0 | 1 | 1 |
| gemma | replay | full90 | 128 | 225 | 225 | 27969 | 6 | 0 | 6 | 0 | 5 |
| gemma | replay | full90 | 256 | 206 | 206 | 52736 | 1 | 0 | 0 | 1 | 1 |
| gemma | replay | full90 | 512 | 688 | 688 | 189184 | 51 | 51 | 0 | 0 | 36 |
| gemma | replay_q128_prefix32 | full90 | 32 | 419 | 419 | 25710 | 9 | 1 | 3 | 5 | 9 |
| gemma | replay_q128_prefix32 | full90 | 64 | 225 | 225 | 20182 | 1 | 0 | 0 | 1 | 1 |
| gemma | replay_q128_prefix32 | full90 | 128 | 238 | 238 | 30764 | 4 | 0 | 4 | 0 | 4 |
| gemma | replay_q128_prefix32 | full90 | 256 | 218 | 218 | 55808 | 1 | 0 | 0 | 1 | 1 |
| gemma | replay_q128_prefix32 | full90 | 512 | 729 | 729 | 199680 | 51 | 51 | 0 | 0 | 36 |
| gemma | dwell | full90 | 32 | 366 | 366 | 27326 | 14 | 4 | 5 | 5 | 10 |
| gemma | dwell | full90 | 64 | 221 | 221 | 18207 | 2 | 0 | 0 | 2 | 2 |
| gemma | dwell | full90 | 128 | 284 | 284 | 34757 | 2 | 0 | 1 | 1 | 1 |
| gemma | dwell | full90 | 256 | 214 | 214 | 54784 | 1 | 0 | 0 | 1 | 1 |
| gemma | dwell | full90 | 512 | 711 | 711 | 195840 | 56 | 54 | 0 | 2 | 40 |
| gemma | dwell_q128_prefix32 | full90 | 32 | 357 | 357 | 27148 | 12 | 5 | 3 | 4 | 9 |
| gemma | dwell_q128_prefix32 | full90 | 64 | 215 | 215 | 17153 | 2 | 0 | 0 | 2 | 2 |
| gemma | dwell_q128_prefix32 | full90 | 128 | 278 | 278 | 35046 | 2 | 0 | 1 | 1 | 1 |
| gemma | dwell_q128_prefix32 | full90 | 256 | 208 | 208 | 53248 | 1 | 0 | 0 | 1 | 1 |
| gemma | dwell_q128_prefix32 | full90 | 512 | 689 | 689 | 189440 | 53 | 51 | 0 | 2 | 39 |
| llada21 | slottrace_8 | full90 | 8 | 11701 | 11701 | 213127 | 0 | 0 | 0 | 0 | 0 |
| llada21 | slottrace_16 | full90 | 16 | 10010 | 10010 | 222550 | 0 | 0 | 0 | 0 | 0 |
| llada21 | slottrace_8_8_16_16 | full90 | 8 | 7910 | 7910 | 178566 | 0 | 0 | 0 | 0 | 0 |
| llada21 | slottrace_8_8_16_16 | full90 | 16 | 3127 | 3127 | 43334 | 0 | 0 | 0 | 0 | 0 |
| llada21 | slottrace_16_16_8_8 | full90 | 8 | 2833 | 2833 | 24725 | 0 | 0 | 0 | 0 | 0 |
| llada21 | slottrace_16_16_8_8 | full90 | 16 | 7473 | 7473 | 186667 | 0 | 0 | 0 | 0 | 0 |
| llada21 | dual64_noedit | full90 | 64 | 7960 | 7960 | 395363 | 5118 | 4490 | 0 | 628 | 90 |

## Identity findings

- `llada21`: `llada-fixed-q16`, `llada-switch-16` — bit-identical delivered ids, N, F and Q on every request.
- `llada21`: `llada-fixed-q32`, `llada-fixed-q64` — bit-identical delivered ids, N, F and Q on every request.
- `llada21`: `llada-fixed-q8`, `llada-switch-8` — bit-identical delivered ids, N, F and Q on every request.

## Cross-checks against existing offer-work.csv

- 36/36 per-tier cells matched the published offer-work.csv.

## Caveats

- Zero visible opportunities means the cross-block contrast was not measured; it must not be reported as a zero difference.
- A single 256-row (Gemma) or 32-row (LLaDA) physical block cannot host a cross-block read, so Q32/Q64/Q128/Q256 fixed Gemma arms and Q4/Q8/Q16/Q32 fixed LLaDA arms are structurally outside the mask question.
- Gemma compact traces retain a persisted opportunity boolean but not the selected/visible sets; this audit verifies the boolean against physical work and first_opportunity, and cross-checks per-tier counts against existing offer-work.csv files when present.
- LLaDA opportunities are recomputed from absolute_positions and visible_kv, matching mask_pair_report.eligible; stored per-request opportunity counts are checked when present.
- Low-tier opportunities inside the replay path occur only after a 512-row exposure and are derived (cached/tail-refresh) opportunities, not independent per-tier mask interventions.

## Scope

- Included: prefix90 unified90 DiffusionGemma fixed/replay arms and LLaDA matched fixed/slots/dual64 arms.
- Excluded from this run: LLaDA 16<->32n raw traces under private-llada-native-slot90-v1 and the older Gemma slottrace roots.  They are separate data roots; append ArmSpec entries to audit them, and do not infer their cross-block effect from this inventory.

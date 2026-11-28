# Full successor admission: fixed-budget requests

## Freeze decision: incomplete

The candidate keeps the ordinary Live policy and changes one event: on the first projection of a successor, add every valid row in that block and let eligible rows sample immediately. It is a **fixed rolling offer**, not a hard per-call budget. This release has one audited candidate tier out of ten. Neither a same-device control nor a suffix branched from an earlier state counts as a new candidate request.

| LLaDA 2.1, fixed Q16 | Correct / 90 | Macro TPF | Token efficiency | Final N | Generation F | Decoder Q |
|---|---:|---:|---:|---:|---:|---:|
| Full successor admission | 73 | 3.642 | 20.35% | 42,581 | 15,350 | 209,234 |
| Optimus IBS, hard cap | 72 | 3.375 | 20.54% | 42,500 | 15,480 | 206,927 |
| Optimus IBS, nominal window | 73 | 3.371 | 20.51% | 44,079 | 16,152 | 214,873 |
| Previous strict Live | 75 | 3.356 | 25.60% | 44,943 | 17,583 | 175,591 |

The candidate gained **8** correct cases and lost **7** vs capped IBS. Original 30: 25 vs 23; new 60: 48 vs 49. Candidate minus capped IBS: +0.268 macro TPF, +1/90 correct, -0.19 percentage points TE. Descriptive paired 95% bootstrap intervals (4,000 resamples stratified by dataset) are [+0.042,+0.534] TPF, [-6.67,+10.00] percentage points accuracy and [-2.00,+1.45] percentage points TE. This is evidence of a TPF improvement on this panel, not proof of higher or equivalent quality. Against previous Live the tradeoff is +0.286 TPF, -2 correct, -5.24 points TE.

All 90 candidate outputs were reconstructed and re-scored on the receipt host: 85 EOS, 5 reached cap, 0 failures. Capped IBS request #75 failed in generation and remains in that baseline's 90-case denominator. Prompt IDs agree, but IBS prefills only complete blocks: in 88/90 cases it and the candidate bill different amounts as prefill. TPF excludes each executor's actual prefill. In 1,335 real admission forwards the candidate projected 42,657 new-block rows; 21,303 were above the nominal offers. These are included in total Q, not additional to it. Ordinary peak never exceeded 16; admission peak was 32. Compared with capped IBS the candidate used 130 fewer generation F and 2,307 more decoder Q. Cross-stack comparisons do not establish same-device latency.

[All 90 audited cases](llada-full-admission-q16-frozen90.json) · [Paired intervals](llada-q16-full-paired-intervals.json) · [Ten-tier matrix, including subgroup deltas](full-admission-status.json).

## What qualifies

LLaDA Q4/8/16/32/64 and Gemma Q32/64/128/256/512 must each finish the **same 90 prompts and seeds**, starting fresh at a fixed offer. The first 30 and additional 60 are reported separately from the same run. No sampled subsets, strict controls or branch suffixes can fill missing tiers. Do not skip failed model requests or substitute a different sample. A genuine model failure must be scored wrong and retain its actual work; an infrastructure interruption without a valid terminal receipt is incomplete. Any new emitting adapter needs an independent audit before its compact ledger is eligible.

`python -m experiments.t2t.serving_slots.fixed90_gate --require-ready` reads the registered full-cohort ledgers and the measured baselines, and exits nonzero if coverage or any required metric is missing. `ready_to_compare` certifies evidence, not an algorithm. `meets_performance` checks observed accuracy and macro TPF against **each** designated control at the same nominal offer; TE and actual peak are separate. A positive point estimate is not a significant win, an equivalence claim or a hard-cap result. For all regular budgets up to B the primary control is capped IBS; at B also include native and Flow W1, at 2B use Flow W2. Nominal IBS and previous Live are additional references. Do not switch to trajectory modeling based on one passing tier.

The matrix reports the actual Q bill alongside each comparison. `mean_decoder_Q` is pooled history-reading Q divided by decoder forwards (not all generation forwards). `extra_history_Q` is candidate minus baseline total history-reading Q over the same cases, **not** the sum of per-call excess above the offer. `both_observed_within_offer` is false if a measured peak exceeds the offer, true if both measured peaks fit, and null if the baseline peak is unavailable and the candidate fits. Even true does not imply equal total work.

## LLaDA 2.1, Q16: original 30

All three runs use constant rolling Q16, the same original 30 frozen prompts/seeds and the 2048 generation cap. Only the candidate permits a full successor's first projection to exceed the rolling offer. Comparisons with published controls describe the same tasks, not a same-runtime causal ablation. In particular, the previous H4 gate on #7 and the historical strict Live control on #7 have different trajectories. Don't mix the three-request response forks into any curve or denominator here.

| Original 30 | Correct | Macro TPF | Pooled token efficiency | Output N | Generation F | Decoder Q |
|---|---:|---:|---:|---:|---:|---:|
| Full admission, H4 | 25 | 3.436 | 20.37% | 14,285 | 5,093 | 70,135 |
| Strict Live | 25 | 3.183 | 23.00% | 15,466 | 6,836 | 67,253 |
| Hard-capped Optimus IBS | 23 | 3.189 | 19.32% | 13,620 | 5,278 | 70,484 |

Against strict Live, #14 becomes correct and #4 becomes incorrect; 22/30 requests use fewer generation forwards. The candidate spends 2,882 more actual Q overall while emitting 1,181 fewer final tokens. Against capped IBS, aggregate Q is 349 lower despite peaks of 32 instead of 16. Neither comparison establishes throughput at fixed wall time or a win under a per-forward hard cap. The original 30 are a predefined subset, not an independent test set or a freeze decision.

| Exclusive Q role (H4 candidate) | Actual rows | Share of decoder Q |
|---|---:|---:|
| Previous writes being repaid | 14,887 | 21.23% |
| First full projection of a successor MASK block | 14,321 | 20.42% |
| Other projected MASK positions | 32,338 | 46.11% |
| Other, already decoded positions | 8,589 | 12.25% |

448 admission calls used **7,153 rows above the rolling offers**. These are included in the 70,135 decoder Q; they must not be added a second time. The planner explicitly labelled 3,258 of the 8,589 clean decoded rows as optional prefix. Those rows were sampled on all 3,258 visits and edited on seven. Of all 102 non-MASK to different non-MASK edits, 66 occurred on mandatory writeback rows and 36 on clean decoded rows (including those seven). These are row **visits**, not unique positions. An unchanged token can still acquire different KV; seven edits are not evidence that refreshing the rest is useless.

[Candidate audit and per-request metrics](llada-full-admission-q16-original30.json) · [Frozen controls](metrics.csv) · [Row reconstruction](../../../../../tools/fixed_admission_rows.py).

On n164 the case seals and full physical tapes are under `results/llada21-live-fullblock-fixed90-v1/Q16` in the project data directory. Run `python3 tools/fixed_admission_rows.py RUN_DIR benchmarks/frozen90/manifest.json original30`; use `new60` or `frozen90` for subsequent cohorts. The program checks the raw SHA for each seal, per-layer projection sizes, observed token transitions, continuity of debt, exact full-admission rows, terminal and bill reconstruction. It only emits performance and role shares for complete cohorts. No raw model outputs or logits are archived in this branch.

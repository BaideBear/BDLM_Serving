# Two paid observations on the same unresolved row

[四档阶段归档：90 题同卡主表、带数值的图、六题消融与未测档说明](four-slot-checkpoint.md) (`paid-agreement-four-slot-20261128`).

[Paid Agreement v1: frozen results, figure and reproduction](v1.md) (`paid-agreement-v1`). Archived as an experimental version, not a default policy.

**After v1: [four measured points with numerical labels](fixed-budget90-with-q8.png).**
The [new LLaDA Q8 point](llada-q8/README.md) uses the unchanged admission
rule. On the same A1, paid IBS → agreement: **70→71/90, 2.450→2.555
macro TPF, 30.31%→29.95% pooled TE**. Delivered N rose 42,570→44,609,
F 19,054→20,278 and Q 140,426→148,924. Six cases were rescued and five
lost; original 30 scored 22→22, new 60 scored 48→49. The favorable
six-question development result did not predict the increase in paid work.

For the unchanged candidate on these frozen questions, LLaDA Q8→16
was 71→72 correct and 2.555→3.833 TPF; Gemma Q64→128 was 77→79 and
13.635→18.747. These are comparisons **across devices and fixed runs**.
Only each candidate versus its own paid IBS is a matched device contrast.
With only two offers measured per model this is neither a ten-slot
monotonicity result nor a state-conditioned response curve.

To reproduce the four-point supplement without changing the v1 figure:

```sh
d=experiments/t2t/serving_slots/controls/paid-agreement
python3 tools/plot_paid_budget_points.py "$d/llada-q8/all90-summary.json" \
  "$d/llada-q16-frozen90.json" "$d/gemma-q64-frozen90.json" "$d/gemma-q128-frozen90.json" \
  --historical "$d/../fixed-request-comparison/metrics.csv" \
  --matched-live "$d/q128-same-card-live-point.json" --out /tmp/paid-agreement-four-points.png
```

## Fixed-offer decision: reject this as the unified default

The treatment admits a usually rejected ordinary MASK proposal after two
consecutive measured votes above 0.5 for the same token on the same row.
Each new ID still incurs a real writeback. Each arrow compares stock paid
IBS to this treatment at the same fixed offer on the same device.

| Model, physical Q | Correct / 90 | Macro TPF | Pooled TE | Total F | Selected decoder Q | Rescues / losses |
|---|---:|---:|---:|---:|---:|---:|
| LLaDA 2.1, 16 (H0) | 72 → 72 | 3.406 → 3.833 | 20.65% → 23.39% | 15,407 → 14,150 | 205,841 → 187,569 | 6 / 6 |
| Gemma, 64 (A2) | 83 → 77 | 12.704 → 13.635 | 20.84% → 21.61% | 2,930 → 2,997 | 161,762 → 164,152 | 1 / 7 |
| Gemma, 128 (A3) | 78 → 79 | 16.451 → 18.747 | 14.63% → 17.07% | 2,234 → 1,991 | 225,115 → 200,177 | 5 / 4 |

| Model, Q | Original 30: correct; TPF; TE (%) | Added 60: correct; TPF; TE (%) |
|---|---|---|
| LLaDA, 16 | 23 → 25; 3.226 → 3.712; 19.42 → 23.22 | 49 → 47; 3.496 → 3.894; 21.28 → 23.48 |
| Gemma, 64 | 29 → 29; 12.674 → 13.386; 20.89 → 22.59 | 54 → 48; 12.718 → 13.759; 20.81 → 21.24 |
| Gemma, 128 | 27 → 28; 16.419 → 19.166; 13.79 → 16.79 | 51 → 51; 16.467 → 18.537; 15.04 → 17.20 |

On Gemma Q64, N increased from 33,704 to 35,467. The symmetric per-request
decomposition of TPF attributes +1.054 to output length and -0.122 to call
count: the +0.931 does **not** mean less work. Seven scores were lost and only
one rescued, all on the added 60. The stratified paired 95% interval for
accuracy is [-12.22, -1.11] percentage points (descriptive, not adjusted for
multiple comparisons). LLaDA Q16 instead saved 1,257 calls and 18,272 rows;
the +0.428 macro TPF decomposes into +0.085 from length and +0.343 from calls.
Its net score conceals twelve flips; #75 failed to converge in all three arms
and remains charged with zero delivery. Historical LLaDA Live got 75/90 at Q16.
**Gemma Q128 Live was subsequently rerun on the same A3.** It scored 88/90
at 13.047 TPF and 11.97% TE (`N=30,351, F=2,384, Q=253,634`), reproducing the
historical point. The A3 gap versus the repeat treatment is nine correct
answers, with 393 additional forwards. This is an end-to-end method comparison:
both methods use the same board, prompts and scorer, but their native/sparse
acceptance, writeback and closing rules differ. See the [plot data and source
digests](q128-same-card-live-point.json). These three paired registrations
are not a ten-tier curve or evidence for a mixed-budget policy.

Breaking down the Q64 result explains the increase in total computation:

| Dataset (30 each) | Correct, stock → repeat | Change in total N | Change in total F | Change in selected Q |
|---|---:|---:|---:|---:|
| GSM8K | 30 → 29 | -162 | -138 | -8,750 |
| MATH500 | 28 → 27 | -136 | -174 | -10,400 |
| HumanEval | 25 → 21 | +2,061 | +379 | +21,540 |

On frozen index 72 (`HumanEval/9`), stock delivered 176 tokens in 10 F
and passed. The treatment delivered 2,046 in 320 F and failed. At the
first difference (block 0, decoder tick 5) it additionally accepted rows
139 and 166. The later quality change cannot be attributed to either one
of those rows individually. Index 88 also went from passing to failing:
391 → 926 tokens and 45 → 206 F. Both are in the primary 90, not removed
as outliers; neither implies a serving-time policy for code prompts.

At each A2 request's **first additional acceptance**, a row is a verified
out-of-order write if a lower-position row was measured on that same tick
and remained rejected by native **and** treatment. Among 7 lost-score cases
all extras were behind such a hole on 5; among 82 same-score cases this also
happened on 30 (74 had any fork). The one rescued case had a fork but no
such observed hole. For #72 both extras were after still-rejected row 136.
Only measured rows are tested: absence of a witness does not prove a fully
resolved physical prefix. This post-hoc association is neither a cause nor
evidence that a frontier filter would improve quality.

[Budget figure with numeric values](fixed-budget90.png) ·
[Replot script](../../../../../tools/plot_paid_budget_points.py) ·
[LLaDA Q16](llada-q16-frozen90.json) · [Gemma Q64](gemma-q64-frozen90.json) ·
[Gemma Q128](gemma-q128-frozen90.json). The per-model panel includes the
historical nominal IBS, native and Flow W1/W2 where measured. No estimate is
filled in for an unmeasured offer. Clean Q64 receipts match all six A2 pilot
triples as well as all 90 tokenized inputs from the archived A0 Q64 encoder
prefills. Only the inputs, not outputs or scores, are borrowed from A0.

### Can the right edge of Q hide already built K on a sampling tick?

On the A0 paid-IBS control (same 90 questions at each offer) compare the
largest projected row to the set previously initialized in that physical
block. Read the actual sampler event; do not classify every decoder call as
sampling. Resolved and paid rows are separate from unresolved rows, which can
change noise even without admission debt.

| Gemma Q | Sampling ticks hiding any initialized right K / all sampling ticks | Payment-only ticks hiding such K / all payment-only ticks |
|---:|---:|---:|
| 32 | 0 / 5,168 | 5 / 213 |
| 64 | 0 / 2,658 | 3 / 177 |
| 128 | 0 / 1,867 | 3 / 176 |

**All eleven retreats were the final decoder call of the request**: no later
sampler used its writes. Extending this particular read horizon cannot change
an earlier sampling decision on these tapes. This is not proof that the
existing KV is always current, or applies to Live or OBS. No treatment GPU
run is justified for this rule. [Counters and raw hashes](gemma-k-horizon.json) ·
[Audit code](../../../../../tools/audit_gemma_resident_horizon.py).

## First full confirmation: DiffusionGemma Q128

Fixed Q128, same A3, same prompt/seed/scorer and paid IBS stopping rule in
both arms. No change to which rows the planner selects before the first
additional admission. Native resolved-row T2T is not implemented by the IBS
adapter. This is a sampler result, not a complete ten-tier request kernel.

| Cohort | Correct, IBS / candidate | Request TPF | Pooled token efficiency |
|---|---:|---:|---:|
| Original 30 | 27 / 28 | 16.419 / 19.166 | 13.79% / 16.79% |
| Additional 60 | 51 / 51 | 16.467 / 18.537 | 15.04% / 17.20% |
| All 90 | 78 / 79 | 16.451 / 18.747 | 14.63% / 17.07% |
| Excluding the six screening cases (84) | 72 / 73 | 16.508 / 18.797 | 14.69% / 17.09% |

N: 32,931 to 34,166; actual F (decoder plus generation encoder): 2,234 to
1,991; decoder Q: 225,115 to 200,177. 59 requests took fewer calls, 16 the
same and 15 more. Five answers changed to correct and four to incorrect.
This is an efficiency result on this workload, not evidence of general quality
improvement. For each paired N/F, symmetrically averaging the two possible
orders of changing N and F attributes +0.752 macro TPF to output length and
+1.544 to call count. This arithmetic attribution is not a causal model.
Historical Live Q128 got 88/90 at 13.047 TPF, a different point on the
quality-throughput trade-off, not an arm of this experiment.

[Verified cohorts, flips and receipt hashes](gemma-q128-frozen90.json) ·
[Figure](gemma-q128-frozen90.png) · [Plotter](plot.py) ·
[Independent scorer of the completed paired bills](../../../../../tools/agreement_fixed90_comparison.py).
The original 90 A3 receipts are on n164 in `results/gemma-paid-agreement-q128-frozen90-01`.
Its original controller's aggregate reader failed after all three arms were
measured. Four first-fork checks also used absolute positions to look up
block-local sampler rows (indices 26, 80, 83, 89). These are **not** decoder
failures. All failed checks were rerun on A3 at the original seeds and arm
order. A failed first repair compared Python tuples with JSON arrays; that
attempt remains separate and is not scored. The four successful replays in
`results/gemma-q128-fork-audit-v3` verified the local-to-absolute map, raw
proposals/probabilities, RNG, input IDs, unchanged base admissions and ensuing
payment. All arm bills, scores and original compact arm objects agreed exactly.
Use `--replay-dir` with that directory for the independent comparison. Replay
compute is validation overhead, not a free generator step or a fourth arm.

This experiment asks whether a repeated measured proposal can earn acceptance
before the strict IBS confidence threshold fires. It does **not** predict
T2T from unobserved logits, change the selection of Q at the first differing
step, or eliminate the next payment on the new token. See the
[shared rule](../../../../../bdlm_serving/methods/t2t/paid_agreement.py) and
[registration and background](../flow-writeback/boundary-sensitivity.md).

## Development screen, not the frozen90 result

Same six tasks (`7,13,25,33,55,71`); run both policies on the same GPU per
model. The comparator is capped IBS with certified paid-EOS. Each pair uses
the same seed, endpoint and scorer. The stock and observation-only arm agree
on all six outputs in each model. The two models use one common rule with
the same floor of 0.5. No finding here implies increasing quality with Q.

| Model and fixed Q | Arm | Correct | Mean request TPF | Pooled TE | Delivered N | Actual F | Decoder Q |
|---|---|---:|---:|---:|---:|---:|---:|
| LLaDA 2.1 Q16 | Paid IBS | 6/6 | 3.862 | 26.37% | 1,379 | 399 | 5,229 |
| LLaDA 2.1 Q16 | Also accept agreement | 6/6 | 4.156 | 27.62% | 1,384 | 371 | 5,010 |
| DiffusionGemma Q128 | Paid IBS | 6/6 | 15.653 | 13.46% | 1,444 | 100 | 10,725 |
| DiffusionGemma Q128 | Also accept agreement | 6/6 | 18.034 | 16.66% | 1,375 | 79 | 8,252 |

These are half-block offers throughout each request. LLaDA used fewer F on
five cases; index 13 used eight more. Gemma used fewer F on four and the
same F on two; index 13 alone saved 11 of the 21 F. Its answer also shortened
from 274 to 251 nonspecial tokens and required no generation encoder call
instead of one. Neither seed-level attribution nor six ceiling-quality cases
establish a general quality or speed result.

Verified compact sources on n164 under `/data1/project_data/gx-erqw-20261123/results/`:
`llada-ibs-q16-paid-agreement-h0-v1/receipts.ndjson` and
`gemma-paid-agreement-q128-01/summary.compact.json`.
Compute TPF as `mean(N_i/F_i)` and TE as `100*sum(N_i)/sum(decoder_Q_i)`.
LLaDA uses the ticket's `arms[arm]`; Gemma uses `per_case[i].arms[arm]`,
with F/Q inside its `bill`. The six IDs must match before computing metrics.

## Decision recorded before full confirmation

Freeze the algorithm, threshold, seed and cohorts. Confirm independently on
the same 90 questions per model, preserving original30/new60, six-screen and
remaining-84 slices. Check physical Q on every decoder projection, the first
additional acceptance, and its actual payment. Failed requests remain in the
denominator. Report paired correctness gains/losses, IDs, N/F/Q, individual
F gains/losses, macro TPF and pooled TE. Compare to same-device paid IBS;
any native/Flow/Live points from other registrations are references, not
matched interventions. Neither a quality loss with more TPF nor a TPF gain
caused solely by longer output is a joint win. Do not promote to all ten
offers, schedule mixed budgets, or stack another factor on the six-case result.

With matched task IDs and seeds, the historical Live / A3 paid IBS / A3
candidate scores were 30/29/28 on GSM8K, 28/24/27 on MATH500, and 30/25/24
on HumanEval (each out of 30). Of the nine Live successes missed by the
candidate, five were already missed by IBS and four were lost at treatment.
Both Live failures remained failures under both A3 arms. These are outcome
overlaps across executors and devices, not causal evidence about T2T.
[Join and per-task counts](q128-live-gap.json) ·
[Same-dose A3 placement ablation](equal-dose.md).

Resampling the **same paired requests** within each of the three datasets
(4,000 draws, [reproducible intervals](q128-paired-intervals.json)) gives
descriptive 95% intervals for candidate minus same-card IBS: accuracy
**−5.56 to +7.78 percentage points**, macro TPF **+1.66 to +2.90**,
pooled TE **+1.28 to +3.47 points**. Mean F/request has a wider interval
of **−6.06 to +0.16** despite the measured total F falling. The intervals
reflect case variability on this frozen panel, not device or seed variability.

## Is the repeat signal picking different positions from confidence alone?

On the unchanged GPU **observer** trajectory, take each step with `k>0` extra
agreement votes. Select exactly `k` of the currently measured, ordinary,
native-rejected proposals with probability `>0.5`, sorted by current
confidence (absolute row ascending on ties). Both policies are charged the
same number of admissions at each such state; this is a replay, not a
second decoding run.

| Six fixed cases | Positive steps | Equal dose, row-events | Same rows | Replaced by current-confidence top-k | Steps with any difference |
|---|---:|---:|---:|---:|---:|
| LLaDA Q16 | 109 | 221 | 174 | 47 | 35 |
| Gemma Q128 | 24 | 81 | 34 | 47 | 17 |

On LLaDA's complete observer tape we followed **only the differing rows** to
the next native admission of that same absolute position and block. The
earlier proposed ID matched 38/47 for repeat-only versus 39/47 for
confidence-only (no censoring here). The unit is a row-event; repeat visits
are not independent examples. This proxy does not establish correct tokens
or downstream answer quality, and it does not support a claim that the vote
is a more reliable ranking. Gemma kept the positive observation steps, not
the entire unresolved trajectory; row-set comparison is possible, the same
future-label comparison is not. The shared-row counts are not counts of
the applied arm's admissions after its trajectory changes.

[Raw digests and six per-case counters](dose-screen.json) ·
[Replay](../../../../../tools/audit_agreement_dose.py). Reproduce by passing
the n164 result root; no model and no new generation are needed.

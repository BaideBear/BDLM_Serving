# Two paid observations on the same unresolved row

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

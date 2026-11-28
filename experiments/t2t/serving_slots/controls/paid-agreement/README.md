# Two paid observations on the same unresolved row

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

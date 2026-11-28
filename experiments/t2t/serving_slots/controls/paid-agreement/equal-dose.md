# Does placement matter at fixed admission dose?

Before the GPU screen: on A3, run the frozen Gemma Q128 six (7, 13, 25, 33,
55, 71) with the stock, observer and repeat arms unchanged, plus a fourth
arm that accepts exactly as many paid ordinary MASK rows as the repeat rule
requests **on this state**. Rank the eligible, originally rejected rows by
current FP32 probability `>0.5`, breaking ties by increasing absolute position.
The exclusion list and physical Q cap are unchanged. After the arms diverge,
each uses its own history to compute the dose. No equality of whole-run
admissions is assumed. This is not a sweep or a change to Q allocation.

Require stock/observer identity, first common fork with matching inputs,
RNG, logits-derived probabilities, native decisions, identical number of
additional admissions, and a real change of positions. Audit the decision on
every observed step in both treatments; use the same original paid-debt
reconciliation for all four paths. For the prior three arms verify reproduction
of the same-card frozen90 development cases. Report N/F/Q, paired scores and
TPF/TE even if unfavorable; no automatic 90 promotion. No wall-time claim.

## Observed on the six

| Fixed Q128 on A3 | Correct | TPF | Token efficiency | N | F | Selected Q |
|---|---:|---:|---:|---:|---:|---:|
| Paid IBS | 6/6 | 15.653 | 13.46% | 1,444 | 100 | 10,725 |
| Accept the repeated proposals | 6/6 | 18.034 | 16.66% | 1,375 | 79 | 8,252 |
| Same local count, highest current confidence | 6/6 | 18.046 | 16.46% | 1,526 | 88 | 9,270 |

Five requests reached a real placement fork; #71 had none. #13 alone used
eleven fewer F with the repeat rule than with current confidence, more
than the net difference of nine. The near-tie in mean TPF does not imply
equal work: the confidence arm delivered 151 more nonspecial tokens.
All three previously measured arms reproduced their original six-case objects
and their frozen90 scores and bills. No ranking superiority is established.
[Per-case comparisons and digests](equal-dose-six.json) ·
[Independent reproduction checks](../../../../../tools/summarize_gemma_equal_dose.py).

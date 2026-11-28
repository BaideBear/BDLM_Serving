# Are paid writebacks useful for detecting a bad provisional admission?

Gemma, physical A3, constant Q128, same six and original treatment. Observe
the first actual projection of each newly written ID. Separate ordinary native
IBS admissions from the extra repeat-rule admissions. Save the paid input
ID, current top-1 and both tokens' FP32 probabilities on n164. The observer
must reproduce the complete treatment object (not just the score); every
admission must be linked once to a real decoder projection of the new ID.

The signal is an **ordinary different** winner at `p > .9` on an extra's
first payment. Require at least two distinct exposed requests on the six
before considering a one-correction-per-new-token trial. A no-go is scoped
to this threshold and these six. Observation changes no selection, sampling
or KV, and can show exposure only, not a beneficial edit. Agreement after the
token itself was supplied is not independent evidence of correctness.

## Actual Q128 observations

| First paid input after admission | Rows | Different top-1 | Ordinary different top-1 with p > .9 |
|---|---:|---:|---:|
| Original IBS acceptance on the repeat trajectory | 1,853 | 11 | 7 |
| Extra two-observation acceptance | 25 | 4 | 3 |

The three high-confidence alternatives for the second row of this table
occurred in two distinct requests, #13 and #55. This meets the exposure
criterion only. No token was changed by this experiment; do not interpret
3/25 as actual T2T or 22/25 as correct tokens. Native and extra rows are not
exchangeable samples. All six observer trajectories matched their uninstrumented
treatments exactly. The stock/observer/repeat receipts also reproduced the
original A3 six-case objects. Across the six, 1,878 observed first payments
reconciled to the physical bill. The raw answers and readbacks stay on n164
in `results/gemma-payment-observer-a3-v1`. The A3 job exited 0, with 0 MiB
in use afterward.

[Per-case counters and raw receipt digests](payment-observer-six.json) ·
[Independent summarizer](../../../../../tools/summarize_gemma_paid_observer.py).

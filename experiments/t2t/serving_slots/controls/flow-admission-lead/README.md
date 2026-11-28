# Earlier successor admission in compact LLaDA Flow W2

LLaDA 2.1, one request, block size 32, offer 64, n164 physical H4.
The comparator is the sealed [compact W2](../llada-flow-active-w2/README.md),
not dense W2, IBS or a deployed serving endpoint.

## Sealed result: retain native admission

90 distinct triplets, 270 GPU requests. Every fresh control and no-op
exactly reproduced the historical compact trajectory. CPU-only re-scoring
and reconstruction on n164 produced a byte-identical compact report.
The H4 guard returned holder PID 2075819 (start ticks 2041077330).

| Cohort | Correct, control / lead | Paired gain / loss | Output N | Actual F | Decoder Q | Macro TPF | Pooled TE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original 30 | 27 / 26 | 1 / 2 | 12,778 / 13,248 | 1,704 / 2,020 | 75,872 / 88,832 | 9.815 / 10.297 | 16.84% / 14.91% |
| Additional 60 | 51 / 49 | 1 / 3 | 27,701 / 28,948 | 3,854 / 4,207 | 166,080 / 186,048 | 10.075 / 10.084 | 16.68% / 15.56% |
| Frozen 90 | 78 / 75 | 2 / 5 | 40,479 / 42,196 | 5,558 / 6,227 | 241,952 / 274,880 | 9.988 / 10.155 | 16.73% / 15.35% |

Score gains: #15 and #50. Losses: #3, #16, #60, #61, #63. Equal
final token sequences: 13/30 and 36/60 (49/90 total). Each arm returned
86 EOS and 4 capped outputs. Lead made 208 actual early admissions across
50 requests, while no-op counted 216 opportunities on the unchanged paths.
The 669 additional generation calls and 360 additional dual-block calls
account exactly for all `32*(669+360) = 32,928` extra Q. It is possible for
more requests to lower F than raise F (27 lower, 17 higher, 46 unchanged)
while total F rises. The macro TPF increase is not a compute saving or a
quality win. No promotion; no kernel-speed or model-wide conclusion.

## What changed

When exactly one unfinished head has a physical successor, observe two
consecutive sampled frontiers of that same single head. If the native `.8`
gate still defers but `f + max(0, f - prev_f) >= .8`, admit the successor
for the **following** forward. A head change, a two-block sample or a
finished head invalidates the previous progress. The native `.8` path
still has priority. The intervention changes neither sampling and edits
nor visibility, writeback, EOS commit or the 2048-token cap.

The [registration](registration.json) freezes the exact parent sources,
three arms, six qualification indices and all 84 remaining indices.
Control runs the parent; no-op evaluates the same test but cannot admit;
lead is the only arm that may override a deferral. Order rotates C/N/L,
N/L/C, L/C/N. The qualification cases count once in the full population.
The separate [blind audit](exposure.md) qualified the rule for measurement:
50/90 control requests, 194 distinct eligible heads, 216 eligible ticks.
Of the 194 heads, 103 reached the native threshold after one more forward,
3 ultimately committed EOS, and 193 eventually had a visited successor.
None of these counts is a saving in forwards.

Qualification: the six cases were 7, 13, 25, 33, 55, 71. Cases 13, 33
and 55 admitted early. For #13 the zero-based deciding tick was 5, head
start 96, `prev_f=0.2105263174`, `f=0.7333333492`. The next projection
changed 32 to 64, with identical incoming head state. Its total F *rose*
from 22 to 26. For #33 and #55 total F was unchanged.

## Verification and reproduction

The collector compares each fresh control and no-op to the archived control
including all decisions, output IDs and physical calls. For a lead admission
it checks identical history through the deciding tick and equal incoming
head state on the first widened call. Without an admission the entire
lead trajectory must match. Per-layer QKV input widths at all 20 layers,
actual KV writes, request calls and upstream step counts are cross-checked.
Token writeback, immutable prompt rows and ordered EOS are reconstructed
from the raw receipts on n164.

F counts generation calls; one prefill per request is separate. Q counts
real history-reading generation decoder rows once per call, not once per
layer. LLaDA has no encoder here. N excludes special tokens. Macro TPF is
the unweighted mean of request N/F, and pooled TE is sum(N)/sum(Q).
For this executor `Q = 32*(F + double_block_calls)`; both terms can change
when the subsequent token trajectory changes.

GPU launcher (already used, not a request to rerun): `python3 guard.py`
in `n164:/data1/project_data/gx-erqw-20261123/results/llada-flow-admission-lead-h4-v1/`.
The guard checks the exact preceding H4 handoff, UUID, start ticks and
source hashes, and refuses to overwrite results. `study.py report` in the
same frozen Python environment re-scores and audits the **existing** raw
records without running the model. `python3 summarize.py` in this control
directory uses only compact receipts. Raw inputs and trajectories remain
on n164.

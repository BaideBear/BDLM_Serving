# Strict IBS: how much work follows an irrevocable EOS?

Historical exposure, **not** measured acceleration. The current upstream-style
control finishes a physical block before trimming at EOS. Its `ready` pointer
already identifies an irrevocable prefix. For successful EOS requests,
locate the first call where `ready` covers the delivered EOS, and check that
no subsequent ordinary call selects an already retired row. In LLaDA the
address includes prompt length; in Gemma it is generated-canvas relative.
The new runtime must inspect only that currently frozen, generated prefix,
not a final-output oracle. Frozen requests that failed or did not deliver EOS
are kept and contribute no evidence of removable work.

| Model | Q | Original F, incl. generation encoder | Requests with later F / 90 | Actual ordinary F after first certificate | Later F if waiting for all debt to clear naturally | Requests with suffix debt at first certificate |
|---|---:|---:|---:|---:|---:|---:|
| LLaDA 2.1 | 4 | 27,397 | 74 | 556 | 394 | 31 |
| LLaDA 2.1 | 8 | 19,692 | 73 | 294 | 228 | 18 |
| LLaDA 2.1 | 16 | 15,480 | 41 | 85 | 58 | 12 |
| LLaDA 2.1 | 32 | 12,694 | 3 | 3 | 0 | 3 |
| DiffusionGemma | 32 | 6,287 | 78 | 1,036 | 42 | 75 |
| DiffusionGemma | 64 | 3,498 | 76 | 390 | 12 | 74 |
| DiffusionGemma | 128 | 2,055 | 60 | 146 | 11 | 55 |
| DiffusionGemma | 256 | 1,563 | 0 | 0 | 0 | 0 |

A prospective implementation can either wait until there is no debt, or
pay one actual decoder projection over the remaining suffix debt without
sampling. These are different policies. Running and billing that final
projection is essential before reporting measured TPF or efficiency. The
debt recorded here is that of each pinned baseline: accepted rows in Gemma
IBS; admissions and eligible edits recorded by LLaDA IBS. Do not extend this
certificate to a native sampler that may revise the proposed prefix. Apply
the same stopping policy to any compatible baseline before attributing a
gain to query selection.

[Evidence, original30/new60 splits, Q counters and raw-source hashes](exposure.json).
Reproduce on n164: `python3 tools/ibs_frozen_eos_tail.py
/data1/project_data/gx-erqw-20261123/runtime/t2t-native-1125/prefix90`.

## Shared stop predicate on the six registered GPU pairs per model

The [pure decision function](../../../../../bdlm_serving/methods/t2t/frozen_stop.py)
consumes the post-write immutable frontier, generated interval, absolute
EOS position, outstanding row set and offer. It neither performs a model
call nor marks anything paid. Only a caller that guarantees the prefix is
immutable may use it. A full block ending in EOS is an ordinary end, not
an early termination. A suffix payment never advances the sampler; do not
finish until a real paid projection has cleared the debt.

| Registered GPU pilot | Stock decoder ticks replayed | Stop with no debt | Stop after suffix-only payment | Ordinary block EOS | Measured payment F / Q |
|---|---:|---:|---:|---:|---:|
| LLaDA Q4, six | 1,814 | 4 | 2 | 0 | 2 / 3 |
| Gemma Q32, six | 289 | 1 | 4 | 1 | 4 / 20 |

The [audit](../../../../../tools/audit_shared_frozen_stop.py) reconstructs the
LLaDA prefix from ordinary writes and revocations, excludes prompt tokens,
and reads Gemma's independently reconciled per-tick states. It checks the
first trigger and any physical payment against the treatment arm. [Compact
output with raw-tape hashes](pilot-replay.json). The pilot replay is **not**
a second inference run or evidence that the Q-selection policy beats IBS.
The in-flight registered GPU executors were not modified.

## Same-device frozen90 result: Gemma Q32

All four arms ran each of the 90 requests on the same A800, with prompt,
seed, weights, scoring and ceiling fixed. The observation-only arm matched
stock; all 90 sets of final IDs and generation encoder input records match
across arms. The intervention changes the work after the delivered EOS is
irrevocable, not the answer or the number of final tokens.

| IBS stop rule | Correct | Macro TPF | Pooled TE | F (includes 101 generation encoders) | Actual decoder Q |
|---|---:|---:|---:|---:|---:|
| Stock / observation only | 78/90 | 6.466 | 18.64% | 6,549 | 193,583 |
| Wait for zero debt | 78/90 | 6.812 | 19.00% | 6,434 | 189,914 |
| Pay live suffix debt and stop | 78/90 | 8.168 | 22.41% | 5,482 | 160,966 |

The last arm saves real forwards on 84 cases, ties on six and loses none
(median 9 fewer). Its 75 additional suffix payments cost 470 real decoder
Q and are included above. Output N remains 36,080. Original30 is 27/30 in
both arms, TPF 6.439 to 7.989; additional60 is 51/60, 6.479 to 8.258.
The historical IBS row of 79/90, 6.840, 19.33% was not the control on this
device. Do not attribute that one-answer difference to stopping. This is
a stronger *IBS baseline*, not candidate Q selection beating IBS.

[Figure](measured/frozen90.png) · [Plot data by cohort](measured/cohorts.csv)
· [Paired compact receipt](../gemma-factors/ibs-certified-eos/q32-frozen90-summary.json)
· [Plot script](../../../../../tools/plot_certified_ibs.py).

## Same-device frozen90 result: LLaDA Q4

The same certificate on n164 H0 preserves all 90 delivered ID sequences
(including six unchanged failed requests), with exactly 37,754 nonspecial
output tokens in both arms. It skips 556 ordinary decoder forwards but
incurs 31 real suffix payments (51 Q), for a **net 525 fewer F**.

| LLaDA strict capped IBS | Correct | Macro TPF | Pooled TE | F | Decoder Q |
|---|---:|---:|---:|---:|---:|
| Stock / observation only | 69/90 | 1.395 | 35.66% | 27,397 | 105,885 |
| Pay suffix debt and stop | 69/90 | 1.445 | 36.35% | 26,872 | 103,853 |

The [two-model figure](measured/frozen90.png) uses a dashed stock reference
in each panel. This result strengthens the comparator; it does not validate
any candidate selector or solve the six progress failures.
The compact input for redrawing the LLaDA panel is
`n164:/data1/project_data/gx-erqw-20261123/results/llada-ibs-certified-eos-h0-v1/receipts.ndjson`
(SHA-256 `fce9fdaa933815010cee6b5009de2bdff2de535e07ee64e96ab80b52ea3382a4`).

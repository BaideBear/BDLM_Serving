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

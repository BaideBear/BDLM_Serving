# Input-embedding context locates more strict old edits on frozen30

**Pass:** the fixed feature family added 3, 7, and 3 reserved hits at the
three physical budgets. The [protocol](PROTOCOL.md) required at least two
strict gains and no fall. Both arms saw the same 191 eligible target events.

| Evaluation | Seats per arm | Position hits | + embedding context | Difference | Requests better / worse |
|---|---:|---:|---:|---:|---:|
| Q4/1 reserved | 452 | 9 | 12 | +3 | 3 / 1 |
| Q8/1 reserved | 710 | 23 | 30 | +7 | 5 / 0 |
| Q16/2 reserved | 1,688 | 56 | 59 | +3 | 5 / 4 |
| Top 1 | 880 | 28 | 37 | +9 | 6 / 0 |
| Top 4 | 3,437 | 88 | 94 | +6 | 7 / 3 |
| Top 8 | 6,533 | 145 | 148 | +3 | 3 / 3 |

Both logistic models were fitted only on the same 24 disjoint calibration
requests. The augmentation uses the current old token and still-visible IDs
written in the preceding native round. It takes the maximum projected cosine
within eight positions, its interaction with the existing distance-4 near-MASK
indicator, and neighbour presence. No logits, KV, current-transition labels,
future writes or learned token-ID coefficients enter the features. The shared
replay excludes prompt, first-block-step and verification rows from scoring;
reserved hits do not include incidental old-row backfill. Top-k ranks the old
pool without the physical reserve constraints.

Q8/1 is the strongest result: reserved precision improved from 23/710 (3.24%)
to 30/710 (4.23%). The Q16/2 gain was less consistent across requests. This
comparison establishes a gain for the **bundle**, not separately for cosine
or its interaction. Presence duplicates an existing near-fresh positional
indicator and can change its effective L2 regularization. No further ablation
was fitted. Frozen30 is reused; this is evidence for testing on new requests,
not for sparse rollout quality or end-to-end latency.

**CPU cost.** On n164 core 23, calibration24 provided 1,035 selected states
with 16,092 old candidates. Starting with an empty projected-row cache, the
context pass took 83.72 ms; reading, projecting and normalizing the 772 distinct
needed IDs accounted for 32.76 ms of that time. With those rows cached,
the median of 21 passes was 38.59 ms, or 37.29 us per eligible state. Timings
cover only the context feature operation on predecoded states, including
neighbour filtering and dot products. They exclude checkpoint opening/sign
matrix construction, trace decoding, positional features, fit, and inference.
Cold refers to the process's projected-row cache, not the OS page cache.
Each cached unit vector uses 64 bytes of FP32 payload; the shared sign matrix
uses 128 KiB. No vectors, IDs or weights are stored in the study outputs.

The local snapshot's `get_input_embeddings()` returns `model.word_embeddings`;
its LM head is separate and untied. The code checks the archived index/config
hashes, input shard size and tensor shape/dtype, and uses `safe_open` on CPU
for slices of `model.word_embeddings.weight` (157184 x 2048 BF16).

[Hit counts](comparison.csv) | [Timing](timing.json) | [Implementation](study.py)

Reproduce from the n164 repository root using the existing local snapshot:

```sh
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
taskset -c 23 ../venv/bin/python -B -m experiments.t2t.calibrated.embedding_context.study \
  --cal ../t2t-native-1125/calibration24-v1-llada21 --native ../t2t-native-1125/llada21 \
  --checkpoint /data0/ysy/models/LLaDA2.1-mini --out experiments/t2t/calibrated/embedding_context
../venv/bin/python -B -m pytest -q -p no:cacheprovider \
  experiments/t2t/calibrated/embedding_context/test_study.py \
  experiments/t2t/calibrated/history/test_history.py tests/test_lexical_hazard.py
```

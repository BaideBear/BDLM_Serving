# Gemma B native-argmax request qualification

The [prior endpoint audit](gemma-b-native-argmax-endpoint.md) measured native
drafts but still delivered and encoded the ordered verify result. It did not
exercise the new delivery path. This test compares that boundary with real
`model.generate` and a separate paid-verify control on the same H0.

## Request interface

```python
from experiments.t2t.serving_slots.gemma_native_argmax_request import open_native_argmax_gemma

request = open_native_argmax_gemma(model, inputs, seed=seed, max_new_tokens=2048)
while request.status == "ready":
    event = request.step(256)
output_ids = request.result()
```

Each step runs one real decoder call at Q256. It reads all rows of the
prior canvas, including earlier sampled writes, before calling the existing
sampler and model criterion. The hit step returns the exact argmax passed to
that criterion. It delivers through the first EOS inclusive, or the whole
block if there is no EOS. If there is another block, one real encoder call at
Q256 encodes those exact 256 delivered IDs on that same step. No encoder runs
after EOS or at the output cap. Block initialization continues the request RNG.

`dirty_before_paid` tracks previous writes actually read by the Q256 decoder.
`unread_speculative_rows` tracks edits to the post-hit sampled canvas. Neither
the next decoder nor the encoder reads that canvas: the encoder consumes the
separate, measured argmax. These unread edits are not unpaid required work
and are not subtracted from the denoising forward. In the paid control an
additional real Q256 verify reads that edited canvas and may deliver and
encode different IDs. Prefill and every nonterminal encoder remain real work.

This is batch one, serial, one full bidirectional head, dynamic cache, no tail,
no rebuild, no mixed budget. Non-hit denoising clock exhaustion fails closed;
it is not evidence of a native stop. The API does not establish a sparse
sampler, sparse stopping/verification, wall-clock speedup, or a serving default.

## Paired gate

Frozen90 order starts with `0, 10, 20, 1, 11, 21`, then the remaining 84
in index order without rerunning those six. Each request uses the same
checkpoint, prompt IDs, seed and physical GPU across three arms, with forward
and reverse arm orders alternated:

1. Upstream `model.generate`, with observation-only hooks.
2. The Q256 request stepped above, with same-turn native-argmax delivery.
3. Q256 with the same denoising setup but an additional real ordered verify
   per block, using its output for delivery and nonterminal encoding.

Every pair must agree between `generate` and `step` on complete delivered IDs,
all full hit argmaxes, sampled decoder input and argmax fingerprints at every
turn, observed encoder inputs, physical D/E/F/Q ledger, stop and score.
The paid control must agree until the first changed commitment. At least two
of the six requests must exercise a nonterminal encoder. The worker writes a
six-case qualification receipt before running case seven and stops on any
failed subsequent pair. Hooks count actual encoder/decoder forwards and all
85 decoder Q/K/V projectors; no wall-clock claim is made from this instrumentation.

## Result

The six-case gate passed 6/6, with three multiblock requests, nine actual
criterion hits, and three actual generation encoder calls in each arm. The
endpoint matched upstream in all 90 pairs on the same stack (`torch
2.9.0+cu128`, Transformers `5.12.0`), including 59 multiblock requests,
171 full hit argmaxes, and all 81 actual nonterminal encoder inputs.
All requests ended at EOS. Prefill is excluded from the generation F/Q
columns: there were 90 real prefill calls and 10,065 prefill rows in each arm.

| Arm | Sample D | Verify D | Generation E | Generation F | Generation Q | N | Correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Upstream `generate` | 1316 | 0 | 81 | 1397 | 357632 | 31352 | 85/90 |
| Native-argmax `step` | 1316 | 0 | 81 | 1397 | 357632 | 31352 | 85/90 |
| Paid ordered verify | 1313 | 171 | 81 | 1565 | 400640 | 31393 | 85/90 |

The paid control agreed through the first changed commitment in all 90
pairs. Its delivered IDs matched native in 72/90 requests. Verification
changed 18 block deliveries relative to that arm's own hit draft, including
seven actual encoder inputs. The observed difference is 168 generation
forwards, not a subtraction of 171 from an otherwise fixed trace: the changed
contexts also changed later denoising work net by three calls. No encoder
calls were saved. The step arm read 174,873 tracked prior-write rows within
its billed decoder calls; 3,297 post-hit speculative rows were never read
and generate no F/Q credit.

## H0 receipt locations

Host `n164` (`amax`), physical card H0,
`GPU-ceadb3cd-e63f-1446-a2d5-a291144f8efb`. The launch uses the existing
`shared-formal-H0.lock` and staged `ready`/`go` handoff, with observed
UID, start ticks, boot ID and UUID checked before release and on return.

Remote roots:

- `/data1/project_data/gx-erqw-20261123/runtime/gemma-native-argmax-h0-v4`
  (registered snapshot; `source.sha256` and `supervisor.log`).
- `/data1/project_data/gx-erqw-20261123/results/gemma-native-argmax-h0-v4`
  (`registration.json`, `handoff`, `worker.identity`, `worker.log`,
  `qualified-six.json` and `complete.json` when earned, and `returned.identity`
  on a verified handback; per-request `pairs/` and `receipts.ndjson` stay remote).

Attempts `v1` through `v3` did not record any complete pair. `v1` stopped
before `go`; `v2` reached the native result but hit an evaluator field-name
error; `v3` reached the request step but hit a `WaveKV.live` type assumption.
Each attempt verified holder return before a new launch. The `v4` source
snapshot includes the fixes and a real-model CPU integration test across two
blocks for both request arms (with a deterministic test criterion). The
stopping criterion in all GPU arms is the untouched pinned model criterion.

The separate receipt audit checked all 90 raw file hashes and reconstructed
the arm totals and delivery/encoder matches before reporting completion.
At `2026-11-28T17:57:17+08:00` the worker (`1685677`, ticks `2040559488`)
had exited, the shared lock (`66305:200303112`) was acquirable, and the
verified restored holder (`1726358`, UID `1018`, ticks `2040606883`) was the
sole H0 compute process.

| Compact receipt | SHA-256 |
| --- | --- |
| `source.sha256` | `426453e539e6adbe732a58d6229231b7f0c2d3edf29d82d294b9b53cc0ee78cd` |
| `qualified-six.json` | `e85836e5e4099fed151e65ba1450aacc9c81a76b8f5edb6d32f6539a0f340afe` |
| `complete.json` | `146a0c7b2ceb5b998786c99e9f5ad32168234da6c0258c3657eefe8e8659f72c` |
| `receipts.ndjson` | `b95a7a3aec58ac20e2abe98b7ef5f4932e0a643674c6ac9469bc26096a1bdb02` |
| `returned.identity` | `762b41b3c4352979861910154456313927b8d6e11a1df5a9761b2d82959e83b5` |

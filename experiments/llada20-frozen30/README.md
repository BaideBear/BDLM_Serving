# LLaDA 2.0-mini and 2.1-mini: frozen30

Seventeen arms use the exact same 10 GSM8K, 10 MATH-500 and 10 HumanEval
requests, seeds and 2048-token output cap as the original LLaDA 2.1 experiment.
No strict Q cap, sample substitutions or retry of individual failed responses.
The primary metrics are mean request TPF, pooled token efficiency, and
correct / 30. Request Q excludes batch padding; physical Q includes it.

[Dense vs PrefixCache: all W=2/3/4 pairs on both models](flowblock-pairs.md) ·
[W=4 cost breakdown](flowblock-w4.md).

| Policy | LLaDA 2.0 setup |
|---|---|
| Original Native B32 | Actual checkpoint `generate`, 32 steps, M2T > .95, default EOS check at block end. `temperature=0` in this checkpoint still performs multinomial sampling. No T2T. Full visible prefix rows are **really recomputed** and are charged. The published 2.1 Native is a committed-prefix-cache implementation; native Q is not a weights-only comparison. |
| FlowBlock W=1/2/3/4 | Packed upstream controller, greedy top-1, M2T > .95, acceptance-ratio spawn .8, no T2T/post-edit. Counts physical verification/commit forwards. |
| FlowBlock PrefixCache W=2/3/4 | Same settings as dense, but skip each block's stable leading input positions only after encoding their final values. Preserve KV; exclude skipped rows before the transformer. |
| Optimus IBS and OBS, each fixed 4/8/16/32 | Original selected-position and unmasking controllers, top-1 M2T threshold .9, no T2T on 2.0. |
| B3 B32 | Repository's paper-derived entropy gate, at most one open future block, native 2.0 multinomial sampler at .95; committed KV refreshed on a real forward. Not an official B3 implementation. |

M2T thresholds and T2T differ across model versions; the cross-model study
compares these **configured inference systems**, not isolated weight changes.
Dense W=2/3/4 on 2.1 comes from the paired PrefixCache study when comparing
with PrefixCache, including W=4 which was absent from the original 13-arm figure.

```bash
python -m bdlm_serving run --experiment llada20-frozen30 --model llada20 \
  --checkpoint "$LLADA20" --samples "$FROZEN30" --output "$NEW_RUN" --batch-size 32
python -m bdlm_serving score --samples "$FROZEN30" --output "$NEW_RUN"
python tools/llada20_archive.py --run "$NEW_RUN"
```

Native runs request-by-request, using the original checkpoint's square additive
mask; batching is used for the adapted controllers. Only the latter use the
rectangular binary-mask adapter. The checkpoint manifest includes all seven
shard hashes. Weights, responses and cluster logs are not public artifacts.

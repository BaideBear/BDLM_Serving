# LLaDA 2.0-mini and 2.1-mini: frozen30

The original seventeen arms and the added cached B32 arm use the exact same 10 GSM8K, 10 MATH-500 and 10 HumanEval
requests, seeds and 2048-token output cap as the original LLaDA 2.1 experiment.
No strict Q cap, sample substitutions or retry of individual failed responses.
The primary metrics are mean request TPF, pooled token efficiency, and
correct / 30. Request Q excludes batch padding; physical Q includes it.

[Full results: 2.0 vs 2.1, plus the separate uncached HF reference](results.md) ·
[Three-metric comparison chart](figures/comparison.png) ·
[Original 17 arms](requests.csv) · [Cached B32 request and phase counts](../llada20-cached-b32/requests.csv) ·
[Summary by dataset](summary.csv) · [Matched model comparisons](comparison.csv).

[Dense vs PrefixCache: all W=2/3/4 pairs on both models](flowblock-pairs.md) ·
[W=4 cost breakdown](flowblock-w4.md) ·
[PrefixCache deltas and paired quality flips](prefix_deltas.csv).

| Policy | LLaDA 2.0 setup |
|---|---|
| Committed-prefix KV B32 | Same 2.0 checkpoint sampler and stopping rules as the HF reference. Only one live block is forwarded. Re-encode the final tokens on every block commit, including the last. Every decoding forward is measured at all 20 QKV projections. |
| HF generate B32 (uncached reference) | Actual checkpoint `generate`, 32 steps, M2T > .95, default EOS check at block end. `temperature=0` in this checkpoint still performs multinomial sampling. No T2T. Full visible prefix rows are recomputed and charged. The 2.1 native row is a different, committed-prefix KV implementation. |
| FlowBlock W=1/2/3/4 | Packed upstream controller, greedy top-1, M2T > .95, acceptance-ratio spawn .8, no T2T/post-edit. Counts physical verification/commit forwards. |
| FlowBlock PrefixCache W=2/3/4 | Same settings as dense, but skip each block's stable leading input positions only after encoding their final values. Preserve KV; exclude skipped rows before the transformer. |
| Optimus IBS and OBS, each fixed 4/8/16/32 | Original selected-position and unmasking controllers, top-1 M2T threshold .9, no T2T on 2.0. |
| B3 B32 | Repository's paper-derived entropy gate, at most one open future block, native 2.0 multinomial sampler at .95; committed KV refreshed on a real forward. Not an official B3 implementation. |

M2T thresholds and T2T differ across model versions; the cross-model study
compares these **configured inference systems**, not isolated weight changes.
Dense W=2/3/4 on 2.1 comes from the paired PrefixCache study when comparing
with PrefixCache, including W=4 which was absent from the original 13-arm figure.

**Native/B32 correction.** The 2.0 and 2.1 `native32` IDs in the historical
CSV name different execution paths. The table and chart now separate them.
2.0 HF `generate`: 14,651 delivered tokens / 4,302,240 query rows = 0.3405%.
This is **not** the efficiency of cached B32 on LLaDA 2.0. The authors'
[deployment notes](https://github.com/inclusionAI/LLaDA2.X#deployment-and-usage)
and [dInfer](https://github.com/inclusionAI/dInfer) describe cache reuse.
The [cached 2.0 native frozen30 rerun](../llada20-cached-b32/) supplies the like-for-like Q accounting. Summary tables use `native32_cached` for both cached baselines, and `native32_hf_uncached` for the distinct 2.0 reference. The historical raw CSV IDs are unchanged.

An independent CPU boundary probe with full BF16 weights observed QKV
projection inputs and outputs in all 20 layers. With 63 prompt tokens,
HF `generate` passed 64 then 96 rows. Replaying the same 96 input IDs using
the repository's rectangular-mask adapter required 64 rows to build the
committed prefix, then 32 live Q rows attending to 96 KV positions. The
active top-1 predictions matched at 32/32 positions on this step; logits
were not bitwise identical, and this is not an end-to-end quality test.
Reproduce with `PYTHONPATH=. python tools/probe_llada20_native_q.py "$LLADA20"`.

```bash
python -m bdlm_serving run --experiment llada20-frozen30 --model llada20 \
  --checkpoint "$LLADA20" --samples "$FROZEN30" --output "$NEW_RUN" --batch-size 32
python -m bdlm_serving score --samples "$FROZEN30" --output "$NEW_RUN"
python tools/llada20_archive.py --run "$NEW_RUN"
```

Both 2.0 Native runs execute request-by-request. Uncached HF uses its square additive
mask; cached B32 forwards only the needed rectangle from that mask. Other controllers
use batching. The checkpoint manifest includes all seven
shard hashes. Weights, responses and cluster logs are not public artifacts.

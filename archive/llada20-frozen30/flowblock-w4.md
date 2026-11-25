# W=4: dense vs per-block PrefixCache

Same frozen 30 prompts in each row, B=32, W=4 (128 packed rows before
retirement). Q is actual per-request query work, excluding prefill; physical Q
also charges batch padding. The full 17-policy study is indexed in [README](README.md).

| Model | Policy | Correct (GSM / MATH / HE) | Mean TPF | Token efficiency | Forwards | Q | Q/forward | Physical Q |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2.0-mini | Dense | 26 (9/8/9) | 4.061 | 1.97% | 6,291 | 805,248 | 128.00 | 805,248 |
| 2.0-mini | PrefixCache | 24 (9/7/8) | 4.044 | 2.07% | 6,799 | 781,041 | 114.88 | 848,386 |
| 2.1-mini | Dense | 25 (9/9/7) | 9.472 | 5.41% | 1,948 | 249,344 | 128.00 | 249,344 |
| 2.1-mini | PrefixCache | 22 (9/8/5) | 8.118 | 4.18% | 2,923 | 337,390 | 115.43 | 369,054 |

In 2.0 the cache removes query positions: at 6,799 forwards, a full dense
window along that same trajectory would use 870,272 rows. It actually uses
781,041. Against the independent dense run, the saving is only 24,207 rows
because the cache trajectory needs 508 additional forwards. Including padding
the cache run spends 43,138 more rows than the dense run.

The 2.0 dense and cache runs reached the generation limit on three and four
requests respectively; all were scored. The 2.1 rows come from the matching W=4
arms in the [existing PrefixCache archive](../flowblock-prefix/summary.csv),
not the baseline chart which omitted W=4. Caution: 2.1 and 2.0 have different
native decoding settings and the archived batching schedules differ.

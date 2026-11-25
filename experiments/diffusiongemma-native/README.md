# DiffusionGemma 26B-A4B: native frozen30

The adapter calls the official Transformers `DiffusionGemmaForBlockDiffusion.generate`
with the checkpoint chat template and sampler. Only the output limit is overridden
(2,048 new tokens, the common benchmark cap). Single-request batches make physical
Q and the native decoder counter independently auditable. No B32 alignment, prefix
retirement, forced Q cap, custom sampler, or per-question adjustment.

### 固定 30 题实测

| 数据集 | 答对 | TPF | Token efficiency | 实际 Q | 去噪 F | 重新编码 F | 官方 TPF | 达到长度上限 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **全部** | 28/30 | 24.284 | 8.42% | 116,224 | 431 | 23 | 25.546 | 0 |
| GSM8K | 10/10 | 20.254 | 7.73% | 36,096 | 134 | 7 | 21.453 | 0 |
| MATH-500 | 9/10 | 22.297 | 8.34% | 59,648 | 218 | 15 | 24.041 | 0 |
| HumanEval | 9/10 | 30.300 | 9.90% | 20,480 | 79 | 1 | 31.145 | 0 |

同一冻结题单、相同逐题 seed 和 2,048 token 上限。质量使用原评分器，HumanEval 在既有 Linux 沙箱执行。达到长度上限的回答原样计分，没有换题或重采样。表中的 TPF 都是逐请求的算术均值，效率是总交付 token 除以总计算 Q。首次 prompt 编码不计入分母；每次后续 encoder 重新编码均计入。每次送算 256 行 Q，并非每次都能交付 256 个 token。

[逐题审计](requests.csv) · [未舍入汇总](summary.csv) · [复现环境](provenance.json)。离线复核：`python tools/diffusiongemma_results.py`。

Use a separate Linux CUDA environment; do not upgrade the older baselines in place.
Install PyTorch 2.9 with CUDA, Transformers 5.12.0, and its processor dependencies.
Supply the local BF16 snapshot at the revision in `config.json` and the original
frozen `samples.json` (see [reproduction](../../docs/reproduce.md)).

```bash
python -m bdlm_serving diffusiongemma-native --checkpoint "$DG_CHECKPOINT" \
  --samples data/samples.json --output runs/diffusiongemma-native
# In the existing scoring environment, with bubblewrap:
python -m bdlm_serving score --samples data/samples.json --output runs/diffusiongemma-native
```

`N` counts nonspecial delivered tokens through first EOS. `F` and `Q` include
every decoder denoising call **and** encoder re-encoding of a previously generated
canvas. Prompt prefill is recorded separately. No final recompute is assumed.
The official `tokens_per_forward` uses non-padding generated IDs (including EOS)
and only decoder calls; we cross-check it per request and keep it separate from
the benchmark mean of `N/F`. Efficiency is pooled `sum(N)/sum(Q)`. Different
tokenizers, canvas sizes and work per forward preclude reading TPF as latency.

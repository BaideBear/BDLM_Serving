# DiffusionGemma 26B-A4B: native frozen30

The adapter calls the official Transformers `DiffusionGemmaForBlockDiffusion.generate`
with the checkpoint chat template and sampler. Only the output limit is overridden
(2,048 new tokens, the common benchmark cap). Single-request batches make physical
Q and the native decoder counter independently auditable. No B32 alignment, prefix
retirement, forced Q cap, custom sampler, or per-question adjustment.

**Status:** adapter and accounting tests available; no verified frozen30 scores published yet.

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

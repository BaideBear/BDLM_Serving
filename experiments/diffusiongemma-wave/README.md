# DiffusionGemma: FlowBlock and B3, frozen30

`block_size` counts tokens in one native canvas; `flowblock_windows` counts
canvases in the physical FlowBlock decoder window. The checkpoint used here
requires `block_size=256`. The historical LLaDA/SDAR FlowBlock experiments
retain their own `flowblock_block_size=32`. Changing the window never changes
the native canvas length.

```bash
python -m bdlm_serving diffusiongemma-wave \
  --checkpoint "$DG_CHECKPOINT" --samples "$FROZEN30" --output "$NEW_RUN" \
  --block-size 256 --flowblock-windows 1 2 3 4 \
  --native-reference "$NATIVE_RUN"
python -m bdlm_serving score --samples "$FROZEN30" --output "$NEW_RUN"
```

The JSON defaults are in `config.json`; `include_b3` enables a separate B3 arm.
CLI overrides are recorded in `run.json`. Use the native baseline's generation
environment and the repository's separate scoring environment. Store raw runs
on the cluster, not in this repository.

## Results

Same frozen 30 requests (10 each GSM8K, MATH-500, HumanEval), native sampler,
per-request seeds, batch one, 2,048-token limit. No replaced or length-capped
requests. W=1 matched native output IDs and encoder/decoder forward events for
all 30. TPF is the mean of per-request delivered tokens / generation forwards;
token efficiency is total delivered tokens / total measured generation Q.
Only prompt prefill is excluded. These are not wall-clock throughput metrics.

| Policy | Correct (GSM/MATH/HE) | TPF | Token efficiency | Total Q | Q/forward | Mean active slots per decoder call |
|---|---|---:|---:|---:|---:|---:|
| FlowBlock W=1 | 28 (10/9/9) | 24.284 | 8.42% | 116224 | 256.0 | 1.000 |
| FlowBlock W=2 | 29 (10/9/10) | 23.893 | 4.21% | 240640 | 498.2 | 1.109 |
| FlowBlock W=3 | 29 (10/9/10) | 23.290 | 2.77% | 367616 | 741.2 | 1.230 |
| FlowBlock W=4 | 29 (10/9/10) | 23.342 | 2.09% | 485888 | 983.6 | 1.248 |
| B3 | 27 (10/8/9) | 26.252 | 8.30% | 120832 | 291.2 | 1.146 |

These are model-specific controller adaptations, not official fused engines.
FlowBlock computes the entire packed window (including unused slots); only
opened canvases sample tokens. The spawn signal is the fraction of positions
not accepted in the previous step which the native entropy-bound sampler
accepts now, with a 0.8 threshold. Open canvases have bidirectional attention
within a block and block-causal attention across blocks. B3 uses the repository's
paper-derived entropy gate, with one optional future canvas; it computes the
second slot only when opened. Encoder re-encoding is charged for both policies.
`active` counts opened, not-yet-committed slots; it is not a physical Q counter.
Only 30 requests were tested, so one additional correct answer should not be
interpreted as a general quality improvement.

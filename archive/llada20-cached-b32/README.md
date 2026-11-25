# LLaDA 2.0-mini: Native B32 with committed-prefix KV

[Results and phase totals](../llada20-frozen30/results.md) · [Annotated comparison figure](../llada20-frozen30/figures/comparison.svg) · [30 numeric request records](requests.csv).

| Frozen subset | Correct | Mean request TPF | Token efficiency |
|---|---:|---:|---:|
| GSM8K | 10/10 | 2.298 | 6.45% |
| MATH-500 | 10/10 | 2.763 | 8.05% |
| HumanEval | 9/10 | 5.272 | 14.30% |
| All | 29/30 | 3.444 | 8.35% |

12,764 delivered tokens; 4,348 denoise and 428 commit forwards; 152,832 measured decoding Q. The 30 prompt-only prefills (3,008 Q) are excluded. Every request satisfied `Q_i = 32 F_i`; maximum visible KV was 2,144. Zero failures, one length-capped response retained. Token-ID recount and independent rescoring agreed with the formal records. Five of 30 cached outputs were token-identical to the uncached HF run with the same prompts and seeds. This is not a bitwise-equivalence or population-level quality claim.

Run only `native32_cached`. Use the original model `generate` loop, original per-request seeds, sampler, `.95` threshold, 32 steps per block, EOS at block end and 2048-token cap. No greedy substitution, T2T, or skipping tokens inside a live block. No row from the prior uncached run is used to estimate a new result.

1. Encode complete prompt blocks once in a prompt-only prefill. A partial prompt block remains part of the 32-row live block.
2. Denoise all 32 live positions with their absolute position IDs, using the checkpoint's own visibility mask and previously committed KV. Discard tentative KV after each forward.
3. Commit each block from the final sampled token IDs. Charge one real 32-Q forward per commit, including the terminal block, as in the archived 2.1 baseline.
4. Check fused QKV input/output widths in all 20 layers on every forward. Count positions **once per model forward**, not once per layer. Here physical Q equals request Q because this run executes one request at a time without padding.

Mean request TPF is `mean(N_i / F_i)`; pooled token efficiency is `sum(N_i) / sum(Q_i)`. `F_i` includes denoise and commit; pure prefill is reported separately. `N_i` counts delivered non-special tokens up to EOS, not sampling events. Cross-version comparisons retain each checkpoint's own decoding policy and are not comparisons of weights alone.

```bash
python -m bdlm_serving run --experiment llada20-cached-b32 --model llada20 \
  --checkpoint "$LLADA20" --samples "$FROZEN30" --output "$NEW_RUN"
python -m bdlm_serving score --samples "$FROZEN30" --output "$NEW_RUN"
python tools/llada20_archive.py --cached-run "$NEW_RUN"
```

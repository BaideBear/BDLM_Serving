# DiffusionGemma 26B-A4B: fixed Optimus controls

Seven nominal chunk sizes per mode (`4,8,16,32,64,128,256`), one frozen 30-request manifest per configuration. `B=256`, maximum 2,048 generated positions. IBS never queries across the current canvas boundary. OBS may see up to `B+C` positions, and carries its prefix state and resident decoder KV into the next window. Selection uses the pinned upstream `DelayedCacheState`; the actual number of Q rows is **not** clipped to `C`.

This is an Optimus **control adaptation**, not native Gemma with only a smaller window: the decoder uses upstream dynamic sparse unmasking (argmax, confidence 0.9, at least one unresolved row when possible). Gemma has no MASK input token; each request starts with identical seeded random canvases across arms, plus a separate monotone resolution bitset. Re-noise is restricted to selected unresolved rows, no T2T. Raw previous logits provide the self-conditioning signal for previously visited positions. Fresh positions get a zero soft embedding. We do not use the native entropy/temperature sampler or its stability stop, so IBS C=256 need not reproduce Native B256.

Only selected decoder rows run through the model, including the output head. Every decoder attention layer checks its projected Q width. All previously written decoder KV through the rightmost selected position within the active region is available; no uninitialized holes are allowed. Finalized full canvases are re-encoded into Gemma's native encoder cache before continuing. The measured F/Q include those re-encodes and all decoder forwards, including prefix retirement; only pure prompt prefill and the separate qualification probe are excluded. Stop after a resolved canvas contains EOS or reaches the cap. Batch size is one; padded Q equals actual Q.

```bash
python -m bdlm_serving.runtime.diffusiongemma_optimus \
  --checkpoint "$CHECKPOINT" --samples "$SAMPLES" --output "$RUN_DIR" \
  --modes ibs obs --chunks 4 8 16 32 64 128 256
python -m bdlm_serving score --samples "$SAMPLES" --output "$RUN_DIR"
```

Use the pinned DiffusionGemma native generation environment, and the repository's separate Linux scoring environment for the second command. Before formal measurement the worker verifies bit-identical native vs adapted first-pass logits on identical selected inputs. Scoring uses the same frozen GSM8K, MATH-500 and HumanEval protocols as other baselines. The blank CSV in `tools/templates/` is a plotting aid, not evidence of a completed sweep.

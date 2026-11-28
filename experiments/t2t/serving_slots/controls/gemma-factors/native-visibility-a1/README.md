# Gemma Q256: first divergent operation and one representation change

The control is the public `dense_stop_joint_verified_eos` single-request handle
held at Q256. The candidate calls the exact same handle; after every real WaveKV
write it passes `None` instead of a materialized all-true SDPA mask **only** for
unrestricted, unpadded, dynamic-cache, single-block bidirectional attention.
It does not bypass WaveKV, change row order, reuse logits, alter acceptance,
stop on all-accepted as a surrogate for native stop, or suppress ordered
verification or any post-prefill encoder. The production profile is untouched.

## Source and existing measurements

* Installed `generation_diffusion_gemma.py` SHA256
  `cded57b7e365f0265b1c088e993c358b5f70acdf48055fce396540b4d9d691bc`:
  `_prepare_denoiser_inputs` initializes a canvas from the current device RNG.
  `_denoising_step` draws, accepts, renoises, evaluates the processed-logit
  stopper, then stores BF16 SC. `generate` delivers the last argmax, not a
  further physical forward over the updated canvas.
* Installed `modeling_diffusion_gemma.py` SHA256
  `69d30fd9842cdbe0c64f508e13c709ea51f2889fe694148ec7c397a038e2cdce`:
  decoder attention is bidirectional; `create_diffusion_decoder_attention_mask`
  returns per-type `None` for an unrestricted unpadded dynamic cache.
* `gemma_fixed_dual._project` uses `wave_attention`. `WaveKV.extend` writes
  per-layer K/V then materializes explicit prefix/live visibility, even if
  every key is visible. Equal permitted edges do not ensure bitwise-equal
  SDPA output. The experiment tests the mask argument, not a particular
  backend kernel (no kernel profiler).
* Archived W1 is an IDs-and-events-qualified native proxy (`run_wave` checks
  both before writing). It ran Torch 2.9.0+cu128; our same-A1 native generate,
  off and on all run 2.6.0+cu124. Do not assign the whole published gap to
  one cause. The historical 1,520 vs 1,397 F difference is exactly
  (1,268 strict denoising - 1,316 native denoising) + 171 strict verifies +
  (81 - 81 encodes) = 123. Native vs off delivered IDs first differ in the
  first block on 78/90, later on 2, never on 10. An existing draft-shadow
  study already found 22/171 blocks differed between draft and verified
  delivery and four differed at encoder input: no draft-as-verify shortcut.

## Direct evidence on A1

The preregistered anchors (0,10,20) compared `model.generate`, off and on in
one resident model. First forward, reverse clock 48: initial IDs, positions,
empty SC and layer-0 Q/K/V hashes match between off and native, but the first
SDPA output already differs on all three. Native passes `None` on 30/30
layers; off does so on 0/30. With the candidate, all 30 layer outputs and the
full FP32 raw logits and processed scores match native byte for byte on all
three. The matching first-head denoising sequences (IDs, sampled IDs,
accepted masks and argmax) are 8/8, 9/9 and 4/4 rounds; all three
verified first-block deliveries match. See `anchors.json`.

The mandatory extra verify is still a different action. Also, on both
anchors that open a second block (0,10), the encoded first block matches
native and the next positions match, but the next initial canvas already
differs **before denoising**. `StrictRequest._start_block` seeds an independent
generator with `seed+ordinal`; native continues the request RNG. Do not infer
whole-request parity, or bundle an RNG change into this mask experiment.

## Reproduce and integrate

The sole worker (A1 PID 240872) ran `bash /gs/runtime/gemma-native-visibility-a1-1128/launch.sh`.
The launcher reruns the prior 21-case `strict_handle_report`, checks exit of
the prior PIDs, checks A1 UUID/0 MiB/no compute processes, and caps runtime at
1795843443. No second worker. Use the local `registration.md` for the
non-quality-based 3-to-90 gate. `pairs.csv`, `cohorts.csv` and `anchors.json`
are compact; raw ID lists and detailed hashes stay under
`/gs/results/gemma-native-visibility-a1-1128`.

Only proposed production insertion point: the return-mask branch of
`WaveKV.extend`, **after** all K/V writes, with the same bidirectional,
unrestricted-prefix, dynamic solo-head and single-live-block proof. Keep
explicit masks for a parked/multiple block, real native mask, static/padded
cache or unknown geometry. The experiment intentionally still constructs and
tests `.all()` on the mask; its timing is not a latency result. Other offers
can use the predicate, but neither sparse behavior nor switching is qualified
by this fixed-Q study. Files outside this folder: `gemma_native_visibility.py`,
`gemma_native_visibility_eval.py`, `gemma_native_visibility_report.py`,
`tests/test_gemma_native_visibility.py`.

## Sealed frozen90 decision

All 90 independent same-A1 triplets completed (270 requests; anchors counted
once). All off arms replayed historical strict IDs, score, N, F, encoder F
and decoder Q. All measured 85 projection module widths were 256 per
decoder forward. All emitted blocks were checked against actual physical
argmax; all strict encodes were checked against the verified block.

| Cohort | Correct off/on/native A1 | Macro TPF off/on/native A1 | TE off/on | Exact first block vs A1 native off/on | Exact delivery vs A1 native off/on |
|---|---|---|---|---|---|
| Original 30 | 28/29/29 | 23.258/23.181/26.770 | 8.06%/8.42% | 3/25 | 3/12 |
| New 60 | 55/55/55 | 24.749/23.572/25.358 | 9.01%/8.96% | 7/54 | 5/28 |
| Frozen 90 | 83/84/84 | 24.252/23.442/25.829 | 8.69%/8.78% | 10/79 | 8/40 |

TPF is mean per-request N / all post-prefill F. TE is sum(N) / sum(selected
history-reading decoder Q), excluding encoder Q. Off/on bills are N
32,006/30,897, total F 1,520/1,452 (encoder 81/77), decoder Q
368,384/352,000. Less total F does not imply a higher macro TPF: output
lengths and the per-request ratios changed. Paired stratified 95% bootstrap
intervals (candidate minus control): macro TPF -0.810 [-1.797, +0.181];
correct +1/90 (2 rescues, 1 loss), fraction interval [-0.022, +0.044].
This is a strong numerical fidelity result, not a proven efficiency or
quality win. Retain for opt-in semantic alignment, not default performance.

Historical native was 85/90, 25.510 macro TPF; same-stack native was
84/90, 25.829, with only 5/90 matching the historical proxy on both IDs
and F. No controlled torch-version-only experiment was conducted. The one
point of historical quality not seen even in the same-device native control
must not be attributed to mask representation. Both same-card native and
on getting 84 does not mean they solved the same cases or generated the
same answers.

This worker exited normally; the separate [continuous-RNG experiment](../continuous-rng-a1/README.md)
uses this result as its fixed control. At
2026-11-28 10:11:53 +08 physical GPU1 had 0 MiB and no compute process.
The four compact files were checked against the remote SHA256s. Two
no-model tests passed. No public profile, charts or Git index modified.

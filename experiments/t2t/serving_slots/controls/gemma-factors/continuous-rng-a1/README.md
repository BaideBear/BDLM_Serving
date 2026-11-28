# Gemma fixed Q256: one continuous request RNG

Both arms use the qualified, single-block visibility representation from
`native-visibility-a1` on the existing strict handle. The only treatment is
which generator supplies random numbers on the next sequential block, *before*
the one real canvas initialization draw. Off creates a block-private generator
with `seed + ordinal`. On seeds the first one identically and keeps that private
generator for the request. Neither arm reads the global RNG to synchronize,
advances a sampler during verification, or draws a replacement canvas twice.

The installed upstream `generation_diffusion_gemma.py`, SHA256 in provenance,
initializes with `torch.randint` at line 394 and consumes the device stream for
multinomial at 1045 and renoising thereafter. It does not reset the generator
when resetting the adaptive stopper at a block boundary. A private generator
reproduces this progression only while the draws and their shapes agree; it
cannot repair a previous denoising-step mismatch.

## What was matched

* Same A800 physical GPU1, Q256, Torch 2.6.0+cu124, Transformers 5.12.0,
  checkpoint, prompt digests, scorer, seeds, physical decoder projections,
  writeback and acceptance rules. Alternating order, no separate native run.
  The native reference is the previous identical-stack A1 run, **not** W1
  on another Torch version. The latter is not a valid RNG-only control.
* Regenerated all four prior compact files and checked their published
  hashes before inference. Each fresh off receipt is checked against the
  entire prior visibility-on object, including IDs, N, F, encoder F, selected
  decoder Q, score, masks, and anchor-only native-style layer/tick tapes.
  The two fresh arms additionally record a hash of every step's complete
  event, resulting sampler IDs and acceptance mask, RNG bounds and each
  initial canvas. No per-tick native tape exists beyond the three anchors.
* First-block event sequences, canvas and verified delivery must agree
  between fresh arms on every case. The strict runner checks each physical
  projection (all 85 hooks), input changes and acceptance writes, one real
  full-width verify and the exact block sent to each post-prefill encoder.
  A return from a verify can also open the next canvas: its draw is billed
  to the new block, not attributed to sampling on the verify.

## Anchor falsification

On #0 and #10, the first encoded block was unchanged, the next positions
matched, and the second initial canvas went from different to bit-identical
to native. On #0 all six second-block denoising steps and the delivered
block then matched native. On #10 the first eight steps matched, but strict
had all 256 positions accepted after step eight and performed verified
delivery where native took a ninth denoising step. Delivered IDs happened
to agree. **All accepted is not the native stop predicate.** On #20 there
was no second canvas and the entire pair was identical.

## Reproduction and scope

`registration.md` is the prelaunch decision and no-go contract; `launch.sh`
is the exact A1 launcher. `pairs.csv` contains 90 index-aligned pairs,
hashed provenance, bills and output-difference positions, but no raw tokens
or prompts. `cohorts.csv` splits original30, new60 and frozen90; anchors
are included once. Macro TPF is the mean of per-request N/all post-prefill
F. Canonical TE is total N / total selected history-reading **decoder** Q,
not encoder-inclusive. All post-prefill encoder calls here have Q256.
Intervals are paired, stratified by dataset, 4,000 resamples using seed 1128;
they are descriptive uncertainty on this frozen evaluation, not replication.

Run `bash /gs/runtime/gemma-continuous-rng-a1-1128/report.sh` on A800 to
recompute compact numbers without a device. Raw receipts stay at
`/gs/results/gemma-continuous-rng-a1-1128`. New source files next to the
other experiment runners: `gemma_continuous_rng{,_eval,_report}.py`;
`tests/test_gemma_continuous_rng.py` exercises initialization and failed
audit paths without a model. The initializer's module-local adapter is for
one serial experiment worker; for an API, pass an explicit request-owned
generator at `StrictRequest._start_block` instead. Overlapping blocks, a
change of offer, concurrent requests or public API compatibility were not
qualified. Mandatory dirty repayment, measured terminal argmax and required
encoder calls remain mandatory regardless of budget.

## Sealed decision

Keep as an **opt-in native-alignment candidate for serial fixed Q256**. Do not
promote for speed, call it native-equivalent, or infer a gain at other offers.
The predeclared semantics endpoint improved: 28 newly exact deliveries and
zero lost versus same-A1 native (+31.1 percentage points, paired stratified
95% interval [+22.2,+40.0]). First-block native agreement stayed 79/90;
the later-only mismatches fell from 39 to 11. Of the 22 remaining mismatches,
11 start in the first block, outside the reach of this intervention.

| Cohort | Exposed requests | Correct off/on | Macro TPF off/on | TE off/on | Exact delivery vs native off/on |
|---|---:|---:|---:|---:|---:|
| Original 30 | 17 | 29/29 | 23.181/23.415 | 8.421%/8.577% | 12/23 |
| New 60 | 40 | 55/56 | 23.572/23.595 | 8.957%/8.969% | 28/45 |
| Frozen 90 | 57 | 84/85 | 23.442/23.535 | 8.778%/8.840% | 40/68 |

90/90 full-row off replays matched. All 90 first heads matched between arms
including every step's event hash; all 33 unexposed requests were identical
throughout. There were 77 later openings off and 78 on (every on opening
continued the previous state). Whole output IDs changed on 40 of 57 exposed
requests. The one rescue was Math500 index 50, with zero losses; the paired
95% interval on quality difference is [0,+3.33] percentage points. Same-A1
native scored 84, at 25.829 macro TPF; index 51 was scored correct by on
and incorrect by native. Neither score equality nor superiority means parity.

Off/on bills: N 30,897/31,027; total post-prefill F 1,452/1,449; sample F
1,208/1,203; verify F 167/168; encoder F 77/78; selected decoder Q
352,000/350,976. All 85 measured projection widths were 256. Original30
saved 9 F, new60 spent 6 more. Macro TPF delta +0.094, 95% interval
[-0.192,+0.382]. The 25 extra generation F versus same-stack native are
1,203 - 1,343 denoising F, plus 168 measured verifies and 78 - 81 encoder
F. These are route counts, not permission to remove the verifies. The
remaining alignment issue is directly visible on #10; assess the difference
between local all-accepted and native stopping separately, without changing
the terminal contract.

A1 worker PID 258557 exited 0. Physical A1 had 0 MiB and no compute process
at 2026-11-28 10:32:40 +08; no follow-up worker or holder. Five no-model tests
passed on Mac and A800. Offline validation and the four compact SHA256s agreed
between A800 and Mac. No raw receipt downloaded, no public profile or index
change and no Git commit or push by this experiment.

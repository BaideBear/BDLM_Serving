# Gemma strict IBS Q128 + certified paid-EOS: paid-agreement six-case pairing

Status: registered for A800 physical A3. Six fixed cases; no automatic 90.
Logs stream to n164; no holder and no card occupation beyond the run.

## Shared rule

`bdlm_serving/methods/t2t/paid_agreement.py: EvidenceLedger(floor=0.5)`.
The adapter passes only this tick's measured unresolved rows:
`observe(*, block, tick, positions, proposals, probabilities, native,
excluded_ids)`. `native` is the base sampler acceptance, equal length to
`positions`. An extra acceptance requires the same physical block and absolute
position in two adjacent actual decoder ticks, both selected MASK, the same
measured argmax ID both times, each selected probability FP32 > 0.5, base
sampling still rejecting, and the token not in the model special IDs.

## Arms

All three use the same model load, seed and `Stop(arm="paid", offer=128)`:

1. stock paid-close: the certified paid-EOS loop, no agreement observation;
2. observer-only paid-close: the same loop plus the ledger; the would-be extra
   rows are recorded but never applied;
3. apply paid-close: the same loop plus the ledger; extra accepted rows are
   written into the base IBS unmask result before the stock `take`/noise write.

The adapter reuses the certified-EOS `_PLAN`/`_AFTER` hooks and adds one
`_UNMASK` hook around the existing IBS unmasker. The noise draw shape is
unchanged, so all RNG, including the noise draws of the extra accepted rows,
is consumed exactly as before. Added rows become normal admissions and enter
the original admission debt; no resolved-row T2T is introduced and this is not
a final-model claim.

## Six cases and first-fork gate

Cases: `7,13,25,33,55,71`. Arm order alternates per case. The gate:

- stock and observer-only paid-close must be identical on IDs, trace, decoder
  tape, encoder tape, states, bills, RNG and score;
- apply must match observer until the first extra admission;
- at that tick the selected positions, offer and Q are unchanged;
- the trace `admitted` is exactly observer `admitted` + the extra positions;
- the physical take count is observer take + the extra count, each extra row
  accepted exactly once;
- the stock `reconcile` must pass for every arm, including the apply arm's
  added debt.

If the first-fork gate fails, stop; do not expand to 90.

## Files

- `experiments/t2t/serving_slots/gemma_paid_agreement.py` (adapter)
- `experiments/t2t/serving_slots/gemma_paid_agreement_eval.py` (three-arm eval)
- `tests/test_gemma_paid_agreement.py` (model-free contracts)
- `tools/launch_gemma_paid_agreement_q128.sh` (A3 launcher)
- this directory (registration and compact results)

## Results

Six cases on A800 physical A3, Q128, gate **passed**.

| Case | First extra (tick/block/positions) | Apply D/E/F/Q | Apply adm. | Apply correct/N | Stock D/E/F/Q | Stock adm. | Stock correct/N | Stock=observer |
| ---: | --- | --- | ---: | --- | --- | ---: | --- | --- |
| 7 | 3/0/[57] | 11/0/11/1267 | 256 | true/182 | 12/0/12/1423 | 256 | true/184 | identical |
| 13 | 2/0/[26,27,38] | 12/0/12/1150 | 256 | true/251 | 22/1/23/2228 | 405 | true/274 | identical |
| 25 | 5/0/[171,192,193,200,204] | 8/0/8/962 | 256 | true/224 | 8/0/8/962 | 256 | true/224 | identical |
| 33 | 1/0/[9] | 15/1/16/1599 | 384 | true/254 | 18/1/19/2076 | 396 | true/265 | identical |
| 55 | 3/0/[42,111] | 22/1/23/2249 | 496 | true/365 | 28/1/29/3011 | 512 | true/398 | identical |
| 71 | none | 9/0/9/1025 | 230 | true/99 | 9/0/9/1025 | 230 | true/99 | identical |

- Stock and observer-only paid-close are identical on IDs, trace, decoder
  tape, encoder tape, states, bills, RNG and score in 6/6 cases. The observer
  arm is behavior-neutral.
- The first extra admission is exactly the added row(s) at the first fork:
  **12 first-fork positions** (1+3+5+1+2+0), not the full-run extra-admission
  total. At that tick the selected positions, offer and Q are unchanged; the
  trace `admitted` is observer `admitted` + the extra positions; the physical
  take count is observer take + the extra count, each extra row accepted
  exactly once. The stock `reconcile` passed for every arm, including the
  apply arm's added debt.
- Full-run extra admissions over the six cases: the apply arm actually
  applied **25**; the observer's would-be extras total **81**. Per case
  (apply/observer): 7 3/9, 13 10/40, 25 5/5, 33 3/6, 55 4/21, 71 0/0. The
  apply trajectory diverges after the first extra, so later would-be extras
  differ; do not read 12 as the total extra payments.
- The apply arm's post-extra trajectory may diverge: final F/Q/admissions/N
  differ in 5/6 cases. Cases 25 and 71 finish with the same final bill as stock
  (case 25 has five extra rows at tick 5; case 71 has no extra). All six apply
  arms remain correct; this is a six-case pairing, not a 90-case quality or
  TPF result.
- The noise draw shape is unchanged; the pre-extra state and RNG match. The
  RNG after the first extra may differ with the changed trajectory. No auto 90.

Archive: n164
`/data1/project_data/gx-erqw-20261123/results/gemma-paid-agreement-q128-01/`
with the full `summary.json` (`e64094d78fe60aaae0dbbd16d075a3f06151b6ab8a2d2c87a234b12b0942f151`),
`summary.compact.json` (`ad08de6a2d50c07c0f100cef97f1e5061af0ae41340db0225237ca62f90c0464`),
per-case JSON, `stdout.log` and `stderr.log`. This directory keeps the compact
summary, `audit.json` and this README.

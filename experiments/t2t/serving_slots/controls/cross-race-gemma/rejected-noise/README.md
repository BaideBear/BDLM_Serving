# Fixed Gemma: retain previously rejected noise

One paired ablation, only on A800 GPU6: fixed head-only (`00`), physical
C32/64/128, right-edge prefix 1/8, no paid-mask reuse, no verify fusion and no
fullstop. This is not an evaluation of full-head admission at C=B.

The baseline draws candidate IDs, runs the native acceptance rule and draws
full-shaped random noise. Both arms do all three. On a previously **sampled**
row that was rejected before and is rejected again, `on` chooses the old ID
instead of the fresh noise ID. This changes the sampler transition, not just its
cost. First visits, new acceptances, edits to accepted IDs, revocations, the SC
update and terminal verification keep the baseline rules. RNG use is identical
for the same call geometry; trajectories and future call geometries need not match.

**Dirty contract:** subtract physically projected rows from prior debt, then
union all actual token value changes and all acceptance flips. Changing the chosen
ID can avoid creating new debt; it cannot cancel outstanding debt, forgive an
acceptance-only flip, or skip a verify. Counts of `avoided` are same-step
counterfactuals against the actual `on` trajectory's discarded draws, not `off`
minus `on` debt or a count of forwards saved.

## Sealed results

540/540 arm receipts, 270/270 matched pairs, and all 270 off arms exactly
replayed their corresponding unfused historical controls on every field listed
in `provenance.json`. The three qualification questions are part of the 90, not
additional trials. No failed or missing terminal receipts. Values below are
off to on; TPF differences are paired means calculated before rounding.

| Cohort | C | Total F | Drain F | Correct | Delta macro TPF |
|---|---:|---:|---:|---:|---:|
| frozen90 | 32 | 7161 to 6865 | 1581 to 1498 | 85 to 83 | +0.0745 |
| frozen90 | 64 | 4113 to 4260 | 888 to 858 | 83 to 83 | -0.0448 |
| frozen90 | 128 | 2629 to 2622 | 520 to 499 | 85 to 84 | +0.2727 |
| original30 | 32 | 2351 to 2279 | 514 to 495 | 29 to 28 | +0.0267 |
| original30 | 64 | 1369 to 1443 | 295 to 280 | 29 to 29 | -0.1520 |
| original30 | 128 | 884 to 916 | 173 to 163 | 30 to 28 | -0.0790 |
| new60 | 32 | 4810 to 4586 | 1067 to 1003 | 56 to 55 | +0.0984 |
| new60 | 64 | 2744 to 2817 | 593 to 578 | 54 to 54 | +0.0088 |
| new60 | 128 | 1745 to 1706 | 347 to 336 | 55 to 56 | +0.4486 |

Keep this opt-in. All three tiers reduced actual dirty row writes and drain
calls. C32 improved paired macro TPF (95% interval +0.0181 to +0.1355),
but the on arm emitted 827 fewer non-special tokens and had one score rescue
and three losses. At C64 the 30-call reduction in drains was outweighed by
177 additional sparse calls. C128 saved only seven total forwards; its
TPF interval, like C64's, crosses zero. None establishes a quality-neutral
throughput improvement or a reason to change C=B.

## Files and denominators

- `paired.csv`: one row per question and width. Includes historical control
  equality, both arm scores and ID equality, the first effective exposure, cold/warm
  eligibility denominators, actual token changes and flips, debt-causing row
  writes, and physically paid work. `warm_rows` are sampled rows that have been
  sampled previously in the same block; `sampled_rows - warm_rows` are first
  visits. Empty first-exposure fields mean no effective change, not missing data.
- `cohorts.csv`: original30 = indices 0-29; new60 = 30-89; frozen90 = their union.
  Each subset is balanced across GSM8K, MATH500, and HumanEval. Do not count the
  union as a third independent sample. `F` counts post-prefill decoder and encoder
  calls; `drain_F` counts only dedicated decoder repayment calls. Macro TPF is the
  mean of each question's non-special output tokens / F, not pooled tokens / F
  or measured wall-clock throughput. Intervals are 4,000 paired within-dataset
  bootstrap draws (seed 1128), not simultaneous intervals or a seed sweep.
- `provenance.json`: GPU/checkpoint/software/source fingerprints, original sealed
  qualification, historical-comparison fields and receipt locations. The producer
  replayed per-tick writes and terminal verification before sealing each arm;
  `gather.py` checks sealed counters and compares actual final IDs against the
  historical control on the receipt host. It does not reconstruct a tick tape
  from aggregate counters or rescore generated text.

## Recompute from sealed receipts

On A800, using the registered source snapshot; standard library only, no model:

```sh
PYTHONPATH=/gs/runtime/reject-noise-1128/repo /usr/bin/python3 -B \
  /gs/runtime/reject-noise-1128/gather_cohorts.py \
  --root /gs/results/reject-noise-1128 --history /gs/results/paid-close-1128 \
  --out /gs/runtime/reject-noise-1128/cohorts
```

`gather_cohorts.py` is this directory's `gather.py`. Missing pairs, control
mismatches, inconsistent bills or mixed registrations fail the gather before
export. Only the three compact files return from A800, never prompts, token IDs,
generations, traces or raw receipts.

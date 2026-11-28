# Q32 capped IBS is a diagnostic, not the matched control

Fixed90 historical counters; each head includes the last head of a request.
The last two columns normalize by *each method's own* number of heads.
For IBS, a resolved canvas is emitted without a separate ordered argmax pass.

| Cohort | IBS MASK-work / replay-only F | IBS decoder F / heads | Strict decoder F / verify F / heads | IBS decoder F/head | Strict non-verify decoder F/head |
|---|---:|---:|---:|---:|---:|
| Original 30 | 1,578 / 55 | 1,633 / 53 | 2,221 / 344 / 56 | 30.81 | 33.52 |
| New 60 | 4,395 / 166 | 4,561 / 130 | 4,573 / 734 / 114 | 35.08 | 33.68 |
| All 90 | 5,973 / 221 | 6,194 / 183 | 6,794 / 1,078 / 170 | 33.85 | 33.62 |

The strict decoder difference of +600 F is +1,078 verify F and -478 other
decoder F. Post-prefill encoders are 80 vs. 93; total generation F differs
by +587. The cohorts reverse sign on non-verify F per head. Excluding a
required verification to make this comparison does **not** make it removable.
These numbers do not align warm start states, output lengths, or accuracy.
The strict run delivers 32,774 non-special tokens and scores 85/90; capped IBS
delivers 36,118 and scores 79/90. No IBS results are pooled into the GPU arms.

The IBS source physically projects the plan, records projected SC, samples
only previously unresolved rows by argmax/dynamic admission, and checks that
every MASK-work call admits at least one row. It re-noises rejected selected
rows but its cap debt only adds admissions. Its 221 replay-only F are not a
warm subset. The strict selected entropy sampler permits T2T/revocation and
bills changes to input IDs and acceptance. Its 1,576 pure drains are 1,190
cold, 221 warm and 165 closing. Making even *all* these drains free on fixed
historical trajectories would yield macro TPF 6.478 vs. IBS 6.840. No
available Q32 IBS receipt here identifies an exactly aligned warm-only gap.

This experiment isolates sampler consumption of a *mixed, already planned*
warm payment, allowing a direct check of what the extra T2T opportunities
did. It does not adopt IBS admission, SC or close semantics. In particular,
it differs from `warm_debt_resample`, which **adds** a sample and clock to
a previously pure drain and is never eligible on the mixed call studied here.

References within the checkout:

- `bdlm_serving/runtime/diffusiongemma_optimus.py`, `gemma_prefix_capacity.py`
  and [capped IBS stages](../../gemma-prefix-capacity.csv).
- [Strict phase counts](../first-tile-kv-q32-a1/cohorts.csv), [fixed-trajectory
  drain bound](../first-tile-kv-q32-a1/low-q-decision.md).
- [Warm-fewer-fail negative](../warm-fewer-fail-q32-a1/README.md): 3 paired
  anchors, 65 contested ticks, no different row chosen; no 90 estimate.
- [First-tile KV negative](../first-tile-kv-q32-a1/README.md): 90 paired,
  85 to 82 correct, macro TPF 4.879 to 4.580, despite one fewer cold F/head.
- Both old controls, tests, registered source and compact evidence are in
  `/tmp/gemma-q32-previous-negatives-1128-compact.tar.gz` (SHA-256
  `6153ead29fff7ffd568eddee66299b2ca8eae32a0562f08e479d7d47f5016422`).
  No change to either old launcher and no raw per-request logs in the archive.

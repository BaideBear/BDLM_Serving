# Q128: relax joint entropy admission only after full head initialization

## Why this factor

On frozen90 the fixed Q128 control paid 2,430 post-prefill forwards vs.
2,042 for historical IBS, with 30,351 vs. 30,610 delivered non-special
tokens and 88 vs. 79 correct. The fixed control's disjoint stage bills from
the cold-pacing off audit were 492 cold, 1,415 warm (1,239 sampled and 176
drain), 121 closing drain, 328 verification and 74 encoder F. Warm work is
the largest testable target; this does not identify 1,415 wasted calls or
establish the same phase boundary in IBS. Q32 was a less clean target:
IBS has higher macro TPF despite paying *more* F (7,849 vs. 7,156) and
producing more tokens.

On A800 GPU1 the paired experiment changes exactly one scalar: on a
genuinely sampled sparse step whose current head had already initialized all
256 positions before this call, change the selected-row joint entropy bound
from 0.10 to 0.20. This is a cumulative bound on the selected set, not an
independent per-token confidence threshold. The first visits, the tick that
finishes initialization, pure drains, and terminal verification use the
unchanged policy. Both arms use the 1/4 right-edge law, no successor, strict
debt, seed, sampler clock, self-conditioning, re-noising, T2T, and guarded
first verification of `fixed90_fused`. The random draw procedure is
unchanged, not necessarily subsequent draws or outputs after divergence.
Admission changes can retain a previously accepted row as well as admit a
new one. No verified-EOS optimization or native stop is used here.

## Pair validation

Both arms run on the same device with alternating order. Frozen indices
0/10/20 are included in the 90. The A1 off arm must replay historical
fixed90_fused IDs, text, score, N, encoder/decoder F/Q and route bills;
inputs, scorer, samples and software version are pinned. Physical hooks
check 85 projection widths, selected positions, terminal argmax and encoder
inputs. The strict auditor verifies dirty writes, row selection, and clock.
The paired runner accepts a first divergence only at a warm sample, with
identical earlier trace, selected rows, previous debt and age. The first mix
of admissions and revocations must be a strict expansion. Cases without a
divergence must have identical traces and final IDs.

The offline pass rechecks all 90 historical file digests, exact off replay,
arm bills, verified rows (256 per generated block for *both* arms), and
exposure certificates. Compact outputs contain hashes and scalars only:
`pairs.csv` includes individual first changes and arm-specific phase F/Q;
`cohorts.csv` gives original30, new60, frozen90 and the historical IBS
comparator from `block-history90-matrix.csv`. Here `drain` pools all pure
payments; it does **not** mean all drains occurred while warm. Prefill is
excluded, post-prefill encoder included. TPF is mean per-request N/F,
not wall-clock throughput. The paired 95% intervals use 4,000 within-dataset
bootstrap samples and are descriptive, with no multiplicity adjustment.

## Reproduce without inference

The bounded worker ran from `/gs/runtime/warm-entropy-a1-1128/repo`, a
snapshot based on `be0463873cf1b02f023bcc7b9ef1a8b265d5f032` plus the
source-hashed factor overlay. All raw outputs remain on A800. To regenerate:

```sh
cd /gs/runtime/warm-entropy-a1-1128/repo
/usr/bin/python3 -B -m experiments.t2t.serving_slots.gemma_warm_entropy_report \
  --root /gs/results/warm-entropy-a1-1128 \
  --history /gs/results/profile-close-1128 \
  --manifest benchmarks/frozen90/manifest.json \
  --ibs-matrix /gs/runtime/warm-entropy-a1-1128/block-history90-matrix.csv \
  --out /gs/runtime/warm-entropy-a1-1128/compact
```

## Decision: retain 0.10

All 90 A1 controls replayed, with no missing or failed pairs. The first
changed warm step admitted or retained extra rows in 89/90 questions.
The anchors 0/10/20 were all exposed (+1/+4/+12 acceptance positions);
their total F changed 30->36, 30->28 and 14->13. This confirmed the gate,
not the efficiency hypothesis.

| Cohort | Exposed | Correct off/on | Rescued/lost | Post-prefill F off/on | Physical Q off/on | Macro TPF off/on | IBS TPF |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original 30 | 30 | 30/29 | 0/1 | 814/855 | 91,952/96,660 | 12.212/12.533 | 16.471 |
| New 60 | 59 | 58/55 | 0/3 | 1,616/1,717 | 186,514/196,488 | 13.086/13.127 | 16.392 |
| Frozen 90 | 89 | 88/84 | 0/4 | 2,430/2,572 | 278,466/293,148 | 12.794/12.929 | 16.418 |

Only 13/90 output ID sequences were identical. The candidate saved F on
39, tied on 17 and cost more on 34, with a **net cost of 142 F and 14,682
rows**. Counts grew in every phase: +16 cold sampled F, +77 warm sampled F,
+25 pure payments, +16 verify F, and +8 encoder F. The off/on arms verified
164/172 whole blocks. Output N rose 30,351->32,139. Pooled TPF was nearly
flat (12.490->12.496); the paired macro difference was +0.135 with descriptive
95% interval [-0.154, +0.400]. IBS uses 2,042 F on these 90 and scores
79 correct; the candidate does not close the compute gap, and it gives up
four of the control's correct answers. The four losses are not evidence
against all context-dependent acceptance policies; this particular treatment
is not a drop-in improvement. Do not enable it in a public profile.

`tests/test_gemma_warm_entropy.py`: 20 non-model tests passed, covering
registration gates, a single unchanged parent call, first divergence and
rejection of a partial terminal bill. No A1 follow-up or replication queued.

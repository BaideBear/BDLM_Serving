# Near-term Gemma low-Q decision: retain the qualified verified-EOS path

Use the qualified single-head verified-EOS path at fixed Q32 and Q64:
sample the first tile normally, prioritize owed writes and unresolved rows,
retain right-edge clean-prefix quota 1/8, and preserve the sampler, RNG,
ordered verification and nonterminal encoder. No public profile edit.

| Fixed offer | Original 30: correct / macro TPF | New 60: correct / macro TPF | Frozen 90: correct / macro TPF / TE | EOS-exposed requests | Measured saving vs full terminal sweep |
|---|---|---|---|---|---|
| Q32 | 29 / 4.832 | 56 / 4.902 | 85 / 4.879 / 15.63% | 70/90 | 282 F, 9,024 decoder Q |
| Q64 | 29 / 8.030 | 54 / 8.249 | 83 / 8.176 / 13.96% | 68/90 | 141 F, 9,024 decoder Q |

**Conditional guarantee.** After all debt is paid, a single head with a
contiguous sealed prefix can terminate at EOS measured by ordered verification,
provided that the baseline would not encode or continue. For zero-based EOS row
`j` in a 256-row head, only `(255-j)//Q` whole forwards after its chunk are
removable. The EOS-containing chunk is fully charged. No eligible EOS means
zero saving. A draft EOS is insufficient; no claim for a parked successor or
rebuild. On both 90-case fixed comparisons, all delivered IDs, scores and
encoder inputs matched. These savings are already in the table, not incremental.
See the [verification evidence](../verified-eos/README.md).

**Why no new sampling rule.** In the [first-tile Q32 pairs](pairs.csv),
all 5,344 deferred rows got a fresh first sample. Front-row first-sample
admission rose 40.96% to 51.20%. But per head cold F fell 15 to 14 and warm F
rose 17.65 to 19.75. Even on the first head alone, cold saved 90 F and warm
added 258 F. Correct fell 85 to 82/90 (2 rescues, 5 losses); macro TPF fell
4.879 to 4.580 (paired delta -0.299, descriptive 95% interval [-0.418,-0.181]).
The reverse idea, [sampling on the first cold drain](../cold-first-retry-q32/README.md),
raised Q32 macro TPF to 5.067 but used more F and scored 76/90. At Q64,
[doubling discretionary prefix seats](../eos-prefix-dose-q64/README.md)
rescued four and lost two (83 to 85), but added 279 F and reduced TPF from
8.176 to 8.068. These are changes to later samples and output lengths, not
removable work or additive gains across experiments.

**Headroom on the actual reference trajectories.** Joined the compact off
rows from [frontier](../cold-frontier-last-a1/pairs.csv) and both cold-retry
experiments ([Q32](../cold-first-retry-q32/pairs.csv), [Q64](../cold-first-retry-q64/pairs.csv))
to verified-EOS-on [Q32](../verified-eos/q32/pairs.csv) and [Q64](../verified-eos/q64/pairs.csv).
Each join matched all 90 indices, task IDs, datasets, seeds, output digests,
N, total F and decoder Q. In the cold-retry CSV, `warm_F` is the count of
*all* control `route=="drain"` calls, not the warm phase. Subtract cold-only
drains and post-sampling closing drains to isolate warm-only drains.

| Offer / cohort | All pure drains = cold + warm + closing | Macro TPF with every pure drain free | Capped IBS macro TPF |
|---|---|---:|---:|
| Q32 original30 | 514 = 392 + 66 + 56 | 6.422 | 6.745 |
| Q32 new60 | 1,062 = 798 + 155 + 109 | 6.506 | 6.888 |
| Q32 frozen90 | 1,576 = 1,190 + 221 + 165 | 6.478 | 6.840 |
| Q64 original30 | 289 = 171 + 67 + 51 | 10.439 | 10.918 |
| Q64 new60 | 576 = 345 + 133 + 98 | 10.728 | 11.116 |
| Q64 frozen90 | 865 = 516 + 200 + 149 | 10.632 | 11.050 |

The ceiling is `mean_i(N_i/(F_i-drains_i))` with fixed output and every other
call fixed. It grants mandatory refreshes zero cost: it is **not feasible
work elimination or a prediction**. Capped IBS is a descriptive comparator
(79/90 at Q32, 81/90 at Q64), not a third matched arm.
Macro TPF is mean per-case N/post-prefill F (including encoder F); canonical
TE is pooled N/selected history-reading decoder Q. Neither is wall-clock speed.

**Next decision, not a launch.** The remaining TPF target is work on warm
sampling calls, not simply the price of drains: Q32 has 2,780 warm sampling
calls (3,001 warm calls less 221 drains); Q64 has 1,767 (1,967 less 200).
First use existing traces, where available, to count contests between clean
unresolved rows for the last discretionary seat. One falsifiable hypothesis:
prefer a row with fewer prior measured-but-unaccepted samples, holding dirty
rows, clean-prefix seats and all acceptance rules fixed. This ranks by
unsuccessful sample count, not time since projection. Count how often it
would actually select a different row, by request and by original30/new60;
if ties/current ordering give negligible exposure, stop. The local compact
does not certify these row-level histories or counterfactual outputs. Do
not credit savings or promote without separate authorized paired evidence
of a TPF gain and comparable quality against a predeclared quality margin.
No monitoring, inference, rerun, or follow-up job is part of this assessment.

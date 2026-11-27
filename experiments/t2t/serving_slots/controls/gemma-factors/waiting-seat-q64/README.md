# One waiting MASK seat, fixed Gemma Q64

`registration.md` is the prelaunch hypothesis, gates and decision rule.
Evidence is against the verified-EOS **on** arm at Q64 on the same GPU.
The intervention is not in the serving API. A `swap` in the off arm's
seat tape is a shadow opportunity; only on changes the plan.

## Reproduction

On A800 the runtime is `/gs/runtime/waiting-seat-q64-1128/repo`, copied
from `/gs/runtime/verified-eos-1128/repo` with the private experiment files
added. The reference source hashes are checked before loading the model.
The scorer is pinned for all arms and datasets, including `math-verify`
0.8.0 and sandboxed HumanEval. Use only compact summaries on the Mac.

Actual launcher, single execution, 3,600 s bound:

```sh
bash /gs/runtime/waiting-seat-q64-1128/launch.sh
```

This command is expected to refuse a repeat: it checks that the prior
qualification completed, its PID no longer exists, GPU7 is empty, and the
new result directory does not already exist. `registration.json` under
`/gs/results/waiting-seat-q64-1128` identifies the actual model process,
registration hash, scorer and paired-reference digest.

Offline recheck/export (standard library; no GPU needed):

```sh
cd /gs/runtime/waiting-seat-q64-1128/repo
/usr/bin/python3 -B -m experiments.t2t.serving_slots.gemma_waiting_seat_report \
  --root /gs/results/waiting-seat-q64-1128 \
  --history /gs/results/verified-eos-1128 \
  --manifest benchmarks/frozen90/manifest.json \
  --out /gs/runtime/waiting-seat-q64-1128/compact
```

`pairs.csv` has scores, token counts, output digests, physical bills, shadow
opportunities and first-exchange evidence (local positions and prior trace
hash). `cohorts.csv` separates the three anchors, original30, new60 and
frozen90 if qualified; anchors are included in original30, not additional
samples. `provenance.json` seals the compact export. Total F/Q exclude
prompt prefill and include every post-prefill encoder. Macro TPF is the
mean of per-question N/F, not wall-clock throughput. The reported intervals
are 4,000 within-dataset paired bootstrap draws, descriptive only.

## Results and decision

All 90 off receipts matched the historical verified-EOS on arm in every
field, including output IDs, scores, physical decoder tape and encoder
digest. The paired run and offline replay audit both completed. The 10
synthetic tests did not use a model. There were no missing, failed or
replacement pairs. In the table each arrow is off to on; Q includes encoder
rows and is the actual bill, not the number of offers times 64.

| Cohort | Exposed | Correct | Non-special tokens | Paid F | Physical Q | Macro TPF |
|---|---:|---:|---:|---:|---:|---:|
| Original 30 | 4/30 | 29 -> 29 | 10,348 -> 10,295 | 1,314 -> 1,300 | 82,887 -> 82,431 | 8.03015 -> 8.07147 |
| New 60 | 3/60 | 54 -> 55 | 21,577 -> 21,416 | 2,635 -> 2,626 | 166,752 -> 165,933 | 8.24914 -> 8.23319 |
| Frozen 90 | 7/90 | 83 -> 84 | 31,925 -> 31,711 | 3,949 -> 3,926 | 249,639 -> 248,364 | 8.17614 -> 8.17928 |

The three anchors paid 126 -> 114 F with all three correct. Only #10 was
exposed: at zero-based decoder tick 11, one of 64 selected rows changed
from local 67 to 72, previously projected on ticks 10 and 3. Its F fell
58 -> 46; #0 and #20 matched throughout. Across 90, off had 1,767 sampled
warm ticks, 1,341 contested and 11 strictly-older shadow opportunities.
On had 1,746 sampled warm ticks, 1,326 contested and 14 actual exchanges.
The two opportunity counts describe different trajectories, not paired ticks
after the first exchange. All 83 unexposed pairs matched in full. All seven
exposed pairs changed output IDs: four used fewer F and three used more;
#16 used 135 -> 139 and remained wrong, while #67 used 128 -> 113 and
was the only score rescue. There were no losses.

The net saving was 23 decoder F and 1,275 decoder rows. By route the F
change was sparse -21, drain 0, verify_paid -1 and verify -1; later verify
counts changed with the generated trajectory, not by omitting required
chunks. Encoder stayed 82 F / 20,992 Q. Macro TPF delta was +0.00314
(paired descriptive 95% interval [-0.0238, +0.0415]); pooled TPF fell
8.08433 -> 8.07718 as N also fell. The new60 macro delta was -0.01595
despite one more correct answer.

**Decision:** the registered fixed-Q64 rule passes its numerical criteria,
but this is a marginal experimental candidate, not a reason to change the
default or claim the small-budget TPF gap is solved. Exposure was 7/90;
the 3 anchors are part of that 90 and do not provide independent validation.
The executable, exact launch and per-request verification remain reproducible
as above. The three compact files matched their remote SHA-256 digests;
generations and raw tapes stayed on A800.

# Stop an ordered sparse terminal sweep at measured EOS

Question: after paying for a whole verification chunk, can we omit the
remaining chunks of the **last** canvas without changing any delivered token?
These fixed offers use the `fixed90_fused` controls (Q32/64: 1/8; Q128: 1/4),
right-edge prefix selection, no successor, and the existing guarded first
verification chunk. No sampling threshold, RNG, dirty rule or encoder changed.
The fixed experiment is unchanged. A separate opt-in request profile,
`dense_stop_joint_verified_eos`, adds this same rule to the existing switchable
`dense_stop_joint_single512` executor and passed the [GPU switching qualification](../../strict-handle/gemma-verified-eos.json).
A parked successor, a joint verification or an encoder rebuild keeps the
old behavior, even if discarded rows would appear to be cheap. A new offer never
resets an in-progress verification cursor. The default profile is unchanged.

The sampler freezes ids and self-conditioning and seals the argmax of each
actual ordered terminal forward. All positions from zero through the current
chunk must have been sealed, and all writes paid. Only an EOS in this chunk
can finish the request. The EOS token is delivered; all positions in its chunk
are physically charged, even those after EOS. The unchanged executor would
truncate at the same EOS without an encoder call. A draft EOS never suffices.
Without a measured EOS, the usual full sweep and commit still run, including
the full last canvas on reaching the length cap.

## Evidence and limits

Each pair ran sequentially on physical A800 GPU7. The off arm first has to
match its registered historical result: full output IDs, score, input hash,
token count, encoder and decoder F/Q, and per-route F/Q. For both arms, hooks
record decoder position IDs, 85 projection widths, model-output argmax, and
the actual encoder IDs and positions. The audit reconstructs the output from
terminal argmax, checks the dirty ledger and block clocks, and requires the
same measured per-step fingerprint up to and including the EOS-containing
chunk. All nonterminal blocks must verify 256 rows and encode identically.
The only removable route is ordinary terminal verification after EOS.

Fixed Q32 on the [complete frozen90 cohort](q32/cohorts.csv): 85/90 in
both arms (29/30 original and 56/60 new). Every delivered ID, score and
encoder input matched. 70/90 requests saved at least one whole chunk; the
other 20 had EOS in the last chunk. Totals fell from 7,156 to 6,874 F and
218,713 to 209,689 decoder Q, with 32,774 nonspecial output tokens in both
arms. Macro TPF rose from 4.669 to 4.879 and pooled token efficiency
from 14.98% to 15.63%. These numbers do not include results from other offers.

Fixed [Q64](q64/cohorts.csv) independently matched all 90 delivered ID sequences,
scores and encoder inputs, at 83/90 (29/30 original, 54/60 new).
68/90 requests saved whole terminal chunks; 22 had EOS in the last chunk.
F fell 4,090 to 3,949 and decoder Q 237,671 to 228,647, with the same N=31,925.
Macro TPF rose from 7.881 to 8.176 and pooled token efficiency from 13.43% to
13.96%. The number of skipped rows happens to equal Q32's total, but Q64
skips only 141 forwards. Both physically billed the whole EOS-containing chunk.

Fixed [Q128](q128/cohorts.csv) retained 88/90 (30/30 original, 58/60 new).
All 90 output sequences, scores and encoder tapes matched. EOS fell in the
first 128 rows for 46 requests; the 44 remaining requests paid both chunks.
N=30,351; F fell 2,430 to 2,384 and decoder Q 259,522 to 253,634. Macro TPF
rose 12.794 to 13.047; pooled token efficiency 11.69% to 11.97%.
On this cohort IBS128 is 79/90, 16.418 and 14.66%: a quality advantage,
not a TPF win. The eligibility and exact trajectory checks, not just the
unchanged aggregate score, are why we recommend this optimization.

Independent per-request prediction from the historical final IDs: if EOS
is at zero-based block-local position `j`, the number of removable forwards
is `(255-j)//Q`; removable physical rows are that count times Q. An EOS in
the last chunk, or reaching the cap without EOS, gives zero. Reported savings
must match independently measured off-minus-on bills on every pair.

`pairs.csv` contains scalar paired results and an ID digest, no generation text.
`cohorts.csv` contains complete original30, new60 and frozen90 by offer.
Macro TPF is the mean of per-request `N/F`; micro TPF pools tokens and paid
forwards. Prompt prefill is excluded, post-block encoder work included.
These are accounting metrics, not measured latency, wall-clock throughput or
a claim about other models, unordered verification, or mixed offers.

The paired runner and source snapshot are under `/gs/runtime/verified-eos-1128/repo`
on A800, based on `badf953ac0e51a0b3e3b1e5860041c6b5008ac54`. Runs, token
sequences and tapes stay under `/gs/results/verified-eos-1128`. To regenerate
the compact files without allocating a GPU or downloading raw outputs:

```sh
cd /gs/runtime/verified-eos-1128/repo
/usr/bin/python3 -B -m experiments.t2t.serving_slots.gemma_verified_eos_report \
  --root /gs/results/verified-eos-1128 --history /gs/results \
  --manifest benchmarks/frozen90/manifest.json \
  --generation-config /gs/models/diffusiongemma-cd40195/generation_config.json \
  --out /gs/runtime/verified-eos-1128/compact
```

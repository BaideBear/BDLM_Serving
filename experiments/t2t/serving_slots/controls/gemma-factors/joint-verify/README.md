# Gemma Q512: combine terminal verification and resident-tail payment

Fixed Q512, B=256, `00`, right-edge 1/8, half gate, sampled promotion
rebuild and guarded head-only verify in **both** arms. No fullstop observer,
new admission, cold pacing or noise rule. Source base
`89e23d62a4ae8867a791c79e8e9dc41263a8e896`, same pinned checkpoint,
same single A800 GPU7, same frozen inputs/scorer/seed. Each question alternates
arm order with the previous question. Qualification 0/10/20 is reused in 90.

## What changes

Only at a terminal, initialized head with a resident successor, before its
first verify chunk, with nonempty global dirty debt: in the old encoder
epoch let H and T be the head and successor dirty rows. Off pays H+T then
projects all B head rows for terminal argmax. On selects exactly all B head
rows plus T in one block-causal physical decoder call (at most 512 rows).
The head consumes its current logits for terminal verification. The successor
only pays; no sampling, acceptance change, RNG draw, or sampler step.
The encoder, if needed, still consumes the verified head, not a draft.
Empty debt is not counted. An event with an already clean resident tail is
counted separately from one that actually writes tail rows.

Direct mechanical saving at an event: one decoder F and exactly |H| decoder
rows. All |T| rows are projected, even for a successor ultimately discarded.
This is **not** a proof of numerical equivalence: in off, the intermediate
drain changes resident layer K/V before the all-head forward, and the physical
batch shape differs. Changes to a verified head (including positions after
EOS), subsequent blocks, output length and score are empirical questions.

## Receipts and reproduction

`pairs.csv` retains all matched questions, including no opportunity. Both arms
report paid encoder and decoder calls and row counts (prompt prefill excluded),
quality, N, per-question N/F, N/(decoder_Q+encoder_Q), first exposure tick
and head ordinal (zero-based), equality of first and later aligned head seals,
and equality of complete returned token IDs. A common event ordinal is not
necessarily a common pre-intervention state after the first intervention.
`events.csv` gives exact row-set hashes, counts, first added head-local row
and physical F/Q per event and arm. The cluster receipts additionally contain
block-local dirty-row lists. No prompts, token IDs or generations are exported.

At the first opportunity the auditor requires matching physical prefix-trace
hash, head ordinal/tick, selected row identities and input-state fingerprint
(current token IDs, accepted/dirty sets, RNG state, sampler age and
self-conditioning keys; not a hash of hidden tensors). On a pair without an
opportunity, output IDs, scores and bills must be identical. All off receipts
must reproduce the original `C512-promotion1-*` ID hash, score and physical
encoder/decoder/route bills. Independent hooks measure absolute row positions,
85 QKV projectors/30 layers, LM head, encoder and actual verified IDs.

`cohorts.csv`: macro TPF = mean of individual N/F. Macro physical row efficiency
is likewise a per-question mean. Pooled decoder and pooled physical efficiency
are separately labeled ratios of sums, the first matching the repository's
history-row denominator. `F_residual` is (on F - off F) plus on-event count;
`decoder_Q_residual` adds back on-event |H| to the decoder-row difference.
They describe the deviation from mechanical savings on the on trajectory,
not a causal decomposition once the two trajectories differ. The 4,000
paired, within-dataset bootstrap draws quantify finite-benchmark variation;
no wall-clock speedup, request-level equivalence or population quality
guarantee is implied.

```sh
cd /gs/runtime/joint-verify-1128/repo
/usr/bin/python3 -B -m experiments.t2t.serving_slots.gemma_joint_verify_report \
  --root /gs/results/joint-verify-1128 --history /gs/results/promotion-1128 \
  --out /gs/runtime/joint-verify-1128/compact
```

The compact export refuses an incomplete or mixed registration. It reads
sealed receipts only and does not rerun inference.

## Results and decision

| Matched cohort | Exposed questions | Applied / all head closures | Correct off/on | Total post-prefill F off/on | Decoder Q off/on | Macro TPF off/on | Paired delta TPF (95%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original 30 | 17/30 | 22/52 | 28/28 | 470/448 | 138067/137903 | 23.324/24.160 | +0.835 [+0.607,+1.069] |
| Added 60 | 42/60 | 59/115 | 56/56 | 981/922 | 295869/294921 | 23.923/25.287 | +1.363 [+1.078,+1.657] |
| Frozen 90 | 59/90 | 81/167 | 84/84 | 1451/1370 | 433936/432824 | 23.723/24.911 | +1.187 [+0.981,+1.425] |

All 90 off arms matched historical promotion1. Both arms returned identical
output IDs in **90/90**, 31,590 non-special tokens total. Their event head
ordinals and all 81 aligned event seal digests matched. The remaining 31
questions had no opportunity. Of 81 events, 70 paid nonempty tail debt
(11,927 tail rows), 11 had a resident but already clean tail. Each question's
F and decoder-Q differences match exactly its count of events and duplicate
head rows (1,112 total). Paid decoder calls 1,374 -> 1,293; encoder 77 calls
and 19,712 rows in each arm. No missing pairs, rescues or losses.

Qualification #0 and #10 first intervened at zero-based decoder ticks 11 and
9 respectively (H/T 5/26 and 37/135); #20 had no opportunity. The independent
firewall tests and unchanged planner tests passed (72 cases, no model run).
Both arms and the entire frozen run used physical GPU7 only.

**Recommend integrating the guarded Q512 route for this fixed profile.**
Macro TPF improved about 5.0%; decoder rows fell only 0.26%. This primarily
removes calls, not tail-row work. The test observed no paired quality or
output change; it does not prove equivalence outside these 90 requests or
unchanged wall-clock latency. Keep the exact terminal/epoch/debt gate and do
not extrapolate the route to a smaller offer or a switching profile.

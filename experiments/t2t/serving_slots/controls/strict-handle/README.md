# Can a single live request consume any of the five offers?

This is a functional GPU test on frozen indices 0, 10 and 20, not a mixed-budget
quality experiment. Drive the same strict handle to EOS or the registered length
limit. Never compare scores from these paths to fixed-budget curves.

The Gemma test uses the A800 checkpoint, frozen prompt template and seed, and
the `fixed90_candidate` profile. For each of 5 constant offers and each of 3
requests, compare the **entire output token sequence** and F/decoder Q/encoder Q
with the already measured control at that exact configuration. Follow every
physical decoder position and 85 projection-module widths; verify encoder input
IDs against the committed output and verify uninterrupted sampler ages and
absolute dirty rows, including across a budget change. The LLaDA test uses its
pinned environment, `"head"` coverage and the half gate. It compares each fixed
case to an independent run through the fixed adapter, including every decoder
position sequence and final IDs. It checks the 20-layer projection ledger, first
layer physical mask, LM head width, sampling clock and dirty rows at every tick.

Second, `switch25` repeats an Euler cycle over the five offers, starting at the
minimum; `park` deliberately includes top-to-bottom transitions. Stop at the
real request boundary, not when the test sequence finishes. Distinct observed
transitions are counted on consecutive forwards of the *same* request, never
stitched across requests. A cap change does not imply that parked debt existed;
the report counts rows actually paid at the downshift. An offer assigned to a
pure writeback or terminal step is counted as a real transition, not evidence
of productive sampling at both ends.

On Gemma, all 15 fixed cases replayed exactly. Across the three `switch25`
requests all 25 transitions occurred, with 29 downshifts carrying debt and 140
parked-row refreshes on the downshift. The `park` run covered four directed
pairs and paid 26 such rows. Nothing in this test establishes arbitrary-path
quality, global equivalence to dense decoding, or that each individual state
was exercised for every directed pair.

On LLaDA, all 15 fixed cases matched every generated ID, projected position,
F and decoder Q in the independent fixed adapter. `switch25` collectively
exercised all 25 directed pairs: 101 downshifts with outstanding debt and
46 parked rows paid on the downshift. `park` had 22 such downshifts and 19
parked rows. The GPU snapshot's `llada_budget_request.py` SHA-256 was
`6853c576a3f111ef7dca8e22de9cb1e2615a36369ab5a822ee31e854d38358ed`.
Compared with the snapshot, commit `2bd32cf` only omits two optional
diagnostic fields from sparse records; the planner, sampler and KV writes
are unchanged. This evidence does not qualify later experimental samplers.

On the GPU host, run `python -m experiments.t2t.serving_slots.strict_handle_qualify
--model gemma` or `--model llada` in the corresponding pinned environment,
with the checkpoint, frozen samples, controls and output directories. Reproduce
`gemma.json` using `strict_handle_report.py gemma /gs/results/public-strict-1128
benchmarks/frozen90/manifest.json`; for the other model select `llada` and
`/gs/results/public-strict-llada-1128`. The reporter rejects partial cohorts;
its JSON is aggregate and contains no output IDs.

Additional Gemma qualifications on the same frozen 0/10/20: both
[`fixed90_fused`](gemma-fused.json) and
[`fixed90_dense_fused`](gemma-dense-fused.json) replayed all 15 constant
cases exactly on GPU (final IDs and measured workload). Each exercised all
25 directed transitions in `switch25`, with 29 downshifts carrying debt.
They respectively refreshed 140 and 360 parked rows *at* a downshift. These
are different live trajectories, not a cost comparison at identical states.
`fixed90_dense_fused` changes only the actual one-block offer when the entire
head and its mandatory writes fit. Neither configuration restarts upon switching.

[`phase_high_fused`](gemma-phase-high-fused.json) also passed 15/15 exact fixed
replays, using the independently measured high-unresolved arm for Q256. The
three switch-cycle cases covered all 25 directed pairs, with 26 downshifts
carrying debt and 109 parked rows refreshed at the downshift. This is a
functional check, not a mixed-budget quality curve.

[`dense_b_stop_fused`](gemma-dense-b-stop-fused.json) replayed all 15 fixed
cases against their own controls, with zero eligible stop pairs at fixed
Q32/64/128/512. `switch25` exercised 25 directed pairs, with 29 downshifts
carrying debt and 360 parked rows written at downshifts. The separate
`drop_after_stop` schedule applied Q32 after a latched Q256 stop: four such
downshifts with debt and 37 sub-B decoder ticks after a latch. A latch can
coincide with ordinary termination; these counts do not imply four early exits.

[`dense_stop_promotion_fused`](gemma-dense-stop-promotion-fused.json) reproduced
15/15 fixed GPU cases against the appropriate controls (including the
Q512 sampled promotion arm), and exercised all 25 directed pairs. The
targeted `drop_after_promotion` path observed two actual promotions followed
by Q32 with outstanding writes; both were accounted for without a state
reset. The B-only stop was ineligible during sampled promotion, and the
post-stop Q32 path still executed 37 sub-B ticks. This is an API and cache
qualification, not a score for either mixed path.

[`dense_stop_joint_verify`](gemma-joint-verify.json) passed 15/15 fixed
replays on the same indices, using the independently measured joint1 output
digest and physical ledger at Q512. `switch25` saw all 25 directed pairs.
The dedicated fallback offered 32 instead of 512 at **two actual resident-tail
close opportunities**: it did not take the joint shortcut, paid 26 parked
rows at the downshift and verified the two heads in 16 small chunks.
This proves the measured functional paths, not their comparative quality.

[`dense_stop_joint_single512`](gemma-single512-stop.json) passed 15/15 fixed
replays (indices 0, 10, 20 at five offers) and all 25 directed transitions
in `switch25`. The three fixed Q512 requests additionally matched every
one of the 42 decoder ticks against the sealed `single1` tapes: selected
rows, admissions/revocations, writes, clock, stop and physical bills. Of
27 qualified full-head ticks, 22 had a qualified predecessor at the same
offer; 2 latched. Only #10 still had masks and remaining sampler steps.
#0's latch coincided with full acceptance; #20 had no latch.

The three targeted switches therefore each have denominator **one request**.
All were aligned to #10, head 1, encoder epoch 322, fixed hit at tick 13:

| Offer at the boundary | Observed result |
|---|---|
| 512 on the hit, then 32 | 2 masks and 7 dirty rows at latch. Q32 paid the 7 rows in one drain, then freshly verified 8 ordered chunks of 32. Sampler age remained 6. |
| 256 instead of 512 on the hit, then 512 | Both ticks were real full-head samples; neither hit. A pair cannot straddle two otherwise eligible offers. |
| 32 instead of 512 on the hit, then 512 | Partial sampling interrupted the pair; the first fresh full Q512 sample on return did not hit. |

These are 33 functional runs (15 fixed, 15 standard mixed, 3 targeted), not
a mixed-budget accuracy estimate. No new frozen90 quality run. A800 physical
7 only; source snapshot `badf953ac0e51a0b3e3b1e5860041c6b5008ac54` plus
the API/qualification patch in `/gs/runtime/single512-api-1128/repo`.
Rebuild the compact file from existing remote receipts without loading a model:

```sh
cd /gs/runtime/single512-api-1128/repo
/usr/bin/python3 -B -m experiments.t2t.serving_slots.strict_handle_report \
    gemma /gs/results/qualify-single512-api-1128 benchmarks/frozen90/manifest.json
```

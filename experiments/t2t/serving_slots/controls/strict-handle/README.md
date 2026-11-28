# Can a single live request consume any of the five offers?

**Verified-EOS profile.** [Qualification receipt](gemma-verified-eos.json).
Five fixed offers times three requests matched sealed reference outputs and bills.
Twelve mixed executions matched the no-shortcut public handle on the same offer
path: every event and measured argmax through EOS, delivered IDs, encoder inputs,
and exactly the billed post-EOS suffix. On #0 an upward switch paid rows [32,160)
at Q128; a downward switch paid [128,160) at Q32. Both left 96 rows unneeded,
saving one and three forwards respectively. These are two different trajectories,
each compared only to its own control. `switch25` exercised all 25 transitions,
29 downshifts with pending writes and 360 parked rows paid on downshift.
The first incomplete deployment had an older dense executor, caught by the Q512
stop-scope assertion; none of those initial runs are included. Audit the completed
run with `strict_handle_report gemma-eos /gs/results/eos-handle-1128-v2 benchmarks/frozen90/manifest.json`.

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

On A800, [`quarter_edit_warm`](llada-quarter-edit-warm.json) independently
replayed 15/15 fixed LLaDA cases on 0/10/20, including output IDs, absolute
rows and F/Q. The three fixed Q32 cases took 32 terminal edit opportunities
and made five real writes. At fixed Q64, the warm-head exception admitted
78 additional *sampling opportunities* (not necessarily output writes).
`switch25` exercised all 25 ordered pairs, 136 downshifts with dirty rows
and 69 parked rows repaid on downshifts. The four terminal edit opportunities
in the mixed cycle made zero actual writes; do not infer from this that
the cycle measured a new terminal edit immediately followed by a downshift.
The raw 21 runs are on A800 at `/gs/results/qualify-llada-editwarm-api-1128`.

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
## LLaDA active-window qualification

`quarter_edit_window` is a functionally qualified opt-in candidate,
**not a quality-validated replacement** for `quarter_edit_warm`.
Fixed 4/8/16 must reproduce the independent head-only
window arms' physical row sequence, output IDs, F and Q on the original
0/10/20 anchors (no half-block borrowing). Fixed 32/64 must reproduce the
previous terminal-edit and quarter-gated warm-head arms. The five-offer
cycle must cover all 25 directed transitions, including real downshifts with
debt. A separate `park` test waits for a measured write to the successor at
64, then pays it at 4 and returns to 64. No small offer may sample the parked
tail or project any clean parked row. Do not interpret these boundary tests
as a quality ranking for mixed budgets.

[A800 GPU7 result](llada-quarter-edit-window.json): 15/15 constant-offer
replays matched, including at 4/8/16 (160/167/316 forwards actually changed
window selection in those suites). All 25 directed transitions occurred.
The general cycle recorded 87 downshifts carrying debt. In the targeted park
test, all three requests used 64→4→64; 14 dirty successor rows were paid while
parked (9 on the immediate drop and 5 on later small calls). Receipts:
`/gs/results/llada-switch-window-api-replay-1128`, source `c16a6d5`. The first
package lacked the frozen30 identity manifest and exited before any request;
the replay used the missing manifest from the same commit, not different tasks.

## LLaDA: active window with consecutive Q64 confirmation

[`quarter_window_repeat64`, A800 GPU1](llada-quarter-window-repeat64.json)
keeps the lower four fixed paths unchanged. At Q64 a fully resident head may
accept an additional non-MASK draft only after the same row gives the same
draft on consecutive measured head samples, above 0.5 both times. The ordinary
acceptance rule and its fallback run first. A vote is tentative, unlike a KV
write obligation: an intervening small offer invalidates the vote, not the debt.

The 15 fixed replays (five offers on frozen indices 0/10/20) matched output
IDs, selected positions and physical F/Q against their independent controls.
The fixed Q64 cases exercised 64 additional admissions, reconstructed from
paid proposals. `switch25` covered all 25 ordered pairs, with 90 downshifts
carrying debt, 31 parked rows paid on such downshifts, and 111 tentative
votes discarded. In the separate targeted `park` suite all three requests
traversed 64→4→64: 10 parked rows paid, six on the first drop. Checks included
the first-layer causal mask and all 20 projection widths. This qualifies
the live handle, *not* a mixed-budget quality or TPF improvement.

Source snapshot `8fd6d6a`, runtime `/gs/runtime/llada-switch-repeat64-1128`.
Rebuild this compact report there without rerunning inference:

```sh
/usr/bin/python3 -B -m experiments.t2t.serving_slots.strict_handle_report \
  llada /gs/results/llada-switch-repeat64-1128 benchmarks/frozen90/manifest.json
```

## LLaDA: Q64 repeat plus Q16 paid warm-MASK

[`quarter_window_repeat64_paid16`, A800 GPU4](llada-quarter-window-repeat64-paid16.json)
keeps the Q64 repeat behaviour above and adds the measured Q16 paid
waiting warm-MASK sampling on mixed cold calls. It is a separate opt-in
profile; the old `quarter_window_repeat64` interface and Q64 path are
unchanged. The Q16 addition samples only already-projected warm MASK rows:
no extra Q, clean prefix, head dirty row or new sampling tick.

All 15 fixed cases (five offers on 0/10/20) replayed exactly, including the
paid Q16 arm (273 paid waiting rows, 121 window forwards). `switch25` covered
all 25 directed pairs with 101 debt downshifts, 50 parked rows paid and five
paid Q16 rows in the mixed cycle. The targeted Q64→Q16→Q64 `park16` suite
observed `64>16` and `16>64` with 3 debt downshifts and 22 parked rows paid;
that particular parked boundary had no cold Q16 call with a waiting warm
MASK, so its paid-mask exposure is zero. The Q64 fixed path remained exact
(3/3, 64 additional admissions). This is a functional qualification, not a
mixed-budget quality result.

Valid run: `/gs/results/qualify-llada-paid16-api-1128-v2`; the aggregate is
linked above and the 24 compact receipts plus notes are in
[`llada-quarter-window-repeat64-paid16/`](llada-quarter-window-repeat64-paid16/README.md).
The first attempt
`...-1128` is excluded because the report rejected a missing `park16`
`parked_dirty_Q` counter; all 24 receipts had been generated and no model
request failed.

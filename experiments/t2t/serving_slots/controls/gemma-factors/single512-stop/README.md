# Fixed Q512 single-resident native stopping

Off is `joint1` (Q512, half gate, sampled full promotion rebuild, guarded
head-only and resident-tail terminal verification), with a shadow observer.
On applies exactly that observer. No change to admission, joint verification,
sampling, seeds, self-conditioning, mandatory refresh, or encoder writes.
Neither arm uses the public switching profile. Both ran on the same physical
A800 GPU7. No subgroup was selected according to outcomes.

A decoder offer of 512 does not itself qualify a tick. There must be only one
resident canvas, route `dense_single`, and **all 256 head rows freshly sampled**
in order. The pinned generation config uses the native criterion (stability 1,
confidence 0.005); only its FP32 processed scores from the actual sampler
callback are observed. Two immediately consecutive such ticks must share
head ordinal and encoder epoch. Any other tick resets the criterion. This
includes a parked resident successor even if the head alone is sampled, as
well as `dense_joint`, `rebuild`, `rebuild_sample`, payment and verification.
No cached logits or last draft can substitute for the terminal sweep.

The numerator for **native positives** is the criterion's own decisions; it
is not the number of finished masks. An **actionable positive** also requires
remaining masks and remaining sampler steps *after* that tick's normal
writes. Off must actually sample on its following tick. On must fully verify
the same head on the next physical decoder call and pay all outstanding
rows; any encoder call is billed separately and checked against the seal.
A positive with zero remaining masks is a legal hit but not saved work.

## Gate and audit

Indices 0/10/20 were registered first and reused in the 90. All three shadow
arms reproduced `joint1`. At #0 the positive at tick 19 had no masks left;
both arms paid 22 F. At #10 the first actionable positive at tick 13 had two
masks and seven dirty rows; on went straight to measured `verify_paid` and
changed 17 F / 4,743 decoder rows to 16 / 4,487, with identical returned
tokens. #20 had four qualifying windows but zero positives, both arms 6 F.
Only a nonzero actionable count permitted expansion. The run is bounded
to these 90 tasks; order alternates between the two arms.

`row-roles.json` is a retrospective bill on the 90 pinned **on** tapes,
not another model run. From a checkout containing `gemma_top_roles.py`, run
`python -m experiments.t2t.serving_slots.gemma_top_roles /gs/results/single512-stop-1128 experiments/t2t/serving_slots/controls/gemma-factors/single512-stop/pairs.csv benchmarks/frozen90/manifest.json`.
The six exclusive roles sum to the measured decoder bill. Actual debt paid,
successor rows, and rows on a sampled rebuild are overlapping diagnostics,
not additional cost. The nonzero `edit_positions` count is T2T, excluding
first admissions and remasks.

Every off must replay its archived `C512-joint1-*` output hash, scorer,
full route and physical encoder/decoder bills, and the recorded close
opportunities/rows/seals. Before the first actionable stop, the measured
physical trajectory (including that tick's sampler updates) must match.
If there is no actionable stop the full output, route, joint-close state and
trajectory must agree. All 85 QKV projectors and the LM head are measured;
30 decoder layers, encoder inputs, epoch and terminal argmax are checked.
The second audit rereads all compressed ledgers, verifies their SHA-256,
reconciles every physical call/position, recovers residency, qualifying
windows, per-tick debt, sealed heads, and checks mandatory rows after hits.
The observer, dispatch and joint planner suites passed 87 model-free tests
in the pinned environment.

`pairs.csv`: both arms' scores, N, paid F/Q by encoder and decoder, route
histograms, reset reasons, qualified ticks, windows, positives, actionable
positives, first-hit ticks, closed heads and total written/refreshed rows.
`hits.csv`: each actual native decision, including non-actionable ones.
`cohorts.csv`: original30, new60, frozen90, with no-hit tasks retained;
macro TPF is the mean per-question non-special N / post-prefill paid F.
95% intervals use 4,000 paired resamples stratified by dataset. These are
neither wall-clock measurements nor guarantees of population quality.

Full `C512-single{0,1}-*.ledger.json.gz` physical and executor tapes stay
in `/gs/results/single512-stop-1128` on A800. Regenerate only the compact
exports, with no model inference:

```sh
cd /gs/runtime/single512-stop-1128/repo
/usr/bin/python3 -B -m experiments.t2t.serving_slots.gemma_single512_report \
  --root /gs/results/single512-stop-1128 --history /gs/results/joint-verify-1128 \
  --out /gs/runtime/single512-stop-1128/compact
```

## Results

| Cohort | Matched | Actionable cases (on) | Correct off/on | Paid F off/on | Decoder Q off/on | Macro TPF off/on | Paired delta TPF (95%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original 30 | 30 | 8 | 28/28 | 448/436 | 137903/134831 | 24.160/24.923 | +0.763 [+0.276,+1.355] |
| Added 60 | 60 | 10 | 56/56 | 922/907 | 294921/291081 | 25.287/25.932 | +0.645 [+0.283,+1.070] |
| Frozen 90 | 90 | 18 | 84/84 | 1370/1343 | 432824/425912 | 24.911/25.596 | +0.685 [+0.365,+1.044] |

All 90 shadow controls reproduced `joint1`; zero missing or failed pairs.
Off: 691 eligible samples, 556 consecutive windows, 62 native positives,
21 actionable positives in 18 questions; zero applied. On: 664 eligible
samples, 529 windows, 42 native positives (all applied), of which 18 were
actionable in 18 questions. The other 24 decisions on the on trajectory
were already terminal; repeated shadow positives are not independent savings.
72/90 questions had no actionable intervention.

The two trajectories produced 31,590 non-special tokens each with no
scorer rescues or losses. **Two returned ID sequences and their head seals
changed**: #24 (wrong in both) and #39 (correct in both), both first differing
in head ordinal zero. This is not a bitwise-equivalent endpoint. The exact
paid differences were 27 decoder calls and 6,912 decoder rows (27 x 256).
Paid decoder F was 1,293/1,266; encoder F/Q stayed 77/19,712. All 167 head
seals and all 81 guarded joint closes per arm remained billed. In both arms
347 dense joint calls, 58 sampled rebuilds, 18 plain rebuilds and seven
`dense_single` calls with a parked resident tail were excluded from the
stability window. Off/on wrote and subsequently refreshed 198,339/198,276
rows respectively; no unpaid writes. All four compact files matched their
remote SHA-256 checksums after transfer.

**Decision:** positive efficiency result on the fixed Q512 profile, suitable
for separate opt-in qualification, not evidence for generalizing the gate
to 512->lower transitions, resident successors or sampled rebuilds. Observed
unchanged scores on 90 fixed tasks do not establish quality noninferiority.

## Should a current-head T2T defer an active successor's sample?

Only one of these 90 Live Q512 trajectories would have met the proposed
gate on an already fully projected successor: **index 51, one call**.
The 90 requests had 17 actual head T2T writes on 17 calls. Two coincided
with any successor sample, one with a warm successor sample, and that one
also changed the successor's input or acceptance state. In total there
were 225 calls sampling an already warm successor. Warm means all 256
successor rows had been physically projected **before** this tick in the
same head/encoder epoch. First projections, admissions on the head, and
hypothetical argmax differences are not head T2T.

The screening threshold was 10 distinct exposed requests per model.
Do **not** run this GPU intervention: 1/90 on Gemma is sufficient to
reject this test for the shared rule. This is an exposure count on a fixed
control, not an estimate of accuracy or of counterfactual forwards, and
not evidence against cross-block decoding generally. Reproduce with
`gemma_top_roles.py ROOT pairs.csv MANIFEST --tail-edit-exposure` using
the same hash-verified 90 ledgers that generated `row-roles.json`.

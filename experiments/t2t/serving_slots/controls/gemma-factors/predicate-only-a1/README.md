# Fixed Q256: can all-accepted stand in for native stopping?

No. The strict controller in both previous visibility and continuous-RNG
experiments already listened to the real factory-built `FullHeadStop`, yet
`Canvas.can_sample` *also* returns false when `rows.accepted == FULL`. The
previous fullstop/dense-stop/single512 studies tested the observer on/off, or
qualified which observations can trigger it. This one removes only the
separate local all-accepted condition, on the same continuous-RNG,
None-when-unrestricted visibility path. Only a real native hit or step 48
can end denoising on the experimental arm. The experiment is request-local
in scope (module patch in one serial worker) and fixed at Q256, with one head,
no parked tail and no rebuild. It is not a public switching implementation.

`gemma_predicate_only.py` changes the Canvas sample-permission predicate;
`gemma_predicate_only_eval.py` reuses the physical executor, continuous-RNG
runner and its checks. The off arm must replay every field of the previous
continuous-on row and its complete step/RNG ledger. At the first divergence,
both arms must match throughout the triggering sample, including all sampler
events, RNG, actual stop decision and measured entropy. The next operation
must be a full paid verification on off, and a full measured denoising step
on on. Zero-opportunity requests must be identical throughout.

Native `StableAndConfidentStoppingCriteria` on this checkpoint compares
the *processed-score argmax* to the preceding full argmax, **not** the
accepted mask or the renoised canvas. Thresholds are one stable comparison
and strictly less than 0.005 mean full-head categorical entropy. The tap
reads the criterion's own entropy output and return value without drawing
random numbers, rerunning the criterion, or evaluating cached logits.
The original 48-step limit is not a positive native hit.

## Direct #10 evidence

Separate same-card native replay of #10 matched the entire saved native
row. Both candidate blocks matched its criterion tape and denoising
IDs/argmax/sampled IDs/acceptance/positions at each actual sample. On
the second block:

| Completed sample | Accepted / 256 | Changed argmax rows | Mean entropy | Actual native hit |
|---:|---:|---:|---:|:---|
| 7 | 255 | 1 | 0.00241817 | false |
| 8 | 256 | 1 | 0.00278626 | false |
| 9 | 256 | 0 | 0.00006604 | true |

Off verified after sample 8. On paid for the extra denoising forward and
then the ordered terminal forward; total request F was 20 vs 21. #0 and #20
never reached a non-native all-accepted exit; their fresh off/on receipts
and traces matched exactly. `anchor_ticks.csv` includes all three anchors
and the real native criterion series for #10. For other requests, native
output and bills come from sealed same-stack visibility triplets, not a new
native run or the older W1 stack.

## Accounting and scope

Every decoder forward is checked at 85 projection modules for width 256.
All emissions are reconstructed from ordered physical verify argmax; all
post-prefill encoder inputs are checked against the verified block. The
final-draft/verified comparison is truncated at each path's first EOS;
it is a diagnostic, never an alternative output or permission to skip work.
When reported, a first mismatch "at verify" means all previous delivered
blocks matched native and the current last denoising draft, but not the
current verified delivery, matched native's block. It does not claim exact
native per-layer traces outside the probed anchors.

`pairs.csv` records one row per paired request and binding hashes. Macro TPF
is mean per-request N/all post-prefill F. Canonical TE is pooled N/selected
history-reading decoder Q, excluding the encoder. The 95% intervals are
paired dataset-stratified resamples of this frozen cohort, not independent
replications. No inference about sparse Q, two-block residence or live
budget switching is supported by this run.

Raw receipts: `/gs/results/gemma-predicate-only-a1-1128` on A800. Offline
compact reproduction: `bash /gs/runtime/gemma-predicate-only-a1-1128/report.sh`.
`registration.md` and `launch.sh` are the prelaunch contract and actual
launcher. The prior compact was rebuilt and all four SHA256s checked before
model load. No holder, public code or index edits, CPU model, commit or push.

With enough budget to measure the entire current block, an unstable argmax
can justify another denoising tick for fidelity; ordered physical verification
and any required encoder still have to be paid. With a sparse budget, pay
dirty changes and measure unresolved positions before optional clean-prefix
refresh. Never manufacture a full-head stopping observation out of old logits.
The Q32 verified-EOS study gives a different, exact saving: 70/90 requests
skipped one or more *later* whole verify chunks after an ordered, measured EOS
with debt clear, saving 282 F / 9,024 decoder Q and changing no output or
encoder input. This Q256 experiment does not establish a sparse stopping rule.

## Frozen90 decision

Use as a semantic diagnostic or explicit fixed-Q256 native-stop option, **not
as the serving default**. The #10 clock mismatch was explained and corrected;
the broader speed-quality tradeoff does not justify changing the endpoint.
90/90 controls matched the prior continuous-on result and its RNG/event tape.
36/90 cases had an actionable first all-accepted/non-hit; on all 36 the
entropy was already below threshold and the argmax was not stable. The 54
unexposed pairs were exact whole-row no-ops. Only 15/90 changed delivered IDs.

| Cohort | Exposed | Correct off/on | Macro TPF off/on | Canonical TE off/on | Exact IDs vs native off/on |
|---|---:|---:|---:|---:|---:|
| Original 30 | 8 | 29/29 | 23.415/23.120 | 8.577%/8.626% | 23/23 |
| New 60 | 28 | 56/56 | 23.595/21.935 | 8.969%/8.410% | 45/48 |
| Frozen 90 | 36 | 85/85 | 23.535/22.330 | 8.840%/8.478% | 68/71 |

Eight newly exact native deliveries, five formerly exact lost; paired exact-ID
delta +3.33 pp [−4.44,+11.11]. First-block exactness fell from 79 to 77.
No rescues or losses on the scorer. Macro TPF delta −1.205 [−1.777,−0.704].
Same-stack native is 84/90 and macro 25.829. Differing generated lengths
matter (31,027 to 30,992 nonspecial tokens); no wall-clock claim.

Off/on: 1,449/1,506 generation F = 1,203/1,260 denoises + 168/168 verifies
+ 78/78 post-prefill encodes. Total selected decoder Q 350,976/365,568.
Original30 saved 4 F on net, new60 spent 61 more. The aggregate 57 extra F
are not "one extra tick per exposure": continuations can alter later RNG,
stop times and trajectories. Candidate closed all 168 blocks by a real
native hit, with no exhaustion. Off closed 128 by native hit and 40 blocks
by independent all-accepted before a native hit.

Exactly 19 on requests still differed from native. On **all 19**, at the
first mismatching delivery block the last denoising draft up through EOS
already matched native delivery but strict verification changed it. Across
all on deliveries, last draft and verified block differed on 20/168 blocks,
including 5/78 blocks that were sent to the encoder. This isolates an output
boundary, **not** a license to bypass verification, nor proof of native
per-layer identity on every request. The next engineering question concerns
that contract, not tuning entropy or relaxing dirty accounting.

Independent worker 272603 ran the two real arms on GPU1 plus one fresh native
replay of #10 (181 requests total). Exit 0; 2026-11-28 10:53:57 +08 physical
A1 was 0 MiB with no compute PID. Four no-model tests passed locally and on
A800. All raw inputs/tokens/tapes remain remote. No continuation was queued.

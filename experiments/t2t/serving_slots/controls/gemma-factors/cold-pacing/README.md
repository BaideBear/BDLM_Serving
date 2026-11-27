# Gemma cold pacing

Registered against `b068a78`, fixed physical C32/C64/C128 only. Off replays
`fixed90_fused`; on uses the same executor except that while the current head
still has never-sampled rows, a call samples at most C/2 **new** rows, in
baseline order, leaving spare seats empty. The mandatory rows and new rows
are projected together, but only the latter's measured logits are sampled.
There is one denoising clock increment per nonempty sample, zero on payment
or verification alone. All actual token changes and acceptance flips still
become dirty. No paid-mask reuse, rejected-noise retention, cross-block work,
fullstop, or C=B change. The right-edge prefix is 1/4 at C128, 1/8 otherwise;
the guarded verification fusion is identical in both arms.

This is not an algebraic scheduling no-op. The first call on an empty block
projects C rows when off and C/2 when on. The on arm completes the cold
phase in exactly 16/8/4 sampler ticks per head at C32/64/128. This is below
the 48-tick clock limit; however, the sampler enters the warm phase at a
different remaining temperature step. Row cohort size, cross-row attention,
RNG assignment, history, output length, and correctness can consequently
diverge. Identical seed and sampling rule do not imply bitwise paired outputs.

## Evidence

Only A800 GPU6, no CPU model. The first three frozen indices 0/10/20 are part
of, not additional to, the full 90. At every width, candidate and control
must have matching registration. The off arm must match the fused historical
reference including output ID digest, score, encoder/decoder F/Q and route bills
for **every** question. The production auditor instruments the decoder input
positions, the output head and every Q/K/V projector (85 modules, 30 layers);
it also checks the paid encoder tape against verified head tokens. A second
replay checks cold coverage, old and new debt, actual phase widths, one or
zero sampler steps per call, and full-width terminal coverage per block.

- `pairs.csv`: one matched question/width row; both arms, physical phase F/Q,
  head counts, scores, output equality and qualification fields.
- `blocks.csv`: first decoder tick, first physical width, tick of full
  initialization and sampler age at that point, per head and arm.
- `cohorts.csv`: independently matched original30 (0-29), new60 (30-89) and
  their union frozen90. Each has equal counts across GSM8K, MATH500 and
  HumanEval. Macro TPF is mean non-special output tokens / post-prefill F per
  question, not wall-clock throughput. Intervals are 4,000 within-dataset paired
  bootstrap draws. They are descriptive, not multiple-comparison corrected.
- `provenance.json`: hashes and model/software/device registration.

Cold phase includes the tick that first finishes initialization; paying its
remaining writes on the next tick is charged to the next phase. The split
is determined independently on each trajectory. `cold_Q = first_Q + cold_paid_Q`.
`overlap_F` counts cold calls that both pay old debt and sample new rows.
`warm` includes both warm sampling and nonterminal drains; `closing_drain`
is separate. `verify` includes plain and paid-first-chunk verification.
The four disjoint phase bills sum to **decoder** F/Q; post-prefill encoders
are separately charged. Changing the number of generated blocks can affect
all totals. `cold_idle` counts unfilled physical offer seats during cold ticks.

## Reproduce the compact report

On A800, from the registered runtime snapshot:

```sh
cd /gs/runtime/cold-pacing-1128/repo
/usr/bin/python3 -B -m experiments.t2t.serving_slots.gemma_cold_pacing_report \
  --root /gs/results/cold-pacing-1128 --history /gs/results \
  --out /gs/runtime/cold-pacing-1128/compact
```

Omit `--out` for provisional matched counts; compact export fails closed unless
all 270 pairs exist and pass. Historical raw files stay on the receipt host.

## Results and decision

**Keep pacing off at all three widths in the fixed profile.** The candidate
truly overlaps first visits and debt, but does not produce a convincing
whole-request TPF/quality improvement. C32 sacrifices quality; C64 is mixed;
C128 loses macro TPF in both cohorts and loses three correct scores overall.

| Cohort | C | Matched | Correct off/on | Total paid F off/on | Macro TPF off/on | Paired delta TPF (95%) |
|---|---:|---:|---:|---:|---:|---:|
| Original 30 | 32 | 30 | 29/26 | 2351/2392 | 4.607/4.458 | -0.149 [-0.369,+0.061] |
| Original 30 | 64 | 30 | 29/29 | 1363/1416 | 7.742/7.791 | +0.049 [-0.260,+0.367] |
| Original 30 | 128 | 30 | 30/29 | 814/973 | 12.212/11.869 | -0.343 [-1.075,+0.422] |
| Added 60 | 32 | 60 | 56/51 | 4805/5434 | 4.701/4.848 | +0.147 [-0.027,+0.312] |
| Added 60 | 64 | 60 | 54/55 | 2727/2703 | 7.950/7.862 | -0.089 [-0.458,+0.274] |
| Added 60 | 128 | 60 | 58/56 | 1616/1946 | 13.086/11.977 | -1.109 [-1.880,-0.349] |
| Frozen 90 | 32 | 90 | 85/77 | 7156/7826 | 4.669/4.718 | +0.048 [-0.097,+0.190] |
| Frozen 90 | 64 | 90 | 83/84 | 4090/4119 | 7.881/7.838 | -0.043 [-0.318,+0.223] |
| Frozen 90 | 128 | 90 | 88/85 | 2430/2919 | 12.794/11.941 | -0.853 [-1.421,-0.273] |

GPU6 produced 540 arm receipts. All 270 off/on pairs passed; all 270 off
receipts replayed the signed-off historical controls. There were no missing
questions, failed audits or initialization starvation. At C32/C64/C128,
84/86/87 of 90 outputs changed; scorer rescues/losses were 3/11, 5/4 and
0/3. Thus C32's small positive TPF point estimate is not a free gain:
it produced 34,874 instead of 32,774 non-special tokens, with more paid F
and net eight fewer correct. C64's one net rescue does not establish a
reliable quality gain (paired 95% accuracy delta [-0.056,+0.078]).

| C | Finished heads off/on | Cold F/head off/on | Sampler ticks to fully warm off/on | Pure cold drain F off/on | Overlap F off/on | Warm F off/on | Terminal drain + verify F off/on | Encoder F off/on |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 170/181 | 15/16 | 8/16 | 1190/0 | 0/2715 | 3001/3217 | 1525/1622 | 80/91 |
| 64 | 172/163 | 7/8 | 4/8 | 516/0 | 0/1141 | 1967/1942 | 837/800 | 82/73 |
| 128 | 164/170 | 3/4 | 2/4 | 164/0 | 0/510 | 1415/1696 | 449/463 | 74/80 |

Every candidate head finished initialization: 514/514. The matched off
trajectories generated 506 heads. In each arm and width every head exhibited
the cold F/head and sampler tick counts shown above. The pipeline eliminated
pure cold drains but spent **one additional cold forward per generated
head**; the sampler used twice as many temperature steps before warming.
Mandatory debt, complete head verification and encoder writes were not
skipped. At C128 the warm phase also cost 281 more calls. With scarce budget,
keep mandatory debt and head progress first; optional slots being idle is
preferable to using stale logits, but forcing idle/half batches is not an
effective default here.

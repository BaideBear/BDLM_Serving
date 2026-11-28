# Does a reserved seat need to project a prefix row?

Companion to the [prefix-dose screen](README.md#paired-calibration-separate-from-formal90). Not a formal90 arm. Background: in the three-case H0 Gemma Q32 pilot, increasing the fraction from zero to one eighth reduced drain-only decoder calls from 41.1% to 4.2%. Changing the fraction changes both the number of MASK rows and the treatment of clean history. The earlier pilot alone cannot identify which mattered.

Status (2026-11-28): [registration](mechanism18-prefix.json) `c6275852e01e1117d656bbd2ed7fa8a117ed137a37ca0e77cde4e98ce329d015`. The H0 Gemma qualification below is complete. It is not the 18-case mechanism panel or the formal90 comparison.

## Paired intervention

Each request keeps one offer throughout. Tight offers `B/8, B/4` are LLaDA 4/8 and Gemma 32/64. Use the same 18 frozen screen IDs (three per board), models, seeds, checkpoint-specific sampler and scoring as the formal anchor. `open_at=1/4`; the nonzero clean prefix fraction is `1/8` of budget after outstanding writes. Rotate the arm order within each paired case. A complete panel has 18 cases × two offers × three arms × two models = 216 requests. A first three-case board is a qualification, not the full panel.

| Arm | On an identical incoming state |
|---|---|
| `all_mask` | Zero clean prefix allocation. Debt first; all otherwise eligible seats go to MASK. |
| `reserve_idle` | Compute the ordinary one-eighth selection, including its MASK choices, but physically omit the clean prefix rows and do not backfill. Their existing KV stays resident. |
| `prefix` | Same one-eighth selection; physically project its clean prefix rows and allow the native sampler to edit them. |

Compare `reserve_idle - all_mask` for the effect of withholding MASK seats, and `prefix - reserve_idle` for the incremental effect of processing the prefix. The latter includes KV refresh, native edit opportunities, changes to random-number consumption and later trajectories. **It does not isolate KV freshness alone.** After divergence these are paired request outcomes, not identical step-by-step schedules. A positive effect only in `reserve_idle` motivates testing pacing; a positive increment only in `prefix` motivates a further projection-without-edit arm. Neither makes a new default.

Require common seed, board, case, state-dependent prefix identities and MASK allotment at the first eligible step; compare the preceding ordinary-plan digest. Record how many seats were reserved and how many MASKs were actually displaced: a spare prefix row need not displace anything. Report physical decoder Q, macro TPF, pooled TE, paired correctness, drain F, prefix edits and revocations, encoder F and final output digests. At LLaDA Q4 the one-eighth floor is always zero: all three arms must replay the same unexposed plan, output and bill. A native terminal verification, build or writeback is never skipped to make an arm look cheaper.

Use `python -m tools.shared_prefix_mechanism register --base formal90-anchor.json` to freeze the wrapper against the existing worker registration. Run `preflight` and then `run` with `--base`, `--version`, `--board`, `--model`, `--samples` and `--checkpoint`. The wrapper temporarily replaces only the shared row-selector function in a serial experiment process. It also reads already measured sampling changes; native projection, masked attention and score computation are unchanged. Raw logs stay on n164. Complete verified triplets alone reach the study ledger; an error aborts the probe and does not trigger config changes for that case.

## H0 qualification: three paired Gemma cases

Source: n164 `results/mechanism18-c6275852-H0/gemma.ndjson`, 18 receipts, November 28, 16:23 CST (UTC+8). TPF is the mean of request-level N/F. TE is pooled N over physical decoder Q reading history. T2T is the sum of observed prefix edits, not extra output tokens. These are the same three H0 identities used in the dose calibration; they are not independent confirmation.

| Fixed Q | Arm | Correct | TPF | TE | Prefix T2T |
|---:|---|---:|---:|---:|---:|
| 32 | all_mask | 3/3 | 3.599 | 11.17% | 0 |
| 32 | reserve_idle | 3/3 | 4.257 | 14.13% | 0 |
| 32 | prefix | 3/3 | 4.464 | 14.24% | 8 |
| 64 | all_mask | 2/3 | 5.987 | 9.55% | 0 |
| 64 | reserve_idle | 2/3 | 6.176 | 10.66% | 0 |
| 64 | prefix | 1/3 | 6.282 | 10.15% | 6 |

At Q32 much of the observed TPF difference from `all_mask` was already present when prefix rows were withheld entirely. At Q64 enabling the prefix cost an observed correct answer relative to the idle arm. Neither three-case comparison establishes an optimal fraction or identifies a benefit of KV freshness alone.

The other half of the same qualification, LLaDA, completed at 16:28 UTC+8.
Q4 had identical selected-row histories, output digests and bills in all
three arms on every case. Q8 had no observed prefix T2T in these three
requests despite projecting eight clean prefix rows in the `prefix` arm.

| Fixed Q | Arm | Correct | TPF | TE | Prefix T2T |
|---:|---|---:|---:|---:|---:|
| 4 | all_mask | 2/3 | 1.489 | 36.64% | 0 |
| 4 | reserve_idle | 2/3 | 1.489 | 36.64% | 0 |
| 4 | prefix | 2/3 | 1.489 | 36.64% | 0 |
| 8 | all_mask | 2/3 | 2.592 | 28.49% | 0 |
| 8 | reserve_idle | 2/3 | 2.563 | 28.65% | 0 |
| 8 | prefix | 3/3 | 2.471 | 28.20% | 0 |

`python -m tools.shared_mechanism_report BASE_REG RESULTS --board n164:H0
--require-complete` rechecks the entire qualification. Omit `--board` to
audit the 216-request panel; a three-case qualification must not be released
as an 18-case score. H0 is linked into the n164 panel directory so those
specific requests are counted once only.

# Does a reserved seat need to project a prefix row?

Companion to the [prefix-dose screen](README.md#paired-calibration-separate-from-formal90). Not a formal90 arm. Background: in the three-case H0 Gemma Q32 pilot, increasing the fraction from zero to one eighth reduced drain-only decoder calls from 41.1% to 4.2%. Changing the fraction changes both the number of MASK rows and the treatment of clean history. The earlier pilot alone cannot identify which mattered.

Status (2026-11-28): [registration](mechanism18-prefix.json) `c6275852e01e1117d656bbd2ed7fa8a117ed137a37ca0e77cde4e98ce329d015`. This freezes the wrapper for the three-case GPU qualification; it does not certify either adapter on GPU or supply any experiment results. Run after higher-priority registered work.

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

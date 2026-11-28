# Shared row planner: calibration and status

[Method and comparison protocol](../../../../../docs/shared-request-planner.md) |
[Per-checkpoint search space](search-space.json) |
[First formal anchor registration (not results)](formal90-anchor.json) |
[Frozen control measurements](../fixed-request-comparison/README.md).

These are **engineering smoke tests only** on the same `gsm8k/167`, index 0,
seed 17000. Neither model has a completed frozen90 point for this planner.
Do not pool these runs, compare budgets on this basis, or use them as a
version's confirmation set. Both models' selected parameters remain unset.

| Model | Q offer | Device | Final nonspecial N | Post-prefill F | Selected decoder Q | Receipt SHA-256 |
|---|---:|---|---:|---:|---:|---|
| LLaDA 2.1 | 16 | A2 | 332 | 115 | 1709 | `c370f72882b0db6bf69a16ea628203f2b10fd73255ea4bbf79bb6842e4b1ec97` |
| DiffusionGemma | 64 | A3 | 414 | 47 | 2844 | `b410caabf196e4c7d541068e8ab8bb6a8c78cda3953207388cac5097c4c6960d` |
| DiffusionGemma | 128 | A1 | 338 | 37 | 4497 | `a5ff7a01d14e8a2ccc79af352bf7615d58adc229ff5143a7ab0a03d261a9b9eb` |
| DiffusionGemma | 256 | H0 | 354 | 32 | 5860 | `caa2695bfbffd31ff3437dfba0f4996e9ed7343e89c865b97d746da775a57fe4` |

All four completions scored correct on that one input. Gemma's F includes
one post-prefill encoder forward per run. **LLaDA Q32 on H4 failed** at the
terminal successor check: no valid completion, and no quality/TPF/TE score.
The adapter change needs a GPU retest on this same input before freezing it.
Failure log SHA-256: `2deaff5bbb8f011432a98b45887c4f07b255d44c600bf72432a332da354b51df`.
In the renewed November 28 window, the same H4 Q32 input after the EOS fix
completed correctly with N=265, F=45, Q=1181. Q4 on the same input completed
incorrectly (N=233, F=199, Q=785). These are qualifications, not formal90 rows.
The initial formal anchor is `prefix_share=1/8, open_at=1/4` on both models;
this is a reference for independent tuning, not an optimized setting.

## Operational comparison

`shared_fixed_shard register --stage formal90 --llada21 ALPHA TAU --gemma ALPHA TAU`
(invoke with `python -m experiments.t2t.serving_slots.shared_fixed_shard`)
prints the parameter/source registration to stdout and its ID to stderr. Freeze
the registration before looking at formal scores. Transfer its exact source
revision. Use `run --registration FILE --version ID --model MODEL --board BOARD
--samples FROZEN_DATA --checkpoint CHECKPOINT --preflight` to check inputs
without touching a GPU. `run` without `--preflight` requires a single correct
unoccupied GPU exposed by UUID, loads weights once, and writes JSONL to stdout.
External launchers own exclusive handoff, deadline and n164 storage. For LLaDA,
provide `BDLM_LLADA_REFERENCE`; `tools.pack_shared_prompts` can stream just the 90
registered prompt-ID records to private tmpfs on A800, without copying outputs.

`screen18` selects one case per dataset on each of the six boards. Its
registration differs from `formal90`; do not stitch its outcomes into the formal
table. Checkpoint comparisons use the manifest in `benchmarks/checkpoints/`;
the supplemental LLaDA SHA-256 list was verified byte-for-byte independently
on n164 and A800. No generated text is stored in these registrations.

### Paired calibration (separate from formal90)

`python -m tools.shared_screen_sweep register --base formal90-anchor.json --axis prefix`
pins the base workers and the sweep wrapper. Each board runs three specified
inputs from its 15-case shard, at all five fixed offers and four prefix fractions.
The order rotates; a request never changes offer, arm or seed. There are 18
distinct inputs and 360 requests per model (720 across both). The earlier
three-input H0 pilot is exploratory and is not appended to this ledger.
Run the `open` axis only after locking each model's own prefix choice. The
inputs used to choose a fraction must be identified separately from the
remaining 72 of the frozen90 comparison.

Selection, fixed before opening the 18-case results: compare against `1/8`
on the prefix axis, or `1/4` on the opening axis. A new constant qualifies
only if no offer loses correct answers or macro TPF on this paired screen,
and some offer improves one of them. If more than one qualifies, maximize
the worst offer's TPF ratio to the control, then the geometric mean of the
five ratios, then total correct, then use the smaller fraction. If none
qualifies, retain the current value; publish per-offer tradeoffs rather
than switching rules by tier. This is a conservative promotion rule for
an exploratory screen, not a statistical non-inferiority test. Physical Q
roles and pooled TE are reported alongside; they do not mask losses.

`python -m tools.shared_anchor_report RESULTS formal90-anchor.json` audits
every complete receipt, including the history-Q partition and measured peak.
It emits only counts for unfinished points and exposes 90/30/60 scores for
complete points. Add `--baselines <fixed-request-comparison/metrics.csv>` for
matched reference differences and `--require-complete` for the final release.

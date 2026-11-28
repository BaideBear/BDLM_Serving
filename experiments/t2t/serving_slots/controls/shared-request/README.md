# Shared row planner: calibration and status

[Method and comparison protocol](../../../../../docs/shared-request-planner.md) |
[Per-checkpoint search space](search-space.json) |
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

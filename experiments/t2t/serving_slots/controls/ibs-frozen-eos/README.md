# Strict IBS: how much work follows an irrevocable EOS?

Historical exposure, **not** measured acceleration. The current upstream-style
control finishes a physical block before trimming at EOS. Its `ready` pointer
already identifies an irrevocable prefix. For successful EOS requests,
locate the first call where `ready` covers the delivered EOS, and check that
no subsequent ordinary call selects an already retired row. In LLaDA the
address includes prompt length; in Gemma it is generated-canvas relative.
The new runtime must inspect only that currently frozen, generated prefix,
not a final-output oracle. Frozen requests that failed or did not deliver EOS
are kept and contribute no evidence of removable work.

| Model | Q | Original F, incl. generation encoder | Requests with later F / 90 | Actual ordinary F after first certificate | Later F if waiting for all debt to clear naturally | Requests with suffix debt at first certificate |
|---|---:|---:|---:|---:|---:|---:|
| LLaDA 2.1 | 4 | 27,397 | 74 | 556 | 394 | 31 |
| LLaDA 2.1 | 8 | 19,692 | 73 | 294 | 228 | 18 |
| LLaDA 2.1 | 16 | 15,480 | 41 | 85 | 58 | 12 |
| LLaDA 2.1 | 32 | 12,694 | 3 | 3 | 0 | 3 |
| DiffusionGemma | 32 | 6,287 | 78 | 1,036 | 42 | 75 |
| DiffusionGemma | 64 | 3,498 | 76 | 390 | 12 | 74 |
| DiffusionGemma | 128 | 2,055 | 60 | 146 | 11 | 55 |
| DiffusionGemma | 256 | 1,563 | 0 | 0 | 0 | 0 |

A prospective implementation can either wait until there is no debt, or
pay one actual decoder projection over the remaining suffix debt without
sampling. These are different policies. Running and billing that final
projection is essential before reporting measured TPF or efficiency. The
debt recorded here is that of each pinned baseline: accepted rows in Gemma
IBS; admissions and eligible edits recorded by LLaDA IBS. Do not extend this
certificate to a native sampler that may revise the proposed prefix. Apply
the same stopping policy to any compatible baseline before attributing a
gain to query selection.

[Evidence, original30/new60 splits, Q counters and raw-source hashes](exposure.json).
Reproduce on n164: `python3 tools/ibs_frozen_eos_tail.py
/data1/project_data/gx-erqw-20261123/runtime/t2t-native-1125/prefix90`.

## Shared stop predicate on the six registered GPU pairs per model

The [pure decision function](../../../../../bdlm_serving/methods/t2t/frozen_stop.py)
consumes the post-write immutable frontier, generated interval, absolute
EOS position, outstanding row set and offer. It neither performs a model
call nor marks anything paid. Only a caller that guarantees the prefix is
immutable may use it. A full block ending in EOS is an ordinary end, not
an early termination. A suffix payment never advances the sampler; do not
finish until a real paid projection has cleared the debt.

| Registered GPU pilot | Stock decoder ticks replayed | Stop with no debt | Stop after suffix-only payment | Ordinary block EOS | Measured payment F / Q |
|---|---:|---:|---:|---:|---:|
| LLaDA Q4, six | 1,814 | 4 | 2 | 0 | 2 / 3 |
| Gemma Q32, six | 289 | 1 | 4 | 1 | 4 / 20 |

The [audit](../../../../../tools/audit_shared_frozen_stop.py) reconstructs the
LLaDA prefix from ordinary writes and revocations, excludes prompt tokens,
and reads Gemma's independently reconciled per-tick states. It checks the
first trigger and any physical payment against the treatment arm. [Compact
output with raw-tape hashes](pilot-replay.json). The pilot replay is **not**
a second inference run or evidence that the Q-selection policy beats IBS.
The in-flight registered GPU executors were not modified.

## Same-device frozen90 results: Gemma Q32, Q64 and Q128

At each fixed decoder cap, the same 90 requests ran in four arms on the
same physical A800 0 (`GPU-d3c3fc9c-b15d-9776-9094-7d3b85b705c1`). Prompts,
seeds, weights and scoring were held fixed. The observation-only arm
matched stock step for step. Within each tier, all four arms returned
identical complete output IDs, per-request scored correctness and output
token counts, with identical generation encoder inputs. Until the stop
certificate, the treatment's ordinary decoder trace matched stock.

| Cap | IBS stop rule | Correct | Macro TPF | Pooled TE | Decoder F (`D`) | Encoder F (`E`) | Total `F` | Actual decoder `Q` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Q32 | Stock / observation only | 78/90 | 6.466 | 18.64% | 6,448 | 101 | 6,549 | 193,583 |
| Q32 | Wait for zero debt | 78/90 | 6.812 | 19.00% | 6,333 | 101 | 6,434 | 189,914 |
| Q32 | Pay live suffix debt and stop | 78/90 | 8.168 | 22.41% | 5,381 | 101 | 5,482 | 160,966 |
| Q64 | Stock / observation only | 76/90 | 10.872 | 17.33% | 3,343 | 85 | 3,428 | 189,137 |
| Q64 | Wait for zero debt | 76/90 | 11.114 | 17.57% | 3,291 | 85 | 3,376 | 186,582 |
| Q64 | Pay live suffix debt and stop | 76/90 | 12.510 | 20.43% | 2,835 | 85 | 2,920 | 160,484 |
| Q128 | Stock / observation only | 80/90 | 16.445 | 14.26% | 2,281 | 86 | 2,367 | 236,085 |
| Q128 | Wait for zero debt | 80/90 | 16.616 | 14.35% | 2,268 | 86 | 2,354 | 234,517 |
| Q128 | Pay live suffix debt and stop | 80/90 | 17.509 | 15.50% | 2,043 | 86 | 2,129 | 217,101 |

TPF is the mean per-request `N/F`; TE is pooled `sum(N)/sum(Q)` over
actual decoder rows. `F=D+E` includes generation encoders and excludes
prompt prefill. All payment forwards and rows are included in `D` and `Q`.
Each payment projects only the currently owed suffix rows once, without
sampling; the policy does not look ahead to decide whether it pays.

| Cap | Output `N` (all arms) | Correct original30 / new60 | Paid suffix `F` / `Q` | Net paid-policy savings vs. paired stock (`F` / `Q`) |
|---|---:|---:|---:|---:|
| Q32 | 36,080 | 27/30, 51/60 | 75 / 470 | 1,067 / 32,617 |
| Q64 | 32,782 | 25/30, 51/60 | 70 / 475 | 508 / 28,653 |
| Q128 | 33,661 | 26/30, 54/60 | 50 / 1,075 | 238 / 18,984 |

Q64 and Q128 each retain five paid requests with zero net forward savings.
Comparisons are paired within cap: the historical Q32 row of 79/90,
6.840 TPF and 19.33% TE was not this device's control. Q256 was not rerun.
These are stronger *IBS stopping baselines*, not gains in query selection,
native T2T, or measured wall-clock speed.

[Figure](measured/frozen90.png) · [Plot data by cohort](measured/cohorts.csv)
· [Plot script](../../../../../tools/plot_certified_ibs.py).

### Sealed receipts and source

| Paired receipt | Compact JSON SHA-256 | Raw run directory under `n164:/data1/project_data/gx-erqw-20261123/results/` | Worker PID / release (November 28, 2026, UTC+8) |
|---|---|---|---|
| [Q32](../gemma-factors/ibs-certified-eos/q32-frozen90-summary.json) | `65fa90a425d095f6ee642be9d30a86b67be1c36e0007bf735945a6ae7eb06345` | `gemma-ibs-certified-eos-a0-20261128T100436Z` | `643239` / 18:58:49 |
| [Q64](../gemma-factors/ibs-certified-eos/q64-frozen90-summary.json) | `d3bdd6dd97235973752df513ef2e3c4832fac41cb1cf54d9115e1e0f75e0344e` | `gemma-ibs-certified-eos-a0-q64-20261128T110532Z` | `710090` / 19:35:40 |
| [Q128](../gemma-factors/ibs-certified-eos/q128-frozen90-summary.json) | `9f3ecbae4bd6a04030ae876c6d1d6f75ca786bacddb2c806ce4ffbb77e5dc1c7` | `gemma-ibs-certified-eos-a0-q128-20261128T110532Z` | `740549` / 19:58:16 |

`cases[].index` pairs requests; `cases[].arms.{stock,control,strict,paid}`
contains the per-arm bill, trigger, payment, `N` and correctness.
`cohorts.{original30,new60,frozen90}` contains the per-arm summaries.
Every `release` records `rc=0`, 0 MiB, no compute apps and closed locks.
The original IDs and full per-step records stay in n164 `raw.log`, verified
against each summary's `raw_sha256`.

Q64 and Q128 share the [registered configuration](../gemma-factors/ibs-certified-eos/q64-q128-registration.md)
and the same Git-tracked [stop hook](../../gemma_ibs_certified_eos.py),
[launcher](../../../../../tools/launch_gemma_ibs_certified_eos.sh),
[runner](../../../../../tools/run_gemma_ibs_certified_eos.py) and
[tests](../../../../../tests/test_gemma_ibs_certified_eos.py). The deployed
source root was `/gs/work/gemma-ibs-certified-eos-a0-q64-q128-20261128T110532Z`
with local base `6afb052bec355291b832864644a02ec203a96e9c`. The base SHA alone
does not identify the deployed overlay. Its six file hashes live in each
compact receipt's `sources_sha256`; `pins` and `optimus` identify the runtime
and upstream dependencies. To verify the overlay and registration from the
Git root, use `jq -rj .sources_sha256 <summary.json> | shasum -a 256 -c`.
Use the [Q32 reconstruction](../gemma-factors/ibs-certified-eos/q32-source.md)
for that tier's deployment bytes, not the live multi-tier source.

## Same-device frozen90 result: LLaDA Q4

The same certificate on n164 H0 preserves all 90 delivered ID sequences
(including six unchanged failed requests), with exactly 37,754 nonspecial
output tokens in both arms. It skips 556 ordinary decoder forwards but
incurs 31 real suffix payments (51 Q), for a **net 525 fewer F**.

| LLaDA strict capped IBS | Correct | Macro TPF | Pooled TE | F | Decoder Q |
|---|---:|---:|---:|---:|---:|
| Stock / observation only | 69/90 | 1.395 | 35.66% | 27,397 | 105,885 |
| Pay suffix debt and stop | 69/90 | 1.445 | 36.35% | 26,872 | 103,853 |

The [multi-tier figure](measured/frozen90.png) uses a dashed stock reference
in each panel. This result strengthens the comparator; it does not validate
any candidate selector or solve the six progress failures.
The compact input for redrawing the LLaDA panel is
`n164:/data1/project_data/gx-erqw-20261123/results/llada-ibs-certified-eos-h0-v1/receipts.ndjson`
(SHA-256 `fce9fdaa933815010cee6b5009de2bdff2de535e07ee64e96ab80b52ea3382a4`).

From the repository root, redraw with Matplotlib (the archived figure used
3.10.0) and a verified local copy of that NDJSON:

```bash
d=experiments/t2t/serving_slots/controls
gemma="$d/gemma-factors/ibs-certified-eos"
python3 tools/plot_certified_ibs.py \
    --gemma "$gemma/q32-frozen90-summary.json" \
    --gemma "$gemma/q64-frozen90-summary.json" \
    --gemma "$gemma/q128-frozen90-summary.json" \
    --llada /path/to/receipts.ndjson \
    --out "$d/ibs-frozen-eos/measured"
```
